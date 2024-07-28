import ast
import asyncio
import configparser
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import httpx
import numpy as np
import openai
import pandas as pd
import pytz
import tiktoken
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

try:
    from print_handler import PrintUtils
except ModuleNotFoundError:
    from utils.print_handler import PrintUtils

load_dotenv()

# Create config parser
config = configparser.ConfigParser()

# Read config file
config.read("./config.ini")


class EvalDataHandler:
    def __init__(self):
        self.HF_TOKEN = os.getenv("HF_TOKEN")

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    async def _query_api(self, user_query, data_type, num_search_results):
        url = "http://localhost:6000/api/query"
        headers = {"Content-Type": "application/json"}
        data = {
            "query": user_query,
            "data_type": data_type,
            "num_search_results": num_search_results,
        }

        try:
            async with httpx.AsyncClient(timeout=240.0) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                return response.json()
        except httpx.ReadTimeout:
            # Handle the timeout exception
            print("The request timed out. Retrying...")
            raise
        except httpx.RequestError as exc:
            # Handle other request exceptions
            print(f"An error occurred while making the request: {exc}")
            raise

    def get_current_timestamp(self):
        eastern = pytz.timezone("US/Eastern")
        now = datetime.now(eastern)
        timestamp = now.strftime("%m/%d/%Y, %I:%M %p %Z")
        return timestamp

    def _stage_data(
        self,
        dataset_name,
        num_samples,
        random_state: int = int(config.get("EVAL", "random_state")),
    ):
        dataset = load_dataset(
            dataset_name, token=self.HF_TOKEN, split="train"
        ).to_pandas()
        return dataset.sample(num_samples, random_state=random_state)

    def _get_data(self, data_type: str, num_samples: int):
        allowed_data_types = ["clinical_phrase", "icd_10_cm", "cpt"]
        if data_type not in allowed_data_types:
            raise ValueError(
                f"Invalid data_type '{data_type}'. Please choose from {allowed_data_types}."
            )

        if data_type == "clinical_phrase":

            def fetch_response(query):
                return asyncio.run(self._query_api(query, data_type, 5))

            HF_REPO = config.get("HUGGINGFACE", "repo_name")
            eval_data = self._stage_data(HF_REPO, num_samples).reset_index(drop=True)

            search_queries = [
                eval_data.loc[idx, config.get("EVAL", "column_name")]
                for idx in range(num_samples)
            ]
            icd_codes = [eval_data.loc[idx, "icd_code"] for idx in range(num_samples)]
            cpt_codes = [eval_data.loc[idx, "cpt_code"] for idx in range(num_samples)]

            responses = [None] * num_samples
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(fetch_response, query): idx
                    for idx, query in enumerate(search_queries)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    responses[idx] = future.result()

            response_data = [
                (query, response[0]["content"], icd_code, cpt_code)
                for query, response, icd_code, cpt_code in zip(
                    search_queries, responses, icd_codes, cpt_codes
                )
            ]

            response_dataset = pd.DataFrame(
                response_data,
                columns=["user_query", "model_response", "gt_icd", "gt_cpt"],
            )
            response_dataset.to_csv(f"./data/{data_type}_evaluations.csv", index=False)

        elif data_type == "icd_10_cm":
            return self._stage_data(config.get("EVAL, icd_dataset"), num_samples)

        elif data_type == "cpt":
            return self._stage_data(config.get("EVAL", "cpt_dataset"), num_samples)

        else:
            return None


class EvalHandler(EvalDataHandler):
    def __init__(self):
        super().__init__()
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @staticmethod
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _get_openai_completion(
        prompt: str, model_name=config.get("EVAL", "openai_model")
    ):
        response = EvalHandler.openai_client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.model_dump_json())

    def _process_row(self, index, row, prompt):
        user_query = row["user_query"]
        icd_code, cpt_code = row["gt_icd"], row["gt_cpt"]
        prompt_template = prompt.get("template").format(
            user_query=user_query, icd_labels=icd_code, cpt_labels=cpt_code
        )
        completion = (
            EvalHandler.get_openai_completion(prompt_template)
            .get("choices")[0]
            .get("message")
            .get("content")
        )
        return index, completion

    def _generator(self, file_name: str):

        eval_df = Evaluator().run_evaluation(
            file_name, "./prompts/phrase_cleaner.yml"
        )
        icd_stats = Evaluator.stats(eval_df, code_type="icd10")
        cpt_stats = Evaluator.stats(eval_df, code_type="cpt")
        icd_summary_dict = Evaluator.summarize_values(
            eval_df["icd_eval_result"].to_list()
        )
        cpt_summary_dict = Evaluator.summarize_values(
            eval_df["cpt_eval_result"].to_list()
        )
        return icd_stats, icd_summary_dict, cpt_stats, cpt_summary_dict

    async def _evaluator(self, data_type, num_samples):
        async def fetch_response(code):
            return await self._query_api(code, data_type, 5)

        if data_type == "clinical_phrase":
            print("Generating Data...")
            self._get_data(data_type, num_samples)
            print("Data generated successfully! 🚀")

            data_path = "./data/clinical_phrase_evaluations.csv"
            dataset = pd.read_csv(data_path)
            print("Data loaded successfully. Commencing evaluation... 📏")

            # Get timestamp
            today = datetime.now()
            date_string = today.strftime("%m.%d.%Y")

            # Get random state
            random_state = int(config.get("EVAL", "random_state"))

            model_name = config.get("EVAL", "openai_model")
            file_name = f"{model_name}_evaluation_{date_string}_{random_state}.xlsx"

            dataset.to_excel(file_name, index=False)

            stats = self._generator(file_name)
            print(f"Evaluation complete. Here are the resutls:\n")
            PrintUtils.print_evaluation_results(dataset, stats, config)

        elif data_type in ["cpt", "icd_10_cm"]:
            eval_data = self._get_data(data_type, num_samples).reset_index(drop=True)
            tasks = [fetch_response(code.strip()) for code in eval_data["code_title"]]
            responses = await asyncio.gather(*tasks)

            match_count = 0

            # Logging
            timestamp = self.get_current_timestamp()
            print(
                f"== Evaluation Record: Data Type - {data_type}\nTimestamp - {timestamp} =="
            )

            for idx, response in enumerate(responses):
                codes = [
                    data_dict.get("metadata").get("code") for data_dict in response
                ]

                if eval_data.loc[idx, "code_title"] in codes:
                    match_count += 1
                    print(f"Match found for code: {eval_data.loc[idx, 'code_title']}")
                else:
                    print(
                        f"Match not found for code: {eval_data.loc[idx, 'code_title']}"
                    )

            print(f"Match count: {match_count}/{num_samples}")
            print(f"Accuracy: {(match_count / num_samples) * 100}%")
            return f"Match count: {match_count}/{num_samples}"

    async def _run_icd_eval(self, data_type, num_samples):
        if data_type != "icd_10_cm":
            return "Invalid data type. Please choose 'icd_10_cm'."

        icd_eval_results = await self._evaluator(data_type, num_samples)
        return icd_eval_results

    async def _run_cpt_eval(self, data_type, num_samples):
        if data_type != "cpt":
            return "Invalid data type. Please choose 'cpt'."

        cpt_eval_results = await self._evaluator(data_type, num_samples)
        return cpt_eval_results

    async def _run_clinical_phrase_eval(self, data_type, num_samples):
        if data_type != "clinical_phrase":
            return "Invalid data type. Please choose 'clinical_phrase'."

        clinical_phrase_eval_results = await self._evaluator(data_type, num_samples)
        return clinical_phrase_eval_results

    async def run_eval(self, data_type, num_samples):
        if data_type not in ["clinical_phrase", "icd_10_cm", "cpt"]:
            return "Invalid data type. Please choose from 'clinical_phrase', 'icd_10_cm', or 'cpt'."

        elif data_type == "icd_10_cm":
            result = await self._run_icd_eval(data_type, num_samples)
            return result

        elif data_type == "cpt":
            result = await self._run_cpt_eval(data_type, num_samples)
            return result

        elif data_type == "clinical_phrase":
            result = await self._run_clinical_phrase_eval(data_type, num_samples)


class ICDEvalUtils:
    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def evaluate_icd10_predictions(
        ground_truth, predictions, method="weighted", partial_weight=0.5, **kwargs
    ):
        def get_code_parts(code):
            return code.split(".")

        results = {
            "exact_matches": 0,
            "partial_matches": 0,
            "over_suggestions": 0,
            "under_suggestions": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

        gt_set = set(ground_truth)
        pred_set = set(predictions)

        exact_matches = gt_set.intersection(pred_set)
        results["exact_matches"] = len(exact_matches)

        gt_set -= exact_matches
        pred_set -= exact_matches

        for gt_code in list(gt_set):
            gt_parts = get_code_parts(gt_code)
            for pred_code in list(pred_set):
                pred_parts = get_code_parts(pred_code)
                if gt_parts[0] == pred_parts[0]:
                    results["partial_matches"] += 1
                    gt_set.remove(gt_code)
                    pred_set.remove(pred_code)
                    break

        results["under_suggestions"] = len(gt_set)
        results["over_suggestions"] = len(pred_set)

        total_predictions = len(predictions)
        total_ground_truth = len(ground_truth)

        if method == "weighted":
            # Weighted Partial Matches
            partial_weight = partial_weight
            correct_predictions = results["exact_matches"] + (
                partial_weight * results["partial_matches"]
            )

            if total_predictions > 0:
                results["precision"] = correct_predictions / total_predictions
            if total_ground_truth > 0:
                results["recall"] = correct_predictions / total_ground_truth

        # Calculate F1 score
        if results["precision"] + results["recall"] > 0:
            results["f1_score"] = (
                2
                * (results["precision"] * results["recall"])
                / (results["precision"] + results["recall"])
            )

        return results

    def calculate_icd10_score(self, ground_truth, predictions, **kwargs):
        # Default weights
        default_weights = {
            "exact_match": 1.0,
            "partial_match": 0.5,
            "over_suggestion": -0.2,
            "under_suggestion": -0.3,
        }

        # Update default weights with any provided weights
        weights = {**default_weights, **kwargs}

        # Get the evaluation results
        results = self.evaluate_icd10_predictions(ground_truth, predictions)

        # Calculate the score
        score = (
            weights["exact_match"] * results["exact_matches"]
            + weights["partial_match"] * results["partial_matches"]
            + weights["over_suggestion"] * results["over_suggestions"]
            + weights["under_suggestion"] * results["under_suggestions"]
        )

        # Calculate the maximum possible score (if all were exact matches)
        max_score = weights["exact_match"] * len(ground_truth)

        # Normalize the score to be between 0 and 1
        normalized_score = "{:.3f}".format(max(0, min(score / max_score, 1)))

        return {
            "raw_score": score,
            "normalized_score": normalized_score,
            "evaluation_results": results,
        }


class CPTEvalUtils:
    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def evaluate_cpt_predictions(ground_truth, predictions):
        """
        Evaluate CPT code predictions against ground truth.

        Args:
            ground_truth (list): List of correct CPT codes.
            predictions (list): List of predicted CPT codes.

        Returns:
            dict: A dictionary containing evaluation metrics:
                - 'exact_matches': Number of correctly predicted codes.
                - 'over_suggestions': Number of incorrectly suggested codes.
                - 'under_suggestions': Number of missed correct codes.
                - 'precision': Ratio of correct predictions to total predictions.
                - 'recall': Ratio of correct predictions to total ground truth.
                - 'f1_score': Harmonic mean of precision and recall.
        """
        results = {
            "exact_matches": 0,
            "over_suggestions": 0,
            "under_suggestions": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

        gt_set = set(ground_truth)
        pred_set = set(predictions)

        exact_matches = gt_set.intersection(pred_set)
        results["exact_matches"] = len(exact_matches)

        gt_set -= exact_matches
        pred_set -= exact_matches

        results["under_suggestions"] = len(gt_set)
        results["over_suggestions"] = len(pred_set)

        total_predictions = len(predictions)
        total_ground_truth = len(ground_truth)

        if total_predictions > 0:
            results["precision"] = results["exact_matches"] / total_predictions

        if total_ground_truth > 0:
            results["recall"] = results["exact_matches"] / total_ground_truth

        if results["precision"] + results["recall"] > 0:
            results["f1_score"] = (
                2
                * (results["precision"] * results["recall"])
                / (results["precision"] + results["recall"])
            )

        return results

    @staticmethod
    def calculate_cpt_score(ground_truth, predictions, **kwargs):
        """
        Calculate a score for CPT code predictions based on exact matches, over-suggestions, and under-suggestions.

        Args:
            ground_truth (list): List of correct CPT codes.
            predictions (list): List of predicted CPT codes.
            **kwargs: Optional custom weights for scoring.

        Returns:
            dict: A dictionary containing:
                - 'raw_score': The calculated score before normalization.
                - 'normalized_score': The score normalized to a 0-1 range.
                - 'evaluation_results': Detailed evaluation metrics.
        """
        default_weights = {
            "exact_match": 1.0,
            "over_suggestion": -0.3,
            "under_suggestion": -0.1,
        }

        weights = {**default_weights, **kwargs}

        results = CPTEvalUtils.evaluate_cpt_predictions(ground_truth, predictions)

        score = (
            weights["exact_match"] * results["exact_matches"]
            + weights["over_suggestion"] * results["over_suggestions"]
            + weights["under_suggestion"] * results["under_suggestions"]
        )

        max_score = weights["exact_match"] * len(ground_truth)

        normalized_score = "{:.3f}".format(max(0, min(score / max_score, 1)))

        return {
            "raw_score": score,
            "normalized_score": float(normalized_score),
            "evaluation_results": results,
        }


class EvalUtils(CPTEvalUtils, ICDEvalUtils):
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_openai_completion(self, prompt: str, model_name: str = "gpt-4o-mini"):
        user_message = {"role": "user", "content": prompt}
        response = EvalHandler().openai_client.chat.completions.create(
            model=model_name, messages=[user_message]
        )
        return json.loads(response.model_dump_json())

    def process_row(self, row, prompt_template):
        query = row.model_response
        prompt = prompt_template.format(query=query)
        response = self.get_openai_completion(prompt)
        return response["choices"][0]["message"]["content"]

    def generate_completions(self, df_path: str, prompt_path: str):
        # Load and preprocess data
        df = pd.read_excel(df_path)
        df["gt_icd"] = df["gt_icd"].apply(ast.literal_eval)
        df["query_length"] = df["user_query"].str.len()
        df["num_codes"] = df["gt_icd"].apply(len)
        df["categories"] = df["gt_icd"].apply(lambda x: [code[:3] for code in x])

        tokenizer = tiktoken.get_encoding("cl100k_base")
        df["query_tokens"] = df["model_response"].apply(
            lambda x: len(tokenizer.encode(x))
        )

        # Load prompt
        with open(prompt_path, "r") as file:
            prompt = yaml.safe_load(file)

        # Generate completions
        with ThreadPoolExecutor(max_workers=10) as executor:
            completions = list(
                tqdm(
                    executor.map(
                        self.process_row,
                        df.itertuples(),
                        [prompt["template"]] * len(df),
                    ),
                    total=len(df),
                )
            )

        df["cleaned_response"] = completions

        # Evaluate completions
        evaluations = [self.extract_codes(response) for response in completions]
        eval_df = pd.DataFrame(evaluations)

        # Combine results
        result_df = pd.concat([df, eval_df], axis=1)
        result_df["gt_cpt"] = result_df["gt_cpt"]  # .str[:5]

        return result_df

    @staticmethod
    def extract_codes(input_text, model_name="gpt-4o-mini"):
        prompt = """Extract ICD-10 and CPT codes from the given input text "input_text" and
      return them in a dictionary "dict". Only return the dictionary and nothing else!

      Example:
      input_text:
      Here are the codes I've found related to the clinical phrase:

      ICD-10 Codes:
      - D22.9 - Melanocytic nevi, unspecified (Justification: "Multiple benign melanocytic nevi discussed.")
      [Link to Codify](https://www.aapc.com/codes/icd-10-codes/D22.9)

      - D17.9 - Benign neoplasm of uncertain behavior of skin, unspecified (Justification: "Dermatofibroma and senile angioma mentioned, classified under other benign neoplasms of skin.")
      [Link to Codify](https://www.aapc.com/codes/icd-10-codes/D17.9)

      - L65.9 - Non-scarring hair loss, unspecified (Justification: "Discussed diagnosis of female pattern alopecia which is a non-scarring hair loss.")
      [Link to Codify](https://www.aapc.com/codes/icd-10-codes/L65.9)

      CPT Codes:
      - 99203 - Office or other outpatient visit for the evaluation and management of a new patient, which requires at least 3 of these components: a problem focused history, a problem focused examination, and medical decision making of low complexity (Justification: "Patient was evaluated for multiple skin conditions and hair loss, concurrent treatment options discussed.")
      [Link to Codify](https://www.aapc.com/codes/cpt-codes/99203)

      - 11900 - Injection, intralesional; up to 7 lesions (Justification: "If any intralesional treatments were administered for benign neoplasms like dermatofibroma, this would be the appropriate code, but it may not specifically apply if only education and reassurance were provided.")
      [Link to Codify](https://www.aapc.com/codes/cpt-codes/11900)

      Please note that the specific procedure codes may vary based on the actual interventions performed, and in this clinical note, the treatments were more educational and no specific surgical or injection procedures were documented. The CPT codes provided represent common evaluations and treatments that may accompany skin evaluations.

      dict: {{'pred_icd': ['D22.9', 'D17.9', 'L65.9'], 'pred_cpt': ['99203', '11900']}}

      Now it's your turn!

      input_text:
      {input_text}

      dict:"""

        user_message = {"role": "user", "content": prompt.format(input_text=input_text)}
        response = EvalHandler().openai_client.chat.completions.create(
            model=model_name, messages=[user_message]
        )

        return json.loads(
            json.loads(response.model_dump_json())["choices"][0]["message"][
                "content"
            ].replace("'", '"')
        )

    @staticmethod
    def extract_codes_re(input_text):
        icd10_codes = set(re.findall(r"\b([A-Z]\d{2}\.\d{1,4})\b", input_text))
        cpt_codes = set(re.findall(r"\b(\d{5})\b", input_text))
        return {"pred_icd": list(icd10_codes), "pred_cpt": list(cpt_codes)}


class Evaluator(EvalUtils):
    def __init__(self):
        super().__init__()

    @staticmethod
    def truncate_column(df, column_name, max_length=5):
        df[column_name] = df[column_name].str[:max_length]
        return df

    def run_evaluation(self, df_path, prompt_path):
        eval_df = self.generate_completions(df_path, prompt_path)

        eval_df = Evaluator.truncate_column(eval_df, column_name="gt_cpt")

        # Pre-allocate columns
        eval_df["normalized_cpt_score"] = pd.Series(dtype=float)
        eval_df["cpt_eval_result"] = pd.Series(dtype=object)
        eval_df["normalized_icd_score"] = pd.Series(dtype=float)
        eval_df["icd_eval_result"] = pd.Series(dtype=object)

        # Custom weights for ICD scoring
        custom_weights = {
            "exact_match": 1.0,
            "partial_match": 0.7,
            "over_suggestion": -0.1,
            "under_suggestion": -0.5,
        }

        # Combine loops and use vectorized operations where possible
        for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
            cpt_results = CPTEvalUtils.calculate_cpt_score(
                [row["gt_cpt"]], row["pred_cpt"]
            )
            icd_results = self.calculate_icd10_score(
                row["gt_icd"], row["pred_icd"], **custom_weights
            )

            eval_df.loc[idx, ["normalized_cpt_score", "cpt_eval_result"]] = [
                cpt_results["normalized_score"],
                cpt_results["evaluation_results"],
            ]
            eval_df.loc[idx, ["normalized_icd_score", "icd_eval_result"]] = [
                icd_results["normalized_score"],
                icd_results["evaluation_results"],
            ]

        return eval_df

    @staticmethod
    def stats(eval_df, code_type="icd10") -> dict[str,]:
        if code_type == "icd10":
            normalized_scores = [
                float(x) for x in eval_df["normalized_icd_score"].to_list()
            ]
        elif code_type == "cpt":
            normalized_scores = [
                float(x) for x in eval_df["normalized_cpt_score"].to_list()
            ]
        return {
            "min": min(normalized_scores),
            "max": max(normalized_scores),
            "mean": np.mean(normalized_scores),
            "median": np.median(normalized_scores),
            "std_dev": np.std(normalized_scores),
            "quartiles": np.percentile(normalized_scores, [25, 50, 75]),
        }

    @staticmethod
    def summarize_values(data):
        summary = {}
        for key in data[0].keys():
            if isinstance(data[0][key], (int, float)):
                summary[key] = sum(item[key] for item in data)
        return summary


if __name__ == "__main__":
    DATA_TYPE = "clinical_phrase"

    NUM_RESULTS = int(config.get("EVAL", "samples_to_eval"))

    eval_handler = EvalHandler()
    eval_results = asyncio.run(
        eval_handler.run_eval(DATA_TYPE, NUM_RESULTS)
    )
    print(eval_results)
