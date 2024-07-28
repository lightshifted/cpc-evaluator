import configparser
import json
import os
from typing import Any, Dict

import openai
import yaml
from dotenv import load_dotenv

load_dotenv()


class QueryHandler:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read("./config.ini")
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_openai_completion(
        self,
        prompt: str,
    ) -> Dict[str, Any]:
        model_name = self.config.get("LLM", "openai_model")
        response = self.openai_client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.model_dump_json())

    def _load_prompt(self, prompt_path: str) -> Dict[str, Any]:
        with open(prompt_path, "r") as file:
            return yaml.load(file, Loader=yaml.FullLoader)

    def query(
        self,
        query: str,
        provider: str = "openai",
    ):
        if provider == "openai":
            prompt = (
                self._load_prompt(self.config.get("LLM", "clinical_phrase_prompt"))
                .get("template")
                .format(query=query)
            )

            response = (
                self.get_openai_completion(prompt)
                .get("choices")[0]
                .get("message")
                .get("content")
            )

            return [{"content": response}]


if __name__ == "__main__":
    sh = QueryHandler()
    clinical_phrase_example = "She presents for pre-op risk assessment. Scheduled for removal of a lipoma from the right arm. She has no cardiorespiratory symptoms of significance. She has a past medical history significant for a post-op CVA detected 2 weeks after a uterine fibroid embolization in 2018. She had a pulmonary embolus after a laparoscopic sleeve gastrectomy in 2021. She has normal exercise capacity of greater than 4 METs and no symptoms with a normal EKG."
    result = sh.query(
        query=clinical_phrase_example,
    )
    print(result)
