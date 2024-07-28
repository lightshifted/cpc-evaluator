import os
from datetime import datetime

from colorama import Fore, Style


class PrintUtils:
    @staticmethod
    def green(text):
        return f"{Fore.GREEN}{text}{Style.RESET_ALL}"

    @staticmethod
    def cyan(text):
        return f"{Fore.CYAN}{text}{Style.RESET_ALL}"

    @staticmethod
    def format_stats(stat_dict, color_func=None):
        lines = []
        for key, value in stat_dict.items():
            if key != "quartiles":
                key_str = color_func(f"{key}:") if color_func else f"{key}:"
                lines.append(f"{key_str:<12} {value:.3f}")

        quartiles_key = color_func("Quartiles:") if color_func else "Quartiles:"
        quartiles_value = ", ".join([f"{q:.3f}" for q in stat_dict["quartiles"]])
        lines.append(f"{quartiles_key} {quartiles_value}")

        return "\n".join(lines)

    @staticmethod
    def format_matches(match_dict, color_func=None):
        lines = []
        for key, value in match_dict.items():
            key_str = color_func(f"{key}:") if color_func else f"{key}:"
            lines.append(f"{key_str:<24} {value}")
        return "\n".join(lines)

    @staticmethod
    def print_and_log(message, file, colored_message=None):
        print(colored_message or message)
        file.write(message + "\n")

    @staticmethod
    def print_evaluation_results(dataset, stats, config):
        log_folder = "evaluation_log"
        os.makedirs(log_folder, exist_ok=True)

        timestamp = datetime.now()
        log_file_path = os.path.join(
            log_folder, f"evaluation_log_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
        )

        with open(log_file_path, "w") as log_file:
            # Add datetime to the beginning of the log
            datetime_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            PrintUtils.print_and_log(
                f"Evaluation DateTime: {datetime_str}\n",
                log_file,
                f"{PrintUtils.green('Evaluation DateTime:')} {datetime_str}\n",
            )

            model_name = config.get("LLM", "openai_model")

            PrintUtils.print_and_log(
                f"Num Samples: {dataset.shape[0]}",
                log_file,
                f"{PrintUtils.green('Num Samples:')} {dataset.shape[0]}",
            )
            PrintUtils.print_and_log(
                f"Model: {model_name}\n",
                log_file,
                f"{PrintUtils.green('Model:')} {model_name}\n",
            )

            PrintUtils.print_and_log(
                "ICD-10 Stats:", log_file, PrintUtils.green("ICD-10 Stats:")
            )
            PrintUtils.print_and_log(
                PrintUtils.format_stats(stats[0]),
                log_file,
                PrintUtils.format_stats(stats[0], PrintUtils.green),
            )
            PrintUtils.print_and_log("", log_file)
            PrintUtils.print_and_log(
                PrintUtils.format_matches(stats[1]),
                log_file,
                PrintUtils.format_matches(stats[1], PrintUtils.green),
            )
            PrintUtils.print_and_log("", log_file)

            PrintUtils.print_and_log(
                "CPT Stats:", log_file, PrintUtils.cyan("CPT Stats:")
            )
            PrintUtils.print_and_log(
                PrintUtils.format_stats(stats[2]),
                log_file,
                PrintUtils.format_stats(stats[2], PrintUtils.cyan),
            )
            PrintUtils.print_and_log("", log_file)
            PrintUtils.print_and_log(
                PrintUtils.format_matches(stats[3]),
                log_file,
                PrintUtils.format_matches(stats[3], PrintUtils.cyan),
            )
            PrintUtils.print_and_log("", log_file)

        print(f"Evaluation results have been logged to: {log_file_path}")
