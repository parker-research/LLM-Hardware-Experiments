"""\
A quick tool to ensure a host of Llama models are downloaded and ready to use.
"""

import argparse

from loguru import logger

from llm_experiments.llms.models.ollama_llm import OllamaLlm, OllamaLlmConfig


def pull_ollama_model(model_name: str) -> None:
    logger.info(f"🏁 Pulling model: {model_name}")
    config = OllamaLlmConfig(model_name=model_name)
    llm = OllamaLlm("ollama", config)
    llm._init_pull_model()

    logger.info(f"🎬 Model {model_name} is ready to use.")


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--model",
        dest="model_names",
        action="append",
        required=True,
        help="The model names to pull (e.g., 'codellama:34b-instruct').",
    )
    args = parser.parse_args()

    logger.info(f"Pulling the following {len(args.model_names)} models: {args.model_names}")

    for model_name in args.model_names:
        pull_ollama_model(model_name)

    logger.info("All models are ready to use.")


if __name__ == "__main__":
    main_cli()
