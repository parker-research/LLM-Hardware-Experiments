from pathlib import Path

import yaml

from llm_experiments.llms.llm_provider_base import LlmProviderBase
from llm_experiments.llms.llm_config_base import LlmConfigBase

# providers to be constructed:
from llm_experiments.llms.models.mock_llm import MockLlmConfig, MockLlm
from llm_experiments.llms.models.chatgpt_llm import ChatGptLlmConfig, ChatGptLlm
from llm_experiments.llms.models.ollama_llm import OllamaLlmConfig, OllamaLlm


def make_llm_providers_from_yaml_config_file(
    yaml_config_file_path: str | Path,
) -> list[LlmConfigBase]:
    with open(yaml_config_file_path, "r") as file:
        config_dicts = yaml.safe_load(file)

    llm_providers: list[LlmProviderBase] = []

    for config_dict in config_dicts:
        llm_class_name = config_dict["llm_class"]

        if llm_class_name == "MockLlm":
            provider = MockLlm(MockLlmConfig.from_dict(config_dict))
        elif llm_class_name == "ChatGptLlm":
            provider = ChatGptLlm(ChatGptLlmConfig.from_dict(config_dict))
        elif llm_class_name == "OllamaLlm":
            provider = OllamaLlm(OllamaLlmConfig.from_dict(config_dict))
        else:
            raise ValueError(f"Unknown LLM class: {llm_class_name}")

        llm_providers.append(provider)

    return llm_providers
