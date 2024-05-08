from dataclasses import dataclass
from typing import Optional, Any

from tqdm import tqdm
import backoff
from loguru import logger
import ollama
import re
import json


from llm_experiments.logging.presenters import flatten_dict

from llm_experiments.llms.llm_provider_base import LlmProviderBase
from llm_experiments.llms.llm_config_base import LlmConfigBase
from llm_experiments.llms.llm_types import LlmPrompt, LlmResponse

from llm_experiments.llms.other.ollama_server import (
    start_ollama_server,
    stop_ollama_server,
    get_ollama_folder_size_bytes,
)


@dataclass(kw_only=True)
class OllamaLlmConfig(LlmConfigBase):
    # model_name comes from the list here:
    model_name: str
    option_seed: Optional[int] = None
    option_temperature: Optional[float | int] = None

    # Passed to the Ollama API as additional options.
    # Valid keys and values are found here:
    # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
    other_model_file_options: Optional[dict] = None

    def __post_init__(self):
        self.llm_class = "OllamaLlm"
        super().__post_init__()

    def validate_config(self) -> None:
        assert isinstance(self.model_name, str)
        assert (
            ":" in self.model_name
        ), "model_name must contain a release tag (e.g., model_name='tinyllama:1.1b')"

    def to_ollama_api_dict(self) -> dict:
        options = {}
        if self.other_model_file_options is not None:
            options.update(self.other_model_file_options)
        if self.option_seed is not None:
            options["seed"] = self.option_seed
        if self.option_temperature is not None:
            options["temperature"] = self.option_temperature
        return options


class OllamaLlm(LlmProviderBase):
    def __init__(self, config: LlmConfigBase):
        assert isinstance(config, OllamaLlmConfig)  # must be specifically THIS config class
        self.config = config

        super().__init__()

        # extract the parameter count
        param_count_match = re.search(
            r"(?P<param_count_billions>\d+(\.\d+)?)b", self.config.model_name
        )
        if param_count_match:
            self.model_metadata["parameter_count_billions"] = float(
                param_count_match.group("param_count_billions")
            )
        else:
            raise ValueError(
                f"Could not find parameter count in model_name: {self.config.model_name}"
            )

        start_ollama_server()
        self._init_pull_model()

    @staticmethod
    def _list_local_models() -> list[dict]:
        # docs: https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models
        local_models_resp = ollama.list()
        local_models: list[dict] = local_models_resp["models"]
        assert isinstance(local_models, list)
        return local_models

    @backoff.on_exception(
        backoff.constant,
        Exception,
        interval=5,
        max_tries=3,
        on_backoff=lambda x: logger.info(f"Retrying Ollama pull ({x})..."),
    )
    def _init_pull_model(self) -> None:
        # Check if the model is already local, and if so, skip the download.
        local_models = self._list_local_models()
        if self.config.model_name in [model["name"] for model in local_models]:
            logger.info(f"Model '{self.config.model_name}' already downloaded.")
            self._is_initialized = True
            return

        logger.info(f"Downloading Ollama model: {self.config.model_name}")
        with tqdm(
            desc=f"Pulling '{self.config.model_name}' model",
            unit="iB",
            unit_divisor=1024,
            unit_scale=True,
        ) as progress_bar:
            for stream_status in ollama.pull(self.config.model_name, stream=True):
                # Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#response-19
                if "completed" in stream_status:
                    progress_bar.n = stream_status["completed"]
                if "total" in stream_status:
                    progress_bar.total = stream_status["total"]
                if ("completed" in stream_status) or ("total" in stream_status):
                    progress_bar.refresh()

        model_size_GiB = self.get_model_metadata()["size"] / (1024**3)
        storage_folder_size_GiB = get_ollama_folder_size_bytes() / (1024**3)
        logger.info(
            f"Downloaded Ollama model: {self.config.model_name}. "
            f"This model size: {model_size_GiB:.2f} GiB. "
            f"Ollama storage folder size: {storage_folder_size_GiB:.2f} GiB."
        )
        self._is_initialized = True

    def get_model_metadata(self) -> dict[str, Any]:
        list_local_models = self._list_local_models()
        model_metadata = next(
            (model for model in list_local_models if model["name"] == self.config.model_name),
            {},
        )

        # flatten the metadata
        model_metadata_flat = flatten_dict(model_metadata)
        return model_metadata_flat

    def destroy_model(self) -> None:
        ollama.delete(self.config.model_name)
        stop_ollama_server()
        self._is_initialized = False

    def check_is_connectable(self) -> bool:
        return True

    def query_llm_basic(self, prompt: LlmPrompt) -> LlmResponse:
        assert self._is_initialized
        resp_full: dict = ollama.generate(
            model=self.config.model_name,
            prompt=prompt.prompt_text,
            options=self.config.to_ollama_api_dict(),
        )
        assert isinstance(resp_full, dict)

        resp_text = resp_full["response"]
        assert isinstance(resp_text, str)
        assert isinstance(resp_full["done"], bool) and resp_full["done"]

        resp_metadata = resp_full.copy()
        del resp_metadata["response"]

        llm_response = LlmResponse(
            response_text=resp_text,
            metadata=resp_metadata,
        )
        return llm_response

    def query_llm_chat(
        self, prompt: LlmPrompt, chat_history: list[LlmPrompt | LlmResponse]
    ) -> LlmResponse:
        assert self._is_initialized
        messages_query = _convert_chat_history_to_ollama_api_dict(chat_history + [prompt])
        resp_full = ollama.chat(
            model=self.config.model_name,
            messages=messages_query,
            options=self.config.to_ollama_api_dict(),
        )
        assert isinstance(resp_full, dict)

        resp_text = resp_full["message"]["content"]
        assert isinstance(resp_text, str)
        assert resp_full["message"]["role"] == "assistant"
        assert isinstance(resp_full["done"], bool) and resp_full["done"]

        resp_metadata = resp_full.copy()
        del resp_metadata["message"]

        llm_response = LlmResponse(
            response_text=resp_text,
            metadata=resp_metadata,
        )
        return llm_response


def _convert_chat_history_to_ollama_api_dict(
    chat_history: list[LlmPrompt | LlmResponse],
) -> list[dict]:
    messages_query: list[dict] = []
    for item in chat_history:
        if isinstance(item, LlmPrompt):
            messages_query.append(
                {
                    "content": item.prompt_text,
                    "role": item.role,
                }
            )
        elif isinstance(item, LlmResponse):
            messages_query.append(
                {
                    "content": item.response_text,
                    "role": item.role,
                }
            )
        else:
            raise ValueError(f"Invalid item found in 'chat_history': {item}")
    return messages_query


ollama_good_configs: dict[str, OllamaLlmConfig] = {
    "tinyllama_no_randomness": OllamaLlmConfig(
        configured_llm_name="tinyllama_no_randomness",
        model_name="tinyllama:1.1b",
        option_seed=9865,  # any fixed number is good
        option_temperature=0,
    ),
    "llama2_7b_no_randomness": OllamaLlmConfig(
        configured_llm_name="llama2_7b_no_randomness",
        model_name="llama2:7b",
        option_seed=101,  # any fixed number is good
        option_temperature=0,
    ),
}

if __name__ == "__main__":
    logger.info(f"Local models: {json.dumps(OllamaLlm._list_local_models(), indent=2)}")
