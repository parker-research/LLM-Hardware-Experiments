from dataclasses import dataclass
from typing import Optional, Literal

import openai
from loguru import logger

from llm_experiments.llms.llm_provider_base import LlmProviderBase
from llm_experiments.llms.llm_config_base import LlmConfigBase
from llm_experiments.llms.llm_types import LlmPrompt, LlmResponse
from llm_experiments.util.path_helpers import read_secrets_file


@dataclass(kw_only=True)
class ChatGptLlmConfig(LlmConfigBase):
    # model_name comes from the list here:
    model_name: Literal["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]  # TODO: add more

    option_seed: Optional[int] = None
    option_temperature: Optional[float | int] = None

    # Passed to the ChatGPT/OpenAI API as additional options.
    # Valid keys and values are found here:
    # https://platform.openai.com/docs/api-reference/chat/create
    other_model_file_options: Optional[dict] = None

    def __post_init__(self):
        self.llm_class = "ChatGptLlm"
        super().__post_init__()

    def validate_config(self) -> None:
        assert isinstance(self.model_name, str)

    def to_openai_api_dict(self) -> dict:
        options = {}
        if self.other_model_file_options is not None:
            options.update(self.other_model_file_options)
        if self.option_seed is not None:
            options["seed"] = self.option_seed
        if self.option_temperature is not None:
            options["temperature"] = self.option_temperature
        return options


class ChatGptLlm(LlmProviderBase):
    def __init__(self, config: ChatGptLlmConfig):
        assert isinstance(config, ChatGptLlmConfig)  # must be specifically THIS config class
        self.config = config

        super().__init__()

        _api_key = read_secrets_file()["openai_api_key"]
        assert isinstance(_api_key, str)
        self._api_client = openai.OpenAI(api_key=_api_key)
        self._is_initialized = True

    def destroy_model(self) -> None:
        self._is_initialized = False
        pass

    def check_is_connectable(self) -> bool:
        return True

    def query_llm_basic(self, prompt: LlmPrompt) -> LlmResponse:
        # TODO: check on system prompt
        assert self._is_initialized
        resp = self.query_llm_chat(prompt, [])
        return resp

    def query_llm_chat(
        self, prompt: LlmPrompt, chat_history: list[LlmPrompt | LlmResponse]
    ) -> LlmResponse:
        # add bypass for empty chat history (using this from the basic call)
        if len(chat_history) == 0:
            return self._query_llm_chat_api(prompt, chat_history)

        _chat_history = chat_history.copy()
        for _ in range(len(chat_history)):
            try:
                return self._query_llm_chat_api(prompt, _chat_history)
            except Exception as e:
                if "This model's maximum context length is" in str(e):
                    logger.info(f"Left-truncated chat history due to context length error: {e}")
                    # Hit ChatGPT model's maximum context length. Chop off message history.
                    # Do not remove the system prompt, though.
                    if _chat_history[0].role == "system":
                        _chat_history.pop(1)
                    else:
                        _chat_history.pop(0)
                else:
                    raise e
        raise Exception("Failed to query LLM after multiple attempts of truncation.")

    def _query_llm_chat_api(
        self, prompt: LlmPrompt, chat_history: list[LlmPrompt | LlmResponse]
    ) -> LlmResponse:
        assert self._is_initialized
        messages_query = _convert_chat_history_to_openai_api_dict(chat_history + [prompt])

        # TODO: check on system prompt
        resp_full = self._api_client.chat.completions.create(
            model=self.config.model_name,
            messages=messages_query,
            **self.config.to_openai_api_dict(),
        )
        resp_full = resp_full.choices[0]

        resp_text = resp_full.message.content
        assert isinstance(resp_text, str)
        assert resp_full.message.role == "assistant"
        # assert isinstance(resp_full.done, bool) and (resp_full.done is True)

        resp_metadata = resp_full.copy()
        del resp_metadata.message

        llm_response = LlmResponse(
            response_text=resp_text,
            metadata=dict(resp_metadata),
        )
        return llm_response


def _convert_chat_history_to_openai_api_dict(
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


chatgpt_good_configs: dict[str, ChatGptLlmConfig] = {
    "gpt-3.5-turbo-default": ChatGptLlmConfig(
        configured_llm_name="gpt-3.5-turbo-default",
        model_name="gpt-3.5-turbo",
    ),
    "gpt-3.5-turbo-no_randomness": ChatGptLlmConfig(
        configured_llm_name="gpt-3.5-turbo-no_randomness",
        # is_stable=True,
        model_name="gpt-3.5-turbo",
        option_seed=101,  # any fixed number is good
        option_temperature=0,
    ),
    "gpt-4-default": ChatGptLlmConfig(
        configured_llm_name="gpt-4-default",
        model_name="gpt-4",
    ),
    "gpt-4-no_randomness": ChatGptLlmConfig(
        configured_llm_name="gpt-4-no_randomness",
        # is_stable=True,
        model_name="gpt-4",
        option_seed=101,  # any fixed number is good
        option_temperature=0,
    ),
}
