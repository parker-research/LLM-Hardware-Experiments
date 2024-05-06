import abc
from typing import Literal
import json

from llm_experiments.llms.llm_config_base import LlmConfigBase
from llm_experiments.llms.llm_types import LlmPrompt, LlmResponse


class LlmProviderBase(abc.ABC):
    def __init__(self):
        # Note: super().__init__() must be called after self.config is set in the child class
        self.llm_provider_name: str = self.__class__.__name__  # e.g. "OllamaLlm"
        self._is_initialized: bool = False
        self.model_metadata = {}

        assert isinstance(self.config, LlmConfigBase)

    def __repr__(self) -> str:
        return f"{self.llm_provider_name}({self.config})"

    @property
    def configured_llm_name(self) -> str:
        return self.config.configured_llm_name

    @abc.abstractmethod
    def destroy_model(self) -> None:
        """Clean up the model when done with it.
        May involve deleting the model's files, etc.
        Method is stable, and can be called multiple times consecutively.
        """
        pass

    @abc.abstractmethod
    def check_is_connectable(self) -> bool:
        pass

    @abc.abstractmethod
    def query_llm_basic(self, prompt: LlmPrompt) -> LlmResponse:
        pass

    @abc.abstractmethod
    def query_llm_chat(
        self, prompt: LlmPrompt, chat_history: list[LlmPrompt | LlmResponse]
    ) -> LlmResponse:
        pass

    def perform_test_query(self, test_query: Literal["apple_test", "count_to_10"]):
        if test_query == "apple_test":
            prompt = LlmPrompt("Write me 10 sentences ending with the word 'apple'.")

            def check_response(response: LlmResponse) -> bool:
                response_text = response.response_text

                # LLMs are notoriously bad at this test, so make the pass criteria very lenient
                if response_text.count("apple") < 3:
                    return False
                return True

        elif test_query == "count_to_10":
            prompt = LlmPrompt(
                "Write me a list of numbers from 1 to 10 as a JSON list of integers. "
                "Do not write anything else. Respond with only the JSON list."
            )

            def check_response(response: LlmResponse) -> bool:
                response_text = response.response_text
                try:
                    response_list = json.loads(response_text)
                except json.JSONDecodeError:
                    return False
                if response_list != list(range(1, 11)):
                    return False
                return True

        else:
            raise ValueError(f"Invalid test query: {test_query}")

        response = self.query_llm_basic(prompt)

        response_passes_test = check_response(response)
        return response_passes_test
