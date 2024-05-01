import abc
from typing import Literal
import json

from llm_experiments.llms.llm_config_base import LlmConfigBase
from llm_experiments.llms.llm_types import LlmPrompt, LlmResponse, LlmMetadata


class LlmBase(abc.ABC):
    def __init__(self, configured_llm_name: str, config: LlmConfigBase):
        self.configured_llm_name: str = configured_llm_name
        self.base_llm_name: str = self.__class__.__name__

        self.config: LlmConfigBase = config

        self._is_initialized: bool = False

        self.llm_metadata = LlmMetadata()

    def __repr__(self) -> str:
        return f"{self.base_llm_name}({self.configured_llm_name})"

    @abc.abstractmethod
    def init_model(self) -> None:
        """Prep the model for use right away.
        May involve downloading the model's file, etc.
        Method is stable, and can be called multiple times consecutively.
        """
        pass

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
