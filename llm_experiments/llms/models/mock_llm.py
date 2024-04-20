from loguru import logger
from typing import Optional
import re
import json
from dataclasses import dataclass

from llm_experiments.llms.llm_base import LlmBase
from llm_experiments.llms.llm_config_base import LlmConfigBase
from llm_experiments.llms.llm_types import LlmPrompt, LlmResponse, LlmQuery


@dataclass
class MockLlmConfig(LlmConfigBase):
    does_respond_to_test_queries: bool = True

    def __post_init__(self):
        self.llm_class = MockLlm

    def validate_config(self) -> None:
        assert isinstance(self.does_respond_to_test_queries, bool)

class MockLlm(LlmBase):
    """This is a mock LLM that responds to test queries and provides a simple response to other queries.
    This LLM is implemented exactly as all real LLMs are: using complex chains of if-statements and regular expressions. /j
    """
    def __init__(
            self,
            configured_llm_name: str,
            config: LlmConfigBase):
        super().__init__(configured_llm_name, config)

    @classmethod
    def validate_configuration(cls, config: LlmConfigBase):
        assert isinstance(config, LlmConfigBase)
        assert isinstance(config, MockLlmConfig) # must be specifically THIS config class
        assert isinstance(config.does_respond_to_test_queries, bool)
    
    def init_model(self):
        pass
    
    def check_is_connectable(self) -> bool:
        return True
    
    def query_llm(self, prompt: LlmPrompt) -> LlmResponse:
        if self._get_regex_match_groups(
                r"Write.*(?P<number_of_sentences>\d+)\s*sentences.*ending.*apple",
                prompt.prompt_text):
            response_text = " ".join([f"This is sentence {i} ending with apple." for i in range(1, 11)])

        elif self._get_regex_match_groups(
                r"Write.*list.*numbers.*(?P<start_idx>\d+).*(?P<end_idx>)",
                prompt.prompt_text):
            response_text = json.dumps(list(range(1, 11)))

        else:
            response_text = "".join([
                "I am an 'AI' 'language' 'model'.",
                "Responding to prompts other than the test prompts the 'morals' of my creators.",
                "The text I was given as input was: ",
                f"\"{prompt.prompt_text}\"",
            ])

        return LlmResponse(response_text)

    @staticmethod
    def _get_regex_match_groups(regex: str, haystack: str) -> dict[str, str]:
        match = re.match(regex, haystack)
        if match is None:
            return {}
        return match.groupdict()

solid_configs: dict[str, MockLlmConfig] = {
    'mock_llm_which_passes_tests': MockLlmConfig(
        does_respond_to_test_queries=True,
    ),
    'mock_llm_which_fails_tests': MockLlmConfig(
        does_respond_to_test_queries=False,
    ),
}
