from typing import Optional
import uuid

# from loguru import logger

from llm_experiments.logging.text_terminal_logging import make_header_str


class LlmPrompt:
    def __init__(self, prompt_text: str):
        self.prompt_text = prompt_text
        self.uuid = uuid.uuid4()

    def __repr__(self) -> str:
        lines = [
            make_header_str("Start Prompt", char=">"),
            self.prompt_text,
            make_header_str("End Prompt", char=">"),
        ]
        return "\n".join(lines)


class LlmResponse:
    def __init__(self, response_text: str):
        assert isinstance(
            response_text, str
        ), f"response_text must be a string, not {type(response_text)}"
        self.response_text = response_text
        self.uuid = uuid.uuid4()

    def __repr__(self) -> str:
        lines = [
            make_header_str("Start Response", char="<"),
            self.response_text,
            make_header_str("End Response", char="<"),
        ]
        return "\n".join(lines)


class LlmQuery:
    def __init__(self, prompt: LlmPrompt, response: Optional[LlmResponse] = None):
        self.prompt = prompt
        self.response = response

    def set_response(self, response: LlmResponse):
        self.response = response

    def __repr__(self) -> str:
        lines = [
            repr(self.prompt),
            (
                repr(self.response)
                if self.response is not None
                else make_header_str("No Response", char="-")
            ),
        ]
        return "\n".join(lines)
