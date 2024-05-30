from typing import Literal, Optional
import uuid
from datetime import datetime, timezone

import orjson

# from loguru import logger

from llm_experiments.logging.text_terminal_logging import make_header_str
from llm_experiments.intermediate_steps.extract_verilog import extract_verilog_module_from_text


class LlmPrompt:
    role_literal_t = Literal["user", "system"]

    def __init__(
        self, prompt_text: str, role: role_literal_t = "user", agent_name: str | None = None
    ):
        assert isinstance(
            prompt_text, str
        ), f"prompt_text must be a string, not {type(prompt_text)}"
        assert role in ["user", "system"], f"role must be 'user' or 'system', not {role}"

        self.prompt_text = prompt_text
        self.role = role
        self.agent_name = agent_name
        self.uuid = uuid.uuid4()
        self.timestamp = datetime.now(timezone.utc)

    def __str__(self) -> str:
        lines = [
            make_header_str(f"Start Prompt ({self.role.title()})", char=">"),
            make_header_str(f"(agent={self.agent_name}, uuid={self.uuid})", char=">"),
            self.prompt_text,
            make_header_str(f"End Prompt ({self.role.title()})", char=">"),
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "prompt_text": self.prompt_text,
            "role": self.role,
            "agent_name": self.agent_name,
            "uuid": str(self.uuid),
            "timestamp": self.timestamp.isoformat(),
        }

    def to_json(self) -> bytes:
        return orjson.dumps(self.to_dict())


class LlmResponse:
    def __init__(
        self, response_text: str, metadata: Optional[dict] = None, agent_name: str | None = None
    ):
        assert isinstance(
            response_text, str
        ), f"response_text must be a string, not {type(response_text)}"
        self.response_text = response_text
        self.metadata = metadata

        self.role = "assistant"  # currently, there's only one option
        self.agent_name = agent_name
        self.uuid = uuid.uuid4()
        self.timestamp = datetime.now(timezone.utc)

    def __str__(self) -> str:
        lines = [
            make_header_str("Start Response", char="<"),
            make_header_str(f"(agent={self.agent_name}, uuid={self.uuid})", char="<"),
            self.response_text,
            make_header_str("End Response", char="<"),
        ]
        return "\n".join(lines)

    def extract_code(self, code_type: Literal["verilog_module"]) -> str | None:
        if code_type == "verilog_module":
            return extract_verilog_module_from_text(self.response_text)
        else:
            raise ValueError(f"Unknown code type: {code_type}")

    def to_dict(self) -> dict:
        return {
            "response_text": self.response_text,
            "metadata": self.metadata,
            "role": self.role,
            "agent_name": self.agent_name,
            "uuid": str(self.uuid),
            "timestamp": self.timestamp.isoformat(),
        }

    def to_json(self) -> bytes:
        return orjson.dumps(self.to_dict())
