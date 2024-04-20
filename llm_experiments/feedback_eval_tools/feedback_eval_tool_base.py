
from loguru import logger
import abc
from typing import Optional

class FeedbackEvalToolBase(abc.ABC):
    def __init__(self, configured_tool_name: str, config: dict = {}):
        self.configured_tool_name: str = configured_tool_name
        self.base_tool_name: str = self.__class__.__name__

        self.validate_configuration(config)
        self.config: dict = config

    def __repr__(self) -> str:
        return f"{self.base_tool_name}, acting as {self.configured_tool_name} ({self.config})"
    
    @abc.abstractmethod
    @classmethod
    def validate_configuration(cls, config: dict):
        """Check that the configuration options are valid. Raise an exception if not."""
        pass
    

    @abc.abstractmethod
    def check_is_connectable(self) -> bool:
        pass
    
    @abc.abstractmethod
    def evaluate(self, evaluation_info: dict) -> float:
        # FIXME: come up with a reasonable way to implement this
        pass
    
    @abc.abstractmethod
    def evaluate_query(self, query: dict) -> dict:
        pass
    