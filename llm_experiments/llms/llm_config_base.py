from dataclasses import dataclass
import abc
import json
import yaml


@dataclass(kw_only=True)
class LlmConfigBase(abc.ABC):
    # most configuration dataclass variables go in child classes
    is_stable: bool = False  # indicates that the same input will always produce the same output

    def __post_init__(self):
        self.validate_config()

    @abc.abstractmethod
    def validate_config(self) -> None:
        """Check that the configuration options are valid. Raise an exception if not."""
        pass

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.to_json(pretty=False)})"

    def to_json(self, pretty: bool = True) -> str:
        if pretty:
            return json.dumps(self.to_dict(), indent=4)
        return json.dumps(self.to_dict())

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict())

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)

    @classmethod
    def from_json(cls, config_json: str):
        return cls.from_dict(json.loads(config_json))

    @classmethod
    def from_yaml(cls, config_yaml: str):
        return cls.from_dict(yaml.load(config_yaml, Loader=yaml.FullLoader))
