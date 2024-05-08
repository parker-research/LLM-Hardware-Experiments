from llm_experiments.llms.llm_from_config import make_llm_providers_from_yaml_config_file
from llm_experiments.util.path_helpers import get_path_to_git_repo_root
from llm_experiments.llms.llm_provider_base import LlmProviderBase


def test_make_llm_providers_from_yaml_config_file():
    config1_path = get_path_to_git_repo_root() / "llm_config.sample.yaml"
    providers = make_llm_providers_from_yaml_config_file(config1_path)

    assert isinstance(providers, list)
    assert len(providers) > 9
    assert all([isinstance(provider, LlmProviderBase) for provider in providers])
