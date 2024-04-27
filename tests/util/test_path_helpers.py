from pathlib import Path

from llm_experiments.util.path_helpers import read_secrets_file, get_path_to_git_repo_root


def test_get_path_to_git_repo_root():
    path = get_path_to_git_repo_root()
    assert path.is_dir()

    expected_path = Path(__file__).parent.parent.parent

    assert path == expected_path


def test_read_secrets_file_sample():
    file_path = get_path_to_git_repo_root() / "secrets.sample.yaml"
    assert file_path.is_file()

    secrets = read_secrets_file(file_path)

    assert isinstance(secrets, dict)

    # check that all the expected keys are present
    assert set(secrets.keys()) == {"openai_api_key"}
    # TODO: add more keys as we add more secrets
