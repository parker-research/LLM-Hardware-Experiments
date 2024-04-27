from typing import Optional, Literal
from pathlib import Path
from datetime import datetime

import git
import yaml


def get_path_to_git_repo_root() -> Path:
    repo = git.Repo(Path(__file__), search_parent_directories=True)
    repo_root = repo.working_tree_dir
    return Path(repo_root)


def get_file_date_str(
    timestamp: Optional[datetime] = None,
    precision: Literal["date", "datetime", "datetime_ms", "datetime_us"] = "datetime_ms",
) -> str:
    if timestamp is None:
        timestamp = datetime.now()

    datetime_str_us = timestamp.strftime("%Y-%m-%d_%H-%M-%S-%f")

    if precision == "datetime_us":
        return datetime_str_us + "L"
    elif precision == "datetime_ms":
        return datetime_str_us[:-3] + "L"
    elif precision == "datetime":
        return timestamp.strftime("%Y-%m-%d_%H-%M-%S") + "L"
    elif precision == "date":
        return timestamp.strftime("%Y-%m-%d") + "L"
    else:
        raise ValueError(f"Unknown precision: {precision}")


def make_data_dir(name: str, append_date: bool = True) -> Path:
    git_root = get_path_to_git_repo_root()
    working_dir = git_root / "working"

    if append_date:
        data_dir = working_dir / f"{name}_{get_file_date_str(precision='datetime')}"
    else:
        data_dir = working_dir / name

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def read_secrets_file(secrets_file_path: Optional[Path] = None) -> dict:
    """Loads the "<repo_root>/secrets.yaml" file and returns the contents as a dict.
    The secrets_file_path argument can be used to specify a different file path, esp. for testing.
    """
    if secrets_file_path is None:
        secrets_file_path: Path = get_path_to_git_repo_root() / "secrets.yaml"
    with open(secrets_file_path, "r") as f:
        secrets = yaml.safe_load(f)
    return secrets
