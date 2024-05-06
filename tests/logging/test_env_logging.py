import time

from llm_experiments.logging.env_logging import get_all_env_info


def test_get_all_env_info():
    env_info = get_all_env_info()

    assert isinstance(env_info, dict)
    assert len(env_info.keys()) > 5


def test_get_all_env_info_is_stable():
    def remove_datetime_values(env_info: dict) -> dict:
        return {
            section: {
                key: value
                for key, value in values.items()
                if key
                not in [
                    "Current Local Time",
                    "Current UTC Time",
                    "Git Commit Time Ago (HMS)",
                    "Git Commit Time Ago (Minutes)",
                ]
            }
            for section, values in env_info.items()
            if section != "Environment Variables"  # they tend to change a tiny bit
        }

    env_info_1 = get_all_env_info()
    env_info_1 = remove_datetime_values(env_info_1)
    time.sleep(0.5)
    env_info_2 = get_all_env_info()
    env_info_2 = remove_datetime_values(env_info_2)
    time.sleep(0.5)
    env_info_3 = get_all_env_info()
    env_info_3 = remove_datetime_values(env_info_3)

    assert env_info_1 == env_info_2
    assert env_info_1 == env_info_3
