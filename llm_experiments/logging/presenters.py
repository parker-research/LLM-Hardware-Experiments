from typing import Iterable, Any


def filter_keys_in_dict(d: dict, keys_to_keep: Iterable) -> dict:
    keys_to_keep = set(keys_to_keep)
    return {k: v for k, v in d.items() if k in keys_to_keep}


def flatten_dict(d: dict) -> dict[str, Any]:
    """
    Flattens a dictionary. Lists are joined by ", ".
    """
    flat_dict = {}

    while any(isinstance(value, dict) for value in d.values()):
        for key_top, value_top in d.items():
            if isinstance(value_top, dict):
                # prefix each key in value with the key
                for key_inner, value_inner in value_top.items():
                    flat_dict[f"{key_top}.{key_inner}"] = value_inner
            else:
                flat_dict[key_top] = value_top

    for key_top, value_top in flat_dict.items():
        if isinstance(value_top, list):
            # csv the list
            flat_dict[key_top] = ", ".join(value_top)

    return flat_dict
