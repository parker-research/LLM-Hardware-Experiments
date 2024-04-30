from typing import Iterable


def filter_keys_in_dict(d: dict, keys_to_keep: Iterable) -> dict:
    keys_to_keep = set(keys_to_keep)
    return {k: v for k, v in d.items() if k in keys_to_keep}
