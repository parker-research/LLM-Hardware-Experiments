from typing import Literal


def add_line_numbers(
    input_contents: str,
    location: Literal["prefix", "postfix"] = "prefix",
    start_line_number: int = 1,
) -> str:
    """
    Prefixes or postfixes line numbers to each line in the input_contents.
    """
    lines = input_contents.replace("\r", "").split("\n")
    if location == "prefix":
        lines = [f"{i}: {line}" for i, line in enumerate(lines, start=start_line_number)]
    elif location == "postfix":
        lines = [f"{line} :{i}" for i, line in enumerate(lines, start=start_line_number)]
    return "\n".join(lines)
