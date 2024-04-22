import re


def _extract_verilog_module_from_text_with_markdown_code_blocks(text: str) -> str | None:
    """Extract an entire Verilog module from a text with markdown code blocks."""

    code_block_regex = r"```[ ]*(?P<language_name>\w*)\s*(?P<main_block>.*?)\s*```"

    code_block_matches = re.finditer(code_block_regex, text, re.DOTALL)

    code_block_contents: list[str] = []

    for code_block_match in code_block_matches:
        code_block_contents.append(code_block_match.group("main_block"))

    code_block_contents = [
        code_block.strip()
        for code_block in code_block_contents
        if code_block and len(code_block) > 5
    ]

    if len(code_block_contents) == 0:
        return None

    return "\n\n".join(code_block_contents)


def _extract_verilog_module_from_text_module_ends(text: str) -> str | None:
    """Extract an entire Verilog module from a text by searching for the "module" and "endmodule"
    keywords."""

    module_start_end_regex = r"\bmodule\b(?P<main_block>.*?)\bendmodule\b"

    module_matches = re.finditer(module_start_end_regex, text, re.DOTALL)

    modules_strs: list[str] = []

    for module_match in module_matches:
        modules_strs.append(module_match.group("main_block"))

    modules_strs = [
        module_str.strip() for module_str in modules_strs if module_str and len(module_str) > 5
    ]

    if len(modules_strs) == 0:
        return None
    return "\n\n".join(modules_strs)


def extract_verilog_module_from_text(text: str) -> str | None:
    assert isinstance(text, str)

    verilog_module_from_markdown = _extract_verilog_module_from_text_with_markdown_code_blocks(
        text
    )
    if verilog_module_from_markdown is not None:
        return verilog_module_from_markdown

    verilog_module_from_ends = _extract_verilog_module_from_text_module_ends(text)
    if verilog_module_from_ends is not None:
        return verilog_module_from_ends

    return None
