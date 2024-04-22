from llm_experiments.intermediate_steps.extract_verilog import (
    _extract_verilog_module_from_text_with_markdown_code_blocks,
    _extract_verilog_module_from_text_module_ends,
    extract_verilog_module_from_text,
)


def test_extract_from_markdown_no_verilog():
    text = "Here is some text with no code blocks."
    assert _extract_verilog_module_from_text_with_markdown_code_blocks(text) is None


def test_extract_from_markdown_with_verilog():
    text = """
Here is some Verilog code:
```verilog
module my_module;
    Some code here.
endmodule
```"""
    expected = """
module my_module;
    Some code here.
endmodule""".strip()
    assert _extract_verilog_module_from_text_with_markdown_code_blocks(text) == expected


def test_extract_from_text_no_modules():
    text = "This is a plain text with no Verilog modules."
    assert _extract_verilog_module_from_text_module_ends(text) is None


def test_extract_from_text_with_module():
    text = "Here is some Verilog code:\nmodule my_module;\nendmodule"
    expected = "my_module;"
    assert _extract_verilog_module_from_text_module_ends(text) == expected


def test_combined_extraction_no_verilog():
    text = "Just some random text."
    assert extract_verilog_module_from_text(text) is None


def test_combined_extraction_markdown_priority():
    text = (
        "```verilog\nmodule markdown_module;\nendmodule\n```"
        "Random text module my_module;\nendmodule"
    )
    expected = "module markdown_module;\nendmodule"
    assert extract_verilog_module_from_text(text) == expected


def test_combined_extraction_with_module_keyword():
    text = "Random text module my_module;\nendmodule"
    expected = "my_module;"
    assert extract_verilog_module_from_text(text) == expected
