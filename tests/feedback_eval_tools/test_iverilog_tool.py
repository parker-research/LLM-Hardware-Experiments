from llm_experiments.feedback_eval_tools.tools.iverilog_tool import IverilogTool


def test__extract_iverilog_version():
    result_stdout = """
Icarus Verilog version 13.0 (devel) (s20221226-516-g615a01c6c)

Copyright (c) 2000-2024 Stephen Williams (steve@icarus.com)

Icarus Verilog Preprocessor version 13.0 (devel) (s20221226-516-g615a01c6c)

Copyright (c) 1999-2024 Stephen Williams (steve@icarus.com)

Icarus Verilog Parser/Elaborator version 13.0 (devel) (s20221226-516-g615a01c6c)

Copyright (c) 1998-2024 Stephen Williams (steve@icarus.com)

 FLAGS DLL vvp.tgt
vvp.tgt: Icarus Verilog VVP Code Generator 13.0 (devel) (s20221226-516-g615a01c6c)

Copyright (c) 2001-2024 Stephen Williams (steve@icarus.com)

  This program is free software; you can redistribute it and/or modify
"""

    parsed = IverilogTool._extract_iverilog_version(result_stdout)

    assert len(parsed) == 4
    assert parsed == [
        {
            "tool_name": "Icarus Verilog",
            "version_num": "13.0",
            "full_version_id": "13.0 (devel) (s20221226-516-g615a01c6c)",
        },
        {
            "tool_name": "Icarus Verilog Preprocessor",
            "version_num": "13.0",
            "full_version_id": "13.0 (devel) (s20221226-516-g615a01c6c)",
        },
        {
            "tool_name": "Icarus Verilog Parser/Elaborator",
            "version_num": "13.0",
            "full_version_id": "13.0 (devel) (s20221226-516-g615a01c6c)",
        },
        {
            "tool_name": "vvp.tgt: Icarus Verilog VVP Code Generator",
            "version_num": "13.0",
            "full_version_id": "13.0 (devel) (s20221226-516-g615a01c6c)",
        },
    ]
