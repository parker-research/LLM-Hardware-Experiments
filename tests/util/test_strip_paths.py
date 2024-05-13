from llm_experiments.util.strip_paths import strip_paths_from_text, STRIP_PATHS_START_FOLDER_NAME

# Ignore long lines in this file:
# flake8: noqa E501


def test_strip_paths_from_text__minimum():
    assert STRIP_PATHS_START_FOLDER_NAME == "project"

    # Test 1a: Core
    input1a = """
    $dumpvar("/home/some_username/more_paths/LLM-Hardware-Experiments/working/llm_code_gen_retries_2024-05-13_04-16-53L/experiments_chatgpt-3.5/verilog-eval_review2015_fsmonehot_e8508505-0538-4398-bde1-f527d7b701e2/cycle_002/project/wave.vcd");
    """.strip()
    expected_output1 = """
    $dumpvar("/project/wave.vcd");
    """.strip()
    assert strip_paths_from_text(input1a) == expected_output1

    # Test 1b: Add a space in the path
    input1b = """
    $dumpvar("/home/some username/more_paths/LLM-Hardware-Experiments/working/llm_code_gen_retries_2024-05-13_04-16-53L/experiments_chatgpt-3.5/verilog-eval_review2015_fsmonehot_e8508505-0538-4398-bde1-f527d7b701e2/cycle_002/project/wave.vcd");
    """.strip()
    assert strip_paths_from_text(input1b) == expected_output1

    # Test 2a: Windows path (no space)
    expected_output2 = """
    $dumpvar("C:/project/wave.vcd");
    """.strip()

    input2a = """
    $dumpvar("C:\\Users\\some_username\\more_paths\\LLM-Hardware-Experiments\\working\\llm_code_gen_retries_2024-05-13_04-16-53L\\experiments_chatgpt-3.5\\verilog-eval_review2015_fsmonehot_e8508505-0538-4398-bde1-f527d7b701e2\\cycle_002\\project\\wave.vcd");
    """.strip()
    assert strip_paths_from_text(input2a) == expected_output2

    # Test 2b: Windows path (with space)
    input2b = """
    $dumpvar("C:\\Users\\some username\\more_paths\\LLM-Hardware-Experiments\\working\\llm_code_gen_retries_2024-05-13_04-16-53L\\experiments_chatgpt-3.5\\verilog-eval_review2015_fsmonehot_e8508505-0538-4398-bde1-f527d7b701e2\\cycle_002\\project\\wave.vcd");
    """.strip()
    assert strip_paths_from_text(input2b) == expected_output2


def test_strip_paths_from_text__large():
    # Test 1: Large text
    input1 = """
    $dumpvar("/home/some_username/more_paths/LLM-Hardware-Experiments/working/llm_code_gen_retries_2024-05-13_04-16-53L/experiments_chatgpt-3.5/verilog-eval_review2015_fsmonehot_e8508505-0538-4398-bde1-f527d7b701e2/cycle_002/project/wave.vcd");
    $dumpvar("/home/some_username/more_paths/LLM-Hardware-Experiments/working/llm_code_gen_retries_2024-05-13_04-16-53L/experiments_chatgpt-3.5/verilog-eval_review2015_fsmonehot_e8508505-0538-4398-bde1-f527d7b701e2/cycle_002/project/wave.vcd");
    $dumpvar("/home/some_username/more_paths/LLM-Hardware-Experiments/working/llm_code_gen_retries_2024-05-13_04-16-53L/experiments_chatgpt-3.5/verilog-eval_review2015_fsmonehot_e8508505-0538-4398-bde1-f527d7b701e2/cycle_002/project/wave.vcd");
    """.strip()
    expected_output1 = """
    $dumpvar("/project/wave.vcd");
    $dumpvar("/project/wave.vcd");
    $dumpvar("/project/wave.vcd");
    """.strip()
    assert strip_paths_from_text(input1) == expected_output1

    # Test 2:
    input2 = """VCD info: dumpfile /some/file/path/LLM-Hardware-Experiments/working/validate_testbenches_2024-04-30_00-04-47L/experiments/verilog-eval_2014_q3c_609adcde-a6ae-49e8-8cc1-6c06ba6bc026/project/wave.vcd opened for output.
/some/file/path/LLM-Hardware-Experiments/working/validate_testbenches_2024-04-30_00-04-47L/experiments/verilog-eval_2014_q3c_609adcde-a6ae-49e8-8cc1-6c06ba6bc026/project/testbench_code.sv:52: $finish called at 1001 (1ps)
Hint: Output 'Y0' has no mismatches.
Hint: Output 'z' has no mismatches.
Hint: Total mismatched samples is 0 out of 200 samples

Simulation finished at 1001 ps
Mismatches: 0 in 200 samples""".strip()
    expected_output2 = """VCD info: dumpfile /project/wave.vcd opened for output.
/project/testbench_code.sv:52: $finish called at 1001 (1ps)
Hint: Output 'Y0' has no mismatches.
Hint: Output 'z' has no mismatches.
Hint: Total mismatched samples is 0 out of 200 samples

Simulation finished at 1001 ps
Mismatches: 0 in 200 samples""".strip()
    assert strip_paths_from_text(input2) == expected_output2
