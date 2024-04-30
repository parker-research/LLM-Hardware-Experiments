"""\
This experiment aims to assess how capable various LLMs are at simply generating code for
the select Verilog problems, with no feedback.

The experiment is as follows:
1. Load a list of Verilog problems from the verilog-eval repository.
2. For each problem:
    1. Request that the LLM generates code from the problem prompt.
    2. Extract the generated code.
    3. Save the generated code to a file.
    4. Run the code through IVerilog to check if it builds.
    5. Run the code through IVerilog+vvp to run it, with the test bench.
    6. Check the testbench output to see if it passed.
    7. Collect data on the results.
3. Save the experiment data to a file.
"""

from pathlib import Path
import uuid
from datetime import datetime
import traceback
import re

import orjson
from loguru import logger
import polars as pl

from llm_experiments.logging.env_logging import (
    get_all_env_info,
    log_env_info,
    write_env_info_to_json_file,
)
from llm_experiments.logging.presenters import filter_keys_in_dict
from llm_experiments.util.path_helpers import (
    make_data_dir,
    get_file_date_str,
)
from llm_experiments.feedback_eval_tools.tools.iverilog_tool import (
    IverilogTool,
    IverilogToolConfig,
)
from llm_experiments.util.execute_cli import CommandResult
from llm_experiments.experiments.common.simple_code_gen_problem import SimpleCodeGenProblem
from llm_experiments.experiments.common.verilog_eval_problems import load_verilog_eval_problems

from llm_experiments.llms.llm_base import LlmBase
from llm_experiments.llms.llm_types import LlmPrompt, LlmResponse
from llm_experiments.llms.models.mock_llm import MockLlm, MockLlmConfig

# LLM Models
from llm_experiments.llms.models.ollama_llm import (  # noqa
    OllamaLlm,
    OllamaLlmConfig,
    ollama_good_configs,
)
from llm_experiments.llms.models.chatgpt_llm import (  # noqa
    ChatGptLlm,
    ChatGptLlmConfig,
    chatgpt_good_configs,
)
from llm_experiments.llms.other.ollama_server import ollama_server_singleton

iverilog_tool = IverilogTool("iverilog", config=IverilogToolConfig())


def do_experiment(
    llm: LlmBase,
    problem: SimpleCodeGenProblem,
    working_dir: Path,
    logging_attributes: dict = {},
) -> dict:
    """Runs a single assessment of the simplest possible code generation experiment.

    Runs the simplest possible experiment:
    1. Request that the LLM generates code from the problem prompt.
    2. Extract the generated code.
    3. Save the generated code to a file.
    4. Run the code through IVerilog to check if it builds.
    5. Run the code through IVerilog+vvp to run it, with the test bench.
    6. Collect data on the results.
    """

    experiment_execution_uuid = uuid.uuid4()

    # Prep the return data
    experiment_save_dir = (
        working_dir
        / f"experiments_{llm.configured_llm_name}"
        / f"{problem.problem_id}_{experiment_execution_uuid}"
    )
    (experiment_inputs_dir := experiment_save_dir / "inputs").mkdir(parents=True)
    (experiment_outputs_dir := experiment_save_dir / "outputs").mkdir(parents=True)

    # Step 1: Request that the LLM generates code from the problem prompt
    # TODO: assess prompt engineering strategies to improve performance here
    llm_prompt_text = (
        "You are a student learning Verilog/SystemVerilog. "
        "Solve the following problem by writing a Verilog module. "
        "Wrap your code in Markdown code blocks. "
        "Do not write anything extraneous to the Verilog solution. "
        "You will be given a template module. In your solution, repeat the template, and complete "
        "the rest of the module. "
        "Your solution should be a Verilog module which meets the following requirements: "
        f"{problem.problem_description}"
        f"\n\nTemplate module:\n{problem.module_header}\n// Your solution here"
    )
    (experiment_outputs_dir / "llm_prompt.txt").write_text(llm_prompt_text)

    llm_prompt = LlmPrompt(llm_prompt_text)
    (experiment_outputs_dir / "llm_prompt.json").write_bytes(
        orjson.dumps(llm_prompt.to_dict(), option=orjson.OPT_INDENT_2)
    )
    llm_response: LlmResponse = llm.query_llm_basic(llm_prompt)
    (experiment_outputs_dir / "llm_response.json").write_bytes(
        orjson.dumps(llm_response.to_dict(), option=orjson.OPT_INDENT_2)
    )
    (experiment_outputs_dir / "llm_response.txt").write_text(llm_response.response_text)

    testbench_code = problem.get_testbench_code(experiment_outputs_dir)

    experiment_data = {
        "experiment_group_start_timestamp": logging_attributes["experiment_group_start_timestamp"],
        "experiment_execution_uuid": str(experiment_execution_uuid),
        "base_llm_name": llm.base_llm_name,
        "configured_llm_name": llm.configured_llm_name,
        "llm_configuration": orjson.dumps(llm.config.to_dict()).decode(),
        "problem_id": problem.problem_id,
        "problem_description": problem.problem_description,
        "problem_module_header": problem.module_header,
        "llm_response": llm_response.response_text,
        "llm_response_metadata": orjson.dumps(llm_response.metadata).decode(),
        "experiment_save_dir": str(experiment_save_dir),
        "extracted_generated_code": None,
        "testbench_code": testbench_code,
        "compile_result_return_code": None,
        "compile_result_stdout": None,
        "compile_result_stderr": None,
        "was_compile_success": None,
        "execute_result_return_code": None,
        "execute_result_stdout": None,
        "execute_result_stderr": None,
        "was_execute_success": None,
        "was_testbench_passed": None,  # True means passed, False means failed
        "testbench_stats": None,
        "exit_stage": "end",
    }

    # Step 2: Extract the generated code
    attempted_solution_code: str | None = llm_response.extract_code("verilog_module")
    if attempted_solution_code is None:
        logger.warning("Could not extract code from LLM response.")
        experiment_data["exit_stage"] = "code_extraction_error"
        return experiment_data

    # Step 3: Save the code to a file
    verilog_solution_file_path = experiment_inputs_dir / "attempted_solution_code.sv"
    verilog_solution_file_path.write_text(attempted_solution_code)

    testbench_file_path = experiment_inputs_dir / "testbench_code.sv"
    testbench_file_path.write_text(testbench_code)

    # Step 4: Run the code through IVerilog to check if it builds
    vvp_file_path = experiment_outputs_dir / "compiled.vvp"
    compile_result: CommandResult = iverilog_tool.run_iverilog_compile_command(
        verilog_file_list=[verilog_solution_file_path, testbench_file_path],
        output_vvp_file_path=vvp_file_path,
    )
    compile_result.command_step_name = "compile"
    experiment_data.update(compile_result.as_update_dict())
    compile_result.write_to_files(experiment_outputs_dir)
    if compile_result.return_code != 0:
        experiment_data["exit_stage"] = "iverilog_compile_error"
        return experiment_data

    # Step 5: Run the code through IVerilog's vvp to run it, with the test bench
    execute_result = iverilog_tool.run_iverilog_vvp_execute_command(vvp_file_path)
    execute_result.command_step_name = "execute"
    experiment_data.update(execute_result.as_update_dict())
    execute_result.write_to_files(experiment_outputs_dir)
    if execute_result.return_code != 0:
        experiment_data["exit_stage"] = "iverilog_execute_error"
        return experiment_data

    # Step 6: Check testbench success, and set experiment_data["was_testbench_passed"]
    # TODO: decide if there's a better place for this assessment
    # Find "Hint: Total mismatched samples is 0 out of 439 samples"
    tb_match = re.search(
        r"Total mismatched samples is (?P<mismatch_sample_count>\d+) out of (?P<total_sample_count>\d+) samples",  # noqa
        execute_result.stdout,
        re.IGNORECASE,
    )
    if tb_match:
        mismatch_sample_count = int(tb_match.group("mismatch_sample_count"))
        total_sample_count = int(tb_match.group("total_sample_count"))
        experiment_data["testbench_stats"] = orjson.dumps(
            {
                "mismatch_sample_count": mismatch_sample_count,
                "total_sample_count": total_sample_count,
            }
        ).decode()
        experiment_data["was_testbench_passed"] = mismatch_sample_count == 0
        if experiment_data["was_testbench_passed"]:
            logger.info(f"Testbench passed: {mismatch_sample_count=}, {total_sample_count=}")
            experiment_data["exit_stage"] = "tb_pass"
        else:
            logger.info(f"Testbench failed: {mismatch_sample_count=}, {total_sample_count=}")
            experiment_data["exit_stage"] = "tb_fail"
    else:
        logger.warning("Could not find mismatched sample count in testbench output.")
        experiment_data["was_testbench_passed"] = None
        experiment_data["exit_stage"] = "tb_stats_not_found"

    return experiment_data


def run_experiment_all_inputs():
    logger.info("Starting simple single code generation experiment.")

    experiment_group_start_timestamp = datetime.now()
    experiment_group_start_timestamp_str = get_file_date_str(
        experiment_group_start_timestamp, precision="datetime"
    )

    # Data Setup
    experiment_name = Path(__file__).stem
    working_dir = make_data_dir(
        f"{experiment_name}_{experiment_group_start_timestamp_str}",
        append_date=False,
    )
    logger.info(f"Working directory: {working_dir}")
    ollama_server_singleton.set_log_file_path(working_dir / "ollama_serve.log")

    # Experiment Setup/Logging
    logger.add(working_dir / "general_log.log")
    logger.info(f"Bound general log to: {working_dir / 'general_log.log'}")
    logger.info(f"Experiment group start timestamp: {experiment_group_start_timestamp}")
    env_info = get_all_env_info()
    log_env_info(env_info)
    write_env_info_to_json_file(env_info, working_dir / "env_info.json")

    # Tool Setup
    iverilog_tool.install_and_init_tool()
    iverilog_tool.assert_is_usable()

    # Problems
    problems = load_verilog_eval_problems()
    logger.info(f"Loaded {len(problems):,} problems.")

    global_stats: dict = {
        "total_experiment_count": 0,
        "exception_nonexception_count": {
            "exception": 0,
            "nonexception": 0,
        },
    }

    experiment_data_jsonl_path = working_dir / "experiment_data.jsonl"

    # LLM Setup
    # TODO: move this into a config file
    llm_list: list[LlmBase] = [
        MockLlm(
            "mock_llm_no_preprogrammed_responses",
            config=MockLlmConfig(
                does_respond_to_test_queries=False,
            ),
        ),
        # ChatGptLlm(
        #     "gpt-3.5-turbo-no_randomness",
        #     config=chatgpt_good_configs["gpt-3.5-turbo-no_randomness"],
        # ),
        OllamaLlm(
            "llama2_7b_no_randomness",
            config=ollama_good_configs["llama2_7b_no_randomness"],
        ),
        OllamaLlm(
            "tinyllama_no_randomness",
            config=ollama_good_configs["tinyllama_no_randomness"],
        ),
    ]

    for llm in llm_list:
        llm.init_model()
        logger.info(f"Initialized LLM: {llm}")

        for problem in problems:
            logger.info(f"Running experiment for {llm=}, {problem=}")
            global_stats["total_experiment_count"] += 1

            assert isinstance(problem, SimpleCodeGenProblem)
            assert problem.has_testbench_code

            try:
                experiment_data = do_experiment(
                    llm=llm,
                    problem=problem,
                    working_dir=working_dir,
                    logging_attributes={
                        "experiment_group_start_timestamp": experiment_group_start_timestamp
                    },
                )
            except Exception as e:
                logger.error(f"Error in experiment: {problem=}, {e=}")
                logger.error(traceback.format_exc())
                experiment_data = {
                    "experiment_group_start_timestamp": experiment_group_start_timestamp,
                    "problem_id": problem.problem_id,
                    "problem_description": problem.problem_description,
                    "exit_stage": "exception",
                    "error": str(e),
                }
                global_stats["exception_nonexception_count"]["exception"] += 1
            else:
                global_stats["exception_nonexception_count"]["nonexception"] += 1

            if experiment_data.get("experiment_save_dir"):
                (
                    Path(experiment_data["experiment_save_dir"]) / "experiment_data.json"
                ).write_bytes(orjson.dumps(experiment_data, option=orjson.OPT_INDENT_2))
                logger.info(f"Experiment data saved to: {experiment_data['experiment_save_dir']}")

            experiment_data_short = filter_keys_in_dict(
                experiment_data,
                [
                    "problem_id",
                    "exit_stage",
                    "was_compile_success",
                    "was_execute_success",
                    "was_testbench_passed",
                ],
            )
            summary_emoji = "✅" if experiment_data["was_testbench_passed"] else "😿"
            logger.info(f"{summary_emoji} Done experiment. {experiment_data_short}")

            with open(experiment_data_jsonl_path, "ab") as f:
                f.write(orjson.dumps(experiment_data))
                f.write(b"\n")

    logger.info("Finished all experiments.")
    logger.info(
        f"Final global stats: {orjson.dumps(global_stats, option=orjson.OPT_INDENT_2).decode()}"
    )

    # merge the .jsonl file into a parquet file
    if experiment_data_jsonl_path.is_file():
        df = pl.read_ndjson(experiment_data_jsonl_path)
        df.write_parquet(working_dir / "experiment_data.parquet")
        logger.info(f"Saved experiment data from JSONL to Parquet: {len(df):,} rows.")

        df_stats_1 = df.group_by(["base_llm_name", "configured_llm_name", "exit_stage"]).agg(
            count=pl.len(),
        )
        df_stats_1 = df_stats_1.sort(df_stats_1.columns)
        logger.info(f"Experiment data stats (by exit_stage): {df_stats_1}")
    else:
        logger.warning("No .jsonl file. Appears that all experiments were skipped.")


if __name__ == "__main__":
    run_experiment_all_inputs()
