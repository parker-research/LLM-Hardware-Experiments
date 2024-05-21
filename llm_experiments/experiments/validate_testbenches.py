"""\
This experiment aims to validate that the testbenches and reference solution work correctly.

The experiment is as follows:
1. Load a list of Verilog problems from the verilog-eval repository.
2. For each problem:
    1. Load the canonical solution and testbench code for the problem.
    2. Save the generated code to a file.
    3. Run the code through IVerilog to check if it builds.
    4. Run the code through IVerilog+vvp to run it, with the test bench.
    5. Check the testbench output to see if it passed.
    6. Collect data on the results.
3. Save the experiment data to a file.
"""

from pathlib import Path
import uuid
from datetime import datetime, date
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
from llm_experiments.intermediate_steps.extract_verilog import extract_verilog_module_from_text

iverilog_tool = IverilogTool(
    config=IverilogToolConfig(
        configured_tool_name="iverilog_v20240501",
        release_version_date=date(2024, 5, 1),
    )
)


def do_experiment(
    problem: SimpleCodeGenProblem,
    working_dir: Path,
    logging_attributes: dict = {},
) -> dict:
    """Runs a single assessment of the simplest possible code generation experiment."""

    experiment_execution_uuid = uuid.uuid4()

    # Prep the return data
    experiment_save_dir = (
        working_dir / "experiments" / f"{problem.problem_id}_{experiment_execution_uuid}"
    )
    (experiment_inputs_dir := experiment_save_dir / "inputs").mkdir(parents=True)
    (experiment_outputs_dir := experiment_save_dir / "outputs").mkdir(parents=True)

    canonical_solution_code = problem.get_canonical_solution(experiment_outputs_dir)
    testbench_code = problem.get_testbench_code(experiment_outputs_dir)

    experiment_data = {
        "experiment_group_start_timestamp": logging_attributes["experiment_group_start_timestamp"],
        "experiment_execution_uuid": str(experiment_execution_uuid),
        "problem_id": problem.problem_id,
        "problem_description": problem.problem_description,
        "experiment_save_dir": str(experiment_save_dir),
        "solution_code": canonical_solution_code,
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

    # Step 2: Extract the "generated" code.
    # In this testbench validation, this is validation that the code extractor works.
    extracted_code: str | None = extract_verilog_module_from_text(canonical_solution_code)
    assert extracted_code, "Could not extract verilog module from canonical solution code."
    assert len(extracted_code) > 0.75 * len(
        canonical_solution_code
    ), "Extracted code is too short."

    # Step 3: Save the code to a file
    verilog_file_path = experiment_inputs_dir / "solution_code.sv"
    verilog_file_path.write_text(canonical_solution_code)

    testbench_file_path = experiment_inputs_dir / "testbench_code.sv"
    testbench_file_path.write_text(testbench_code)

    # Step 4: Run the code through IVerilog to check if it builds
    vvp_file_path = experiment_outputs_dir / "compiled.vvp"
    compile_result: CommandResult = iverilog_tool.run_iverilog_compile_command(
        verilog_file_list=[verilog_file_path, testbench_file_path],
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
            logger.warning(f"Testbench failed: {mismatch_sample_count=}, {total_sample_count=}")
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
    working_dir = make_data_dir(
        f"validate_testbenches_{experiment_group_start_timestamp_str}",
        append_date=False,
    )

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
        "valid_invalid_count": {
            "valid": 0,
            "invalid": 0,
        },
        "exception_nonexception_count": {
            "exception": 0,
            "nonexception": 0,
        },
    }

    experiment_data_jsonl_path = working_dir / "experiment_data.jsonl"

    for problem in problems:
        logger.info(f"Running experiment for {problem=}")
        global_stats["total_experiment_count"] += 1

        if not problem.has_canonical_solution or not problem.has_testbench_code:
            logger.warning(f"Skipping problem due to missing data: {problem=}")
            global_stats["valid_invalid_count"]["invalid"] += 1
            continue
        else:
            global_stats["valid_invalid_count"]["valid"] += 1

        try:
            experiment_data = do_experiment(
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
            with open(
                Path(experiment_data["experiment_save_dir"]) / "experiment_data.json", "wb"
            ) as f:
                f.write(orjson.dumps(experiment_data, option=orjson.OPT_INDENT_2))
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
        logger.info(f"Done experiment. {experiment_data_short}")

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
        df.write_parquet(working_dir / "experiment_data.pq")
        logger.info(f"Saved experiment data from JSONL to Parquet: {len(df):,} rows.")

        df_stats_1 = df.group_by(["exit_stage"]).agg(count=pl.len())
        logger.info(f"Experiment data stats (by exit_stage): {df_stats_1}")
    else:
        logger.warning("No .jsonl file. Appears that all experiments were skipped.")


if __name__ == "__main__":
    run_experiment_all_inputs()
