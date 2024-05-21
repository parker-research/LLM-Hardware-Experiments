"""\
This experiment aims to assess how capable various LLMs are at generating code for the select
Verilog problems, by feeding the result of the compilation/execution back into the LLM
when the previous attempt is unsuccessful.

The experiment is as follows:
1. Load a list of Verilog problems from the verilog-eval repository.
2. For each problem:
    1. Request that the LLM generates code from the problem prompt.
    2. Extract the generated code.
    3. Save the generated code to a file.
    4. Run the code through IVerilog to check if it builds.
    5. Run the code through IVerilog+vvp to run it, with the test bench.
    6. Check the testbench output to see if it passed.
    7. If it didn't pass, provide feedback to the LLM and repeat the process.
3. Save the experiment data to a file.
"""

from pathlib import Path
import uuid
from datetime import datetime, date
import traceback
import shutil
from dataclasses import dataclass, field
from typing import Literal

import orjson
from loguru import logger
import polars as pl
import fire
from tqdm import tqdm

from llm_experiments.logging.env_logging import (
    get_all_env_info,
    log_env_info,
    write_env_info_to_json_file,
)
from llm_experiments.logging.presenters import filter_keys_in_dict
from llm_experiments.logging.data_manipulation import merge_jsonl_to_parquet
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
from llm_experiments.experiments.common.verilog_eval_problems import (
    load_verilog_eval_problems,
    parse_verilog_eval_testbench_output,
)

from llm_experiments.llms.llm_provider_base import LlmProviderBase
from llm_experiments.llms.llm_types import LlmPrompt, LlmResponse
from llm_experiments.llms.models.ollama_llm import OllamaLlm

from llm_experiments.llms.llm_from_config import make_llm_providers_from_yaml_config_file
from llm_experiments.llms.other.ollama_server import set_ollama_server_log_file_path
from llm_experiments.util.strip_paths import STRIP_PATHS_START_FOLDER_NAME

iverilog_tool = IverilogTool(
    config=IverilogToolConfig(
        configured_tool_name="iverilog_v20240501",
        release_version_date=date(2024, 5, 1),
    )
)


@dataclass(kw_only=True)
class ExperimentStateStore:
    """Stores the state of the experiment, including the current phase, the conversation history,
    and the next prompt text. One of these is created for each experiment, and it is mutated
    as the experiment progresses (as cycles occur).
    """

    phase: Literal["init", "retry"] = "init"

    # keys: agent name, values: list of prompts and responses
    conversation_histories: dict[str, list[LlmPrompt | LlmResponse]] = field(default_factory=dict)

    next_prompt_text: str | None = None
    has_success = False

    def to_dict(self) -> dict:
        """Returns a JSON-safe dictionary representation of the state store."""
        conversation_histories_basic = {
            agent_name: [hist.to_dict() for hist in history]
            for agent_name, history in self.conversation_histories.items()
        }
        return {
            "phase": self.phase,
            "conversation_histories": conversation_histories_basic,
            "next_prompt_text": self.next_prompt_text,
            "has_success": self.has_success,
        }


def generate_next_prompt_text_after_fail(
    extract_code_worked: bool = True,
    compile_result: CommandResult | None = None,
    execute_result: CommandResult | None = None,
    was_testbench_passed: bool | None = None,
    problem: SimpleCodeGenProblem | None = None,
) -> str:
    """Generates the next prompt text based on the failure stage and the data to include.

    This function is intended to "add onto" a conversation history with an LLM, and thus doesn't
    have to include the initial prompt text.
    """

    if not extract_code_worked:
        failure_stage_desc_verb = "extracting the code"
    elif compile_result.return_code != 0:
        failure_stage_desc_verb = "compiling the code"
    elif (execute_result.return_code != 0) or (execute_result.timed_out):
        failure_stage_desc_verb = "running the code/simulation"
        if execute_result.timed_out:
            failure_stage_desc_verb += " (simulation timed out)"
    elif was_testbench_passed is False:
        failure_stage_desc_verb = "passing the testbench"
    else:
        raise ValueError("No failure stage detected.")

    text_parts = [
        "Your solution is not quite right.",
        f"It failed at this step: {failure_stage_desc_verb}.",
    ]

    if not extract_code_worked:
        text_parts.append(
            "Please make sure to follow the requested top-level module template in your response."
        )
    else:
        text_parts.append("Here is the success/failure output from each tool:")

    if compile_result is not None:
        text_parts.append(f"Compile result:\n{compile_result.to_llm_text()}")
    if execute_result is not None:
        text_parts.append(f"Execute result:\n{execute_result.to_llm_text()}")
    if was_testbench_passed is False:
        text_parts.append("The testbench did not pass.")

    text_parts.extend(
        [
            "Please try to solve the problem again: ",
            f"{problem.problem_description}",
            f"Template module:\n{problem.module_header}\n// Your solution here",
        ]
    )

    return "\n\n".join(text_parts)


def do_cycle(
    llm: LlmProviderBase,
    problem: SimpleCodeGenProblem,
    cycle_save_dir: Path,
    experiment_state_store: ExperimentStateStore,  # mutated in place
    experiment_execution_uuid: uuid.UUID | str,
) -> dict:
    """Runs one cycle (LLM query -> LLM response -> extract -> compile -> execute -> next steps)
    of the experiment.

    Returns a dictionary of "cycle_log_data".
    """
    cycle_uuid = uuid.uuid4()

    (cycle_llm_dir := cycle_save_dir / "llm").mkdir(parents=True)
    (cycle_iverilog_dir := cycle_save_dir / STRIP_PATHS_START_FOLDER_NAME).mkdir(parents=True)

    # Step 1: Prompt the LLM
    assert experiment_state_store.next_prompt_text is not None
    llm_prompt = LlmPrompt(experiment_state_store.next_prompt_text)
    llm_response: LlmResponse = llm.query_llm_chat(
        llm_prompt,
        chat_history=experiment_state_store.conversation_histories.get("designer_agent", []),
    )
    assert isinstance(llm_response, LlmResponse)

    # Safety: Clear the next prompt text for the next cycle (must be recreated by following logic)
    experiment_state_store.next_prompt_text = None

    # Store the conversation history
    if experiment_state_store.conversation_histories.get("designer_agent") is None:
        experiment_state_store.conversation_histories["designer_agent"] = []
    experiment_state_store.conversation_histories["designer_agent"].extend(
        [
            llm_prompt,
            llm_response,
        ]
    )

    # Write LLM logs before executing the code
    for agent_name in experiment_state_store.conversation_histories.keys():
        agent_chat_history = experiment_state_store.conversation_histories[agent_name]
        (cycle_llm_dir / f"llm_log_{agent_name}.jsonl").write_bytes(
            b"\n".join([orjson.dumps(hist.to_dict()) for hist in agent_chat_history]) + b"\n"
        )
        (cycle_llm_dir / f"llm_log_{agent_name}.txt").write_text(
            "\n\n".join([str(hist) for hist in agent_chat_history]) + "\n"
        )

    testbench_code = problem.get_testbench_code(vcd_folder_path=cycle_iverilog_dir)

    cycle_log_data = {
        "experiment_execution_uuid": str(experiment_execution_uuid),  # FK to the experiment
        "cycle_uuid": str(cycle_uuid),
        "llm_prompt": llm_prompt.prompt_text,
        "llm_response": llm_response.response_text,
        "llm_response_metadata": orjson.dumps(llm_response.metadata).decode(),
        "extracted_generated_code": None,
        # "testbench_code": testbench_code,
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
        # logger.warning("Could not extract code from LLM response.")
        cycle_log_data["exit_stage"] = "20_code_extraction_error"
        experiment_state_store.next_prompt_text = generate_next_prompt_text_after_fail(
            extract_code_worked=False,
            problem=problem,
        )
        experiment_state_store.phase = "retry"
        return cycle_log_data

    # Step 3: Save the code to a file
    # Add timescale to fix IVerilog warning
    attempted_solution_code = "`timescale 1 ps/1 ps\n\n" + attempted_solution_code + "\n"
    verilog_solution_file_path = cycle_iverilog_dir / "attempted_solution_code.sv"
    verilog_solution_file_path.write_text(attempted_solution_code)

    testbench_file_path = cycle_iverilog_dir / "testbench_code.sv"
    testbench_file_path.write_text(testbench_code)

    # Step 4: Run the code through IVerilog to check if it builds
    vvp_file_path = cycle_iverilog_dir / "compiled.vvp"
    compile_result: CommandResult = iverilog_tool.run_iverilog_compile_command(
        verilog_file_list=[verilog_solution_file_path, testbench_file_path],
        output_vvp_file_path=vvp_file_path,
    )
    compile_result.command_step_name = "compile"
    cycle_log_data.update(compile_result.as_update_dict())
    compile_result.write_to_files(cycle_iverilog_dir)
    if compile_result.return_code != 0:
        cycle_log_data["exit_stage"] = "30_iverilog_compile_error"
        experiment_state_store.next_prompt_text = generate_next_prompt_text_after_fail(
            extract_code_worked=True,
            compile_result=compile_result,
            problem=problem,
        )
        experiment_state_store.phase = "retry"
        return cycle_log_data

    # Step 5: Run the code through IVerilog's vvp to run it, with the test bench
    execute_result = iverilog_tool.run_iverilog_vvp_execute_command(vvp_file_path)
    execute_result.command_step_name = "execute"
    cycle_log_data.update(execute_result.as_update_dict())
    execute_result.write_to_files(cycle_iverilog_dir)
    if execute_result.return_code != 0:
        cycle_log_data["exit_stage"] = "40_iverilog_execute_error"
        experiment_state_store.next_prompt_text = generate_next_prompt_text_after_fail(
            extract_code_worked=True,
            compile_result=compile_result,
            execute_result=execute_result,
            problem=problem,
        )
        experiment_state_store.phase = "retry"
        return cycle_log_data

    # Step 6: Check testbench success
    tb_result = parse_verilog_eval_testbench_output(execute_result.stdout)
    cycle_log_data["was_testbench_passed"] = tb_result["was_testbench_passed"]
    cycle_log_data["testbench_stats"] = tb_result["testbench_stats"]

    if not tb_result["was_testbench_passed"]:
        cycle_log_data["exit_stage"] = "60_tb_failed"
        experiment_state_store.phase = "retry"
        experiment_state_store.next_prompt_text = generate_next_prompt_text_after_fail(
            extract_code_worked=True,
            compile_result=compile_result,
            execute_result=execute_result,
            was_testbench_passed=False,
            problem=problem,
        )
        experiment_state_store.phase = "retry"
        return cycle_log_data

    # FINALLY
    cycle_log_data["exit_stage"] = "80_tb_passed"
    experiment_state_store.has_success = True
    return cycle_log_data


def do_experiment(
    llm: LlmProviderBase,
    problem: SimpleCodeGenProblem,
    max_cycles: int,
    working_dir: Path,
    logging_attributes: dict = {},
) -> dict:
    """Runs an assessment of the LLM's ability to generate code for a given problem, with
    feedback.

    Returns a dictionary of "experiment_data".
    """

    experiment_execution_uuid = uuid.uuid4()

    # Prep the return data
    (
        experiment_save_dir := (
            working_dir
            / f"experiments_{llm.configured_llm_name}"
            / f"{problem.problem_id}_{experiment_execution_uuid}"
        )
    ).mkdir(parents=True)
    logger.info(f"Will save experiment data to: {experiment_save_dir}")

    experiment_data = {
        "experiment_group_start_timestamp": logging_attributes["experiment_group_start_timestamp"],
        "experiment_execution_uuid": str(experiment_execution_uuid),
        "llm_provider_name": llm.llm_provider_name,
        "configured_llm_name": llm.configured_llm_name,
        "llm_configuration": orjson.dumps(llm.config.to_dict()).decode(),
        "problem_id": problem.problem_id,
        "problem_description": problem.problem_description,
        "problem_module_header": problem.module_header,
        "testbench_code": problem._testbench_code,
        "experiment_save_dir": str(experiment_save_dir),
        "used_cycle_count": 0,
        "max_cycle_count": max_cycles,
        "latest_exit_stage": None,
        "was_testbench_passed": False,
    }

    initial_llm_prompt_text = "\n".join(
        [
            "You are a Verilog designer solving an elementary implementation problem.",
            "Solve the following problem by writing a Verilog module.",
            "Wrap your code in Markdown code blocks.",
            "Do not write anything extraneous to the Verilog solution.",
            (
                "You will be given a template module. In your solution, repeat the template, and "
                "fill in your solution."
            ),
            "Your solution should be a Verilog module which meets the following requirements: ",
            f"{problem.problem_description}",
            f"\n\nTemplate module:\n{problem.module_header}\n// Your solution here",
        ]
    )
    experiment_state_store = ExperimentStateStore(next_prompt_text=initial_llm_prompt_text)

    for cycle_num in range(0, max_cycles):
        (cycle_save_dir := experiment_save_dir / f"cycle_{cycle_num:02}").mkdir(parents=True)

        (cycle_save_dir / "experiment_state_entry.json").write_bytes(
            orjson.dumps(experiment_state_store.to_dict(), option=orjson.OPT_INDENT_2)
        )

        cycle_log_data = do_cycle(
            llm=llm,
            problem=problem,
            cycle_save_dir=cycle_save_dir,
            experiment_state_store=experiment_state_store,
            experiment_execution_uuid=experiment_execution_uuid,
        )
        logger.info(f"Cycle {cycle_num:02}: {cycle_log_data['exit_stage']}")

        # Save the cycle data
        (cycle_save_dir / "cycle_log.json").write_bytes(
            orjson.dumps(cycle_log_data, option=orjson.OPT_INDENT_2)
        )
        with open(experiment_save_dir / "cycle_log.jsonl", "ab") as f:
            f.write(orjson.dumps(cycle_log_data) + b"\n")
        (cycle_save_dir / "experiment_state_exit.json").write_bytes(
            orjson.dumps(experiment_state_store.to_dict(), option=orjson.OPT_INDENT_2)
        )

        experiment_data["used_cycle_count"] = cycle_num + 1
        experiment_data["latest_exit_stage"] = cycle_log_data["exit_stage"]

        if cycle_log_data["exit_stage"] == "80_tb_passed":
            experiment_data["was_testbench_passed"] = True
            break

    merge_jsonl_to_parquet(experiment_save_dir / "cycle_log.jsonl")

    return experiment_data


def run_experiment_all_inputs(llm_config_file_path: str | Path, max_cycles: int = 5) -> None:
    logger.info(f"Starting: {__file__} -> run_experiment_all_inputs(...)")
    logger.info(f"Using LLM config file: {llm_config_file_path}")
    logger.info(f"Max cycles: {max_cycles}")

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

    # LLM Setup
    shutil.copy(llm_config_file_path, working_dir / "llm_config.yaml")
    llm_list: list[LlmProviderBase] = make_llm_providers_from_yaml_config_file(
        llm_config_file_path
    )
    logger.info(f"Loaded {len(llm_list):,} LLMs: {llm_list}")
    if any(isinstance(llm, OllamaLlm) for llm in llm_list):
        set_ollama_server_log_file_path(working_dir / "ollama_serve.log")
    else:
        logger.info("No Ollama LLMs found in the list.")

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

    for llm_num, llm in enumerate(llm_list, start=1):
        logger.info(f"Using LLM: {llm}")

        for problem in tqdm(
            problems, desc=f"Solving problems with LLM #{llm_num}/{len(llm_list)}", unit="problem"
        ):
            logger.info(f"Running experiment for {llm=}, {problem=}")
            global_stats["total_experiment_count"] += 1

            assert isinstance(problem, SimpleCodeGenProblem)
            assert problem.has_testbench_code

            try:
                experiment_data = do_experiment(
                    llm=llm,
                    problem=problem,
                    max_cycles=max_cycles,
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
                    "latest_exit_stage": "exception",
                    "was_testbench_passed": False,
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
                    "latest_exit_stage",
                    "was_testbench_passed",
                ],
            )
            summary_emoji = "✅" if experiment_data["was_testbench_passed"] else "😿"
            logger.info(f"{summary_emoji} Done experiment. {experiment_data_short}")

            with open(experiment_data_jsonl_path, "ab") as f:
                f.write(orjson.dumps(experiment_data) + b"\n")

    logger.info("Finished all experiments.")
    logger.info(
        f"Final global stats: {orjson.dumps(global_stats, option=orjson.OPT_INDENT_2).decode()}"
    )

    prep_end_of_experiment_summary_stats(working_dir)

    logger.info("Done.")


def prep_end_of_experiment_summary_stats(working_dir: Path):
    """Prepares the end of experiment summary stats.

    Reads the experiment_data.jsonl file, and prepares the summary stats into
    `working_dir / 'summary' / '*.pq'`.
    """

    logger.info("Making end of experiment summary stats...")
    experiment_data_jsonl_path = working_dir / "experiment_data.jsonl"

    merge_jsonl_to_parquet(experiment_data_jsonl_path)

    (summary_dir := working_dir / "summary").mkdir(parents=True, exist_ok=True)

    # Make Summary 1
    df = pl.read_parquet(experiment_data_jsonl_path.with_suffix(".pq"))
    df_stats_1 = df.group_by(
        ["llm_provider_name", "configured_llm_name", "latest_exit_stage"]
    ).agg(
        count=pl.len(),
    )
    df_stats_1 = df_stats_1.sort(df_stats_1.columns)
    logger.info(f"Experiment data stats (by exit_stage): {df_stats_1}")
    df_stats_1.write_parquet(summary_dir / "experiment_summary_type_1.pq")

    # Make Summary 2 (Pivot)
    df_stats_2 = df_stats_1.pivot(
        index=["llm_provider_name", "configured_llm_name"],
        columns=["latest_exit_stage"],
        values="count",
        sort_columns=True,
    ).select(pl.all().fill_null(pl.lit(0)))
    logger.info(f"Experiment data stats (by exit_stage): {df_stats_2}")
    df_stats_2.write_parquet(summary_dir / "experiment_summary_type_2.pq")


if __name__ == "__main__":
    fire.Fire(run_experiment_all_inputs)
