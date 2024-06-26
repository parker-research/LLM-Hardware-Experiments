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
    7. If it didn't pass, make a "debugger agent" LLM explain the issue to the "designer agent".
    8. Pass this feedback from the "debugger agent" to the "designer agent", and regenerate code.
3. Save the experiment data to a file.
"""

import argparse
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
from tqdm import tqdm
import pydash

from llm_experiments.logging.env_logging import (
    get_all_env_info,
    log_env_info,
    write_env_info_to_json_file,
)
from llm_experiments.logging.presenters import filter_keys_in_dict
from llm_experiments.logging.data_manipulation import merge_jsonl_to_parquet
from llm_experiments.logging.llm_logging import write_llm_conversation_to_files
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
from llm_experiments.intermediate_steps.line_numbering import add_line_numbers

iverilog_tool = IverilogTool(
    config=IverilogToolConfig(
        configured_tool_name="iverilog_v20240501",
        release_version_date=date(2024, 5, 1),
    )
)


@dataclass(kw_only=True)
class AgentUsageVariant:
    """Configuration parameters for the agent experiment."""

    name: str = "v1"

    # A conversation cycle is one full loop from Problem or Previous Attempt -> Debugger
    # (if cycle > 1) -> Designer -> Testbench Evaluation tool.
    # A value of 1 means that the Designer Agent is not given a chance to retry.
    max_conversation_cycles: int = 5
    repeat_last_attempt_in_reprompts_to_designer: bool = False
    add_line_number_prefixes_to_code: bool = False
    debugger_prompt_phrasing_variant: Literal["basic", "be_specific"] = "basic"
    debugger_prompt_instruction_location: Literal["top", "bottom"] = "bottom"

    # Whether to wrap the Verilog code in Markdown code blocks in prompts to the LLM
    wrap_verilog_code_in_md_code_blocks: bool = False

    # In the future, the `llm_code_gen_retries.py` experiment should be refactored into
    # this file, and enable_debugger_agent=False should run that experiment.
    enable_debugger_agent: bool = True


agent_usage_variants = [
    # all defaults, original version before identifying these parameters
    AgentUsageVariant(
        name="v1",
        max_conversation_cycles=5,
        repeat_last_attempt_in_reprompts_to_designer=False,
        add_line_number_prefixes_to_code=False,
        debugger_prompt_phrasing_variant="basic",
        debugger_prompt_instruction_location="bottom",
        wrap_verilog_code_in_md_code_blocks=False,
    ),
    AgentUsageVariant(
        name="repeat_last_attempt",
        repeat_last_attempt_in_reprompts_to_designer=True,
        wrap_verilog_code_in_md_code_blocks=True,  # always True going forwards
    ),
    AgentUsageVariant(
        name="add_line_nums",
        add_line_number_prefixes_to_code=True,
        wrap_verilog_code_in_md_code_blocks=True,  # always True going forwards
    ),
    AgentUsageVariant(
        name="be_specific",
        debugger_prompt_phrasing_variant="be_specific",
        wrap_verilog_code_in_md_code_blocks=True,  # always True going forwards
    ),
    AgentUsageVariant(
        name="top_instructions",
        debugger_prompt_instruction_location="top",
        wrap_verilog_code_in_md_code_blocks=True,  # always True going forwards
    ),
    AgentUsageVariant(
        name="v2",
        repeat_last_attempt_in_reprompts_to_designer=True,
        add_line_number_prefixes_to_code=True,
        debugger_prompt_phrasing_variant="be_specific",
        debugger_prompt_instruction_location="top",
        wrap_verilog_code_in_md_code_blocks=True,
    ),
    AgentUsageVariant(
        name="v3",
        repeat_last_attempt_in_reprompts_to_designer=False,  # expensive with little benefit
        add_line_number_prefixes_to_code=True,
        debugger_prompt_phrasing_variant="be_specific",
        debugger_prompt_instruction_location="bottom",
        wrap_verilog_code_in_md_code_blocks=True,
    ),
]


def get_named_agent_usage_variant(name: str) -> AgentUsageVariant:
    matches = [v for v in agent_usage_variants if v.name == name]
    if len(matches) == 0:
        raise ValueError(f"No AgentUsageVariant with name: '{name}'")
    if len(matches) > 1:
        raise ValueError(f"Multiple AgentUsageVariants with name: '{name}'; check setup.")
    return matches[0]


@dataclass(kw_only=True)
class ExperimentStateStore:
    """Stores the state of the experiment, including the current phase, the conversation history,
    and the next prompt text. One of these is created for each experiment, and it is mutated
    as the experiment progresses (as cycles occur).
    """

    phase: Literal["init", "retry"] = "init"

    # keys: agent name, values: list of prompts and responses
    conversation_histories: dict[str, list[LlmPrompt | LlmResponse]] = field(default_factory=dict)

    last_attempted_solution_code: str | None = None
    last_compile_result: CommandResult | None = None
    last_execute_result: CommandResult | None = None
    last_was_testbench_passed: bool = False

    def clear_last_results(self) -> None:
        self.last_attempted_solution_code = None
        self.last_compile_result = None
        self.last_execute_result = None
        self.last_was_testbench_passed = False

    def extend_conversation_history(self, agent_name: str, history: list[LlmPrompt | LlmResponse]):
        assert all(isinstance(hist, (LlmPrompt, LlmResponse)) for hist in history)

        # Ensure all history items have the correct agent name
        for i in range(len(history)):
            if history[i].agent_name is None:
                history[i].agent_name = agent_name
            else:
                assert history[i].agent_name == agent_name

        if self.conversation_histories.get(agent_name) is None:
            self.conversation_histories[agent_name] = []
        self.conversation_histories[agent_name].extend(history)

    def to_dict(self) -> dict:
        """Returns a JSON-safe dictionary representation of the state store."""
        conversation_histories_basic = {
            agent_name: [hist.to_dict() for hist in history]
            for agent_name, history in self.conversation_histories.items()
        }
        return {
            "phase": self.phase,
            "conversation_histories": conversation_histories_basic,
            "last_attempted_solution_code": self.last_attempted_solution_code,
            "last_compile_result": (
                self.last_compile_result.as_dict() if self.last_compile_result else None
            ),
            "last_execute_result": (
                self.last_execute_result.as_dict() if self.last_execute_result else None
            ),
            "last_was_testbench_passed": self.last_was_testbench_passed,
        }


def generate_designer_prompt_text_after_fail(
    problem: SimpleCodeGenProblem,
    debugger_feedback: str,
    attempted_solution_code: str | None,
    variant_repeat_last_attempt: bool,
) -> str:
    """Generated the next prompt for the Designer Agent, based on the Debugger Agent's feedback."""
    text_parts = [
        "You have attempted to solve the following problem: ",
        f"{problem.problem_description}",
        "The solution you came up with is not quite right.",
    ]

    if variant_repeat_last_attempt and (attempted_solution_code is not None):
        text_parts.extend(
            [
                "Recall, your previous solution was:",
                attempted_solution_code,
            ]
        )

    text_parts.extend(
        [
            "The Debugger Employee has provided the following feedback:",
            debugger_feedback,
            "Please attempt to solve the problem correctly.",
        ]
    )

    return "\n\n".join(text_parts)


def generate_debugger_prompt_text_after_fail(
    *,
    problem: SimpleCodeGenProblem,
    attempted_solution_code: str | None,
    compile_result: CommandResult | None = None,
    execute_result: CommandResult | None = None,
    was_testbench_passed: bool | None = None,
    variant_debugger_prompt_phrasing_variant: Literal["basic", "be_specific"],
    variant_debugger_prompt_instruction_location: Literal["top", "bottom"],
    variant_wrap_verilog_code_in_md_code_blocks: bool,
) -> str:
    """Generates the next prompt for the Debugger Agent based on the failure stage, last attempt,
    and the tool (IVerilog) logs.

    This function is intended to "add onto" a conversation history with an LLM.
    """

    if variant_debugger_prompt_phrasing_variant == "basic":
        instruction_text = (
            "Please construct feedback for the Designer Employee, such that they can better "
            "attempt to solve the problem next time. Do not provide a solution. You may, however, "
            "reference parts of the solution in your feedback."
        )
    elif variant_debugger_prompt_phrasing_variant == "be_specific":
        instruction_text = (
            "Please construct a report for the Designer Employee, such that they can better "
            "attempt to solve the problem next time. Be EXTREMELY SPECIFIC about what you think "
            "the issue(s) may be, and how they can be fixed. Be precise and detailed."
        )
    else:
        raise ValueError(
            f"Unknown variant_debugger_prompt_phrasing_variant: {variant_debugger_prompt_phrasing_variant}"  # noqa: E501
        )

    assert variant_debugger_prompt_instruction_location in ["top", "bottom"]

    if attempted_solution_code is None:
        failure_stage_desc_verb = "extracting the code"
    elif compile_result.return_code != 0:
        failure_stage_desc_verb = "compiling the code"
    elif (execute_result.return_code != 0) or (execute_result.timed_out):
        failure_stage_desc_verb = "running the simulation"
        if execute_result.timed_out:
            failure_stage_desc_verb += (
                " (simulation timed out; check for self-referential combinatorial assignments)"
            )
    elif was_testbench_passed is False:
        failure_stage_desc_verb = "passing the testbench"
    else:
        raise ValueError("No failure stage detected.")

    text_parts = [
        "Your coworker, the Designer Employee, has attempted to solve the following problem: ",
        f"{problem.problem_description}",
        "The solution your coworker came up with is not quite right.",
    ]
    if variant_debugger_prompt_instruction_location == "top":
        text_parts.append(instruction_text)

    if attempted_solution_code is not None:
        text_parts.append("The designer proposed the following solution:")
        if variant_wrap_verilog_code_in_md_code_blocks:
            text_parts.append(f"```verilog\n{attempted_solution_code}\n```")
        else:
            text_parts.append(attempted_solution_code)

    text_parts.append(
        f"When testing the solution, it failed at this step: {failure_stage_desc_verb}."
    )

    if attempted_solution_code is None:
        text_parts.append("It is important to follow the requested top-level module template.")
    else:
        text_parts.append("Here is the success/failure output from each tool:")

    if compile_result is not None:
        text_parts.append(f"Compile result:\n{compile_result.to_llm_text()}")
    if execute_result is not None:
        text_parts.append(f"Execute result:\n{execute_result.to_llm_text()}")
    if was_testbench_passed is False:
        text_parts.append("The testbench did not pass.")

    if variant_debugger_prompt_instruction_location == "bottom":
        text_parts.append(instruction_text)

    if any(i is None for i in text_parts):
        raise ValueError("Some text parts are None. This should not happen.")

    return "\n\n".join(text_parts)


def do_conversation_cycle(
    llm: LlmProviderBase,
    problem: SimpleCodeGenProblem,
    agent_usage_variant: AgentUsageVariant,
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

    if not agent_usage_variant.enable_debugger_agent:
        raise NotImplementedError("Experiment does not yet support bypassing the debugger agent.")

    if experiment_state_store.phase == "init":
        # Set the system prompts for the two agents
        experiment_state_store.extend_conversation_history(
            "designer_agent",
            [
                LlmPrompt(
                    "You work in a hardware design company with many employees. "
                    "You are a hardware designer with extensive Verilog experience, who is tasked "
                    "with solving a Verilog problem. You must obey instructions.",
                    role="system",
                    agent_name="designer_agent",
                ),
            ],
        )
        experiment_state_store.extend_conversation_history(
            "debugger_agent",
            [
                LlmPrompt(
                    "You work in a hardware design company with many employees. "
                    "You are a hardware engineer with extensive debugging experience, who is "
                    "tasked with providing feedback on a Verilog solution to better solve the "
                    "problem. You must obey instructions.",
                    role="system",
                    agent_name="debugger_agent",
                ),
            ],
        )

        # Step 1: Prompt the Designer Agent, with no agent feedback
        designer_prompt_text = "\n".join(
            [
                "Solve the following problem by writing a Verilog module.",
                "Wrap your code in Markdown code blocks.",
                "Do not write anything extraneous to the Verilog solution.",
                (
                    "You will be given a template module. In your solution, repeat the template, "
                    "and fill in your solution."
                ),
                "Your solution should be a Verilog module which meets the following requirements:",
                f"{problem.problem_description}",
                f"\n\nTemplate module:\n{problem.module_header}\n// Your solution here",
            ]
        )

    elif experiment_state_store.phase == "retry":
        attempted_solution_code_for_llm = experiment_state_store.last_attempted_solution_code
        if agent_usage_variant.add_line_number_prefixes_to_code and (
            attempted_solution_code_for_llm is not None
        ):
            attempted_solution_code_for_llm = add_line_numbers(
                attempted_solution_code_for_llm,
                location="prefix",
                start_line_number=3,  # 'timescale' line, plus a blank line, are prepended in exec
            )

        # Step 1A: Prompt the Debugger Agent
        debugger_prompt_text = generate_debugger_prompt_text_after_fail(
            problem=problem,
            attempted_solution_code=attempted_solution_code_for_llm,
            compile_result=experiment_state_store.last_compile_result,
            execute_result=experiment_state_store.last_execute_result,
            was_testbench_passed=experiment_state_store.last_was_testbench_passed,
            # variant settings:
            variant_debugger_prompt_phrasing_variant=agent_usage_variant.debugger_prompt_phrasing_variant,  # noqa: E501
            variant_debugger_prompt_instruction_location=agent_usage_variant.debugger_prompt_instruction_location,  # noqa: E501
            variant_wrap_verilog_code_in_md_code_blocks=agent_usage_variant.wrap_verilog_code_in_md_code_blocks,  # noqa: E501
        )
        debugger_llm_prompt = LlmPrompt(debugger_prompt_text, agent_name="debugger_agent")
        debugger_llm_response: LlmResponse = llm.query_llm_chat(
            debugger_llm_prompt,
            chat_history=experiment_state_store.conversation_histories.get("debugger_agent", []),
        )
        debugger_llm_response.agent_name = "debugger_agent"
        experiment_state_store.extend_conversation_history(
            agent_name="debugger_agent",
            history=[debugger_llm_prompt, debugger_llm_response],
        )

        # Step 1B: Prompt the Designer Agent, with the Debugger Agent's feedback
        designer_prompt_text = generate_designer_prompt_text_after_fail(
            problem=problem,
            debugger_feedback=debugger_llm_response.response_text,
            attempted_solution_code=attempted_solution_code_for_llm,
            variant_repeat_last_attempt=agent_usage_variant.repeat_last_attempt_in_reprompts_to_designer,  # noqa: E501
        )

    else:
        raise ValueError(f"Unknown phase: {experiment_state_store.phase}")

    # regardless, prompt the Designer Agent
    designer_llm_prompt = LlmPrompt(designer_prompt_text, agent_name="designer_agent")
    designer_llm_response: LlmResponse = llm.query_llm_chat(
        designer_llm_prompt,
        chat_history=experiment_state_store.conversation_histories.get("designer_agent", []),
    )
    designer_llm_response.agent_name = "designer_agent"
    experiment_state_store.extend_conversation_history(
        agent_name="designer_agent",
        history=[designer_llm_prompt, designer_llm_response],
    )
    write_llm_conversation_to_files(
        pydash.flatten(experiment_state_store.conversation_histories.values()),
        cycle_llm_dir,
    )

    # Clear the last results
    experiment_state_store.clear_last_results()

    testbench_code = problem.get_testbench_code(vcd_folder_path=cycle_iverilog_dir)

    cycle_log_data = {
        "experiment_execution_uuid": str(experiment_execution_uuid),  # FK to the experiment
        "cycle_uuid": str(cycle_uuid),
        "llm_prompt": designer_llm_prompt.prompt_text,
        "llm_response": designer_llm_response.response_text,
        "llm_response_metadata": orjson.dumps(designer_llm_response.metadata).decode(),
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
    attempted_solution_code: str | None = designer_llm_response.extract_code("verilog_module")
    # attempted_solution_code without line numbers
    experiment_state_store.last_attempted_solution_code = attempted_solution_code
    if attempted_solution_code is None:
        # logger.warning("Could not extract code from LLM response.")
        cycle_log_data["exit_stage"] = "20_code_extraction_error"
        experiment_state_store.phase = "retry"
        return cycle_log_data

    # Step 3: Save the code to a file
    # Add timescale to fix IVerilog warning
    if not attempted_solution_code.startswith("`timescale"):
        attempted_solution_code = "`timescale 1 ps/1 ps\n\n" + attempted_solution_code
    attempted_solution_code += "\n"
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
    experiment_state_store.last_compile_result = compile_result
    if compile_result.timed_out:
        # this should be rare/non-existent; compiling these small programs should be fast
        cycle_log_data["exit_stage"] = "25_iverilog_compile_timeout"
        experiment_state_store.phase = "retry"
        return cycle_log_data
    elif compile_result.return_code != 0:
        cycle_log_data["exit_stage"] = "30_iverilog_compile_error"
        experiment_state_store.phase = "retry"
        return cycle_log_data

    # Step 5: Run the code through IVerilog's vvp to run it, with the test bench
    execute_result = iverilog_tool.run_iverilog_vvp_execute_command(vvp_file_path)
    execute_result.command_step_name = "execute"
    cycle_log_data.update(execute_result.as_update_dict())
    execute_result.write_to_files(cycle_iverilog_dir)
    experiment_state_store.last_execute_result = execute_result
    if execute_result.timed_out:
        # This happens especially when the unit-under-test contains assignments which reference
        # other assigned signals, creating a combinatorial loop.
        cycle_log_data["exit_stage"] = "35_iverilog_execute_timeout"
        experiment_state_store.phase = "retry"
        return cycle_log_data
    elif execute_result.return_code != 0:
        cycle_log_data["exit_stage"] = "40_iverilog_execute_error"
        experiment_state_store.phase = "retry"
        return cycle_log_data

    # Step 6: Check testbench success
    tb_result = parse_verilog_eval_testbench_output(execute_result.stdout)
    cycle_log_data["was_testbench_passed"] = tb_result["was_testbench_passed"]
    cycle_log_data["testbench_stats"] = tb_result["testbench_stats"]

    if not tb_result["was_testbench_passed"]:
        cycle_log_data["exit_stage"] = "60_tb_failed"
        experiment_state_store.phase = "retry"
        return cycle_log_data

    # FINALLY
    cycle_log_data["exit_stage"] = "80_tb_passed"
    experiment_state_store.last_was_testbench_passed = True
    return cycle_log_data


def do_experiment(
    llm: LlmProviderBase,
    problem: SimpleCodeGenProblem,
    agent_usage_variant: AgentUsageVariant,
    working_dir: Path,
    logging_attributes: dict = {},
) -> dict:
    """Runs an assessment of the LLM's ability to generate code for a given problem, where it tries
    to solve that problem. Behaviour is governed by the `agent_usage_variant`.

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
        "max_cycle_count": agent_usage_variant.max_conversation_cycles,
        "latest_exit_stage": None,
        "was_testbench_passed": False,
    }

    experiment_state_store = ExperimentStateStore()

    for conversation_cycle_num in range(0, agent_usage_variant.max_conversation_cycles):
        (cycle_save_dir := experiment_save_dir / f"cycle_{conversation_cycle_num:02}").mkdir(
            parents=True
        )

        (cycle_save_dir / "experiment_state_entry.json").write_bytes(
            orjson.dumps(experiment_state_store.to_dict(), option=orjson.OPT_INDENT_2)
        )

        cycle_log_data = do_conversation_cycle(
            llm=llm,
            problem=problem,
            agent_usage_variant=agent_usage_variant,
            cycle_save_dir=cycle_save_dir,
            experiment_state_store=experiment_state_store,
            experiment_execution_uuid=experiment_execution_uuid,
        )
        logger.info(f"Cycle {conversation_cycle_num:02}: {cycle_log_data['exit_stage']}")

        # Save the cycle data
        (cycle_save_dir / "cycle_log.json").write_bytes(
            orjson.dumps(cycle_log_data, option=orjson.OPT_INDENT_2)
        )
        with open(experiment_save_dir / "cycle_log.jsonl", "ab") as f:
            f.write(orjson.dumps(cycle_log_data) + b"\n")
        (cycle_save_dir / "experiment_state_exit.json").write_bytes(
            orjson.dumps(experiment_state_store.to_dict(), option=orjson.OPT_INDENT_2)
        )

        experiment_data["used_cycle_count"] = conversation_cycle_num + 1
        experiment_data["latest_exit_stage"] = cycle_log_data["exit_stage"]

        if cycle_log_data["exit_stage"] == "80_tb_passed":
            experiment_data["was_testbench_passed"] = True
            break

    write_llm_conversation_to_files(
        pydash.flatten(experiment_state_store.conversation_histories.values()),
        experiment_save_dir,
    )

    merge_jsonl_to_parquet(experiment_save_dir / "cycle_log.jsonl")

    return experiment_data


def run_experiment_all_inputs(
    llm_config_file_path: str | Path, agent_usage_variant_name: str
) -> None:
    logger.info(f"Starting: {__file__} -> run_experiment_all_inputs(...)")
    logger.info(f"Using LLM config file: {llm_config_file_path}")
    agent_usage_variant = get_named_agent_usage_variant(agent_usage_variant_name)
    logger.info(f"Agent usage variant '{agent_usage_variant_name}': {agent_usage_variant}")

    experiment_group_start_timestamp = datetime.now()
    experiment_group_start_timestamp_str = get_file_date_str(
        experiment_group_start_timestamp, precision="datetime"
    )

    # Data Setup
    experiment_name = f"{Path(__file__).stem}_{agent_usage_variant_name}"
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
    with open(working_dir / "agent_usage_variant.json", "wb") as f:
        f.write(orjson.dumps(agent_usage_variant, option=orjson.OPT_INDENT_2) + b"\n")

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
            problems,
            desc=f"'{llm.configured_llm_name}' LLM (#{llm_num}/{len(llm_list)}) solving problems",
            unit="problem",
        ):
            logger.info(f"Running experiment for {llm=}, {problem=}")
            global_stats["total_experiment_count"] += 1

            assert isinstance(problem, SimpleCodeGenProblem)
            assert problem.has_testbench_code

            try:
                experiment_data = do_experiment(
                    llm=llm,
                    problem=problem,
                    agent_usage_variant=agent_usage_variant,
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

            logger.info("=" * 80)  # print a break

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


def main_cli():
    parser = argparse.ArgumentParser(
        description="Run the LLM code generation debugger agent experiment."
    )
    parser.add_argument(
        "-l",
        "--llm-config",
        dest="llm_config_file_path",
        type=str,
        help="The path to the YAML file containing the LLM configuration.",
    )
    parser.add_argument(
        "-a",
        "--agent-variant",
        dest="agent_usage_variant_name",
        type=str,
        choices=[v.name for v in agent_usage_variants],
        help="The name of the agent usage variant to use in the experiment.",
    )
    args = parser.parse_args()

    run_experiment_all_inputs(
        llm_config_file_path=Path(args.llm_config_file_path),
        agent_usage_variant_name=args.agent_usage_variant_name,
    )


if __name__ == "__main__":
    main_cli()
