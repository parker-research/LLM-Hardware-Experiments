from pathlib import Path
from dataclasses import dataclass
import uuid
from datetime import datetime, timezone

import orjson
from loguru import logger

from llm_experiments.llms.llm_base import LlmBase
from llm_experiments.llms.llm_types import LlmPrompt, LlmResponse
from llm_experiments.llms.models.mock_llm import MockLlm, MockLlmConfig
from llm_experiments.llms.models.ollama_llm import OllamaLlm, OllamaLlmConfig, solid_configs
from llm_experiments.util.path_helpers import (
    make_data_dir,
    get_path_to_git_repo_root,
    get_file_date_str,
)
from llm_experiments.feedback_eval_tools.tools.iverilog_tool import IverilogTool

iverilog_tool = IverilogTool("iverilog", config={})


@dataclass
class SimpleCodeGenProblem:
    """A simple code generation problem."""

    problem_id: str
    problem_prompt: str

    def __repr__(self) -> str:
        return f"SimpleCodeGenProblem({self.problem_id})"

    def __str__(self) -> str:
        return f"SimpleCodeGenProblem({self.problem_id})"


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

    # Step 1: Generate code
    llm_prompt_text = (
        "You are a student learning Verilog. "
        "Solve the following problem by writing a Verilog module. "
        "Wrap your code in Markdown code blocks. "
        "Do not write anything extraneous to the Verilog solution. "
        "Your solution should be a Verilog module that meets the following requirements: "
        f"{problem.problem_prompt}"
    )
    llm_response: LlmResponse = llm.query_llm_basic(LlmPrompt(llm_prompt_text))

    # Step 2: Extract the generated code
    generated_code = llm_response.extract_code("verilog_module")
    if generated_code is None:
        # TODO: handle logging/data collection here
        return

    # Step 3: Save the generated code to a file
    verilog_save_dir = (
        working_dir / "verilog" / f"{problem.problem_id}_{experiment_execution_uuid}"
    )
    verilog_save_dir.mkdir(parents=True, exist_ok=False)

    verilog_file_path = verilog_save_dir / "generated_code.v"
    verilog_file_path.write_text(generated_code)

    # Step 4: Run the code through IVerilog to check if it builds
    vvp_file_path = verilog_save_dir / "compiled.vvp"
    compile_result = iverilog_tool.run_iverilog_compile_command(
        verilog_file_list=[verilog_file_path],  # TODO: add a test bench path here
        output_vvp_file_path=vvp_file_path,
    )

    was_compile_success = compile_result.return_code == 0

    # Step 5: Run the code through IVerilog's vvp to run it, with the test bench
    # execute_result = iverilog_tool.run_iverilog_vvp_execute_command(vvp_file_path)
    # TODO: run this part, with the test bench

    # Step 6: Collect data on the results
    experiment_data = {
        "experiment_group_start_timestamp": logging_attributes["experiment_group_start_timestamp"],
        "experiment_execution_uuid": str(experiment_execution_uuid),
        "base_llm_name": llm.base_llm_name,
        "configured_llm_name": llm.configured_llm_name,
        "llm_configuration": orjson.dumps(llm.config.to_dict()).decode("utf-8"),
        "problem_id": problem.problem_id,
        "problem_prompt": problem.problem_prompt,
        "llm_response": llm_response.response_text,
        "llm_response_metadata": orjson.dumps(llm_response.metadata).decode("utf-8"),
        "extracted_generated_code": generated_code,
        "verilog_save_dir": str(verilog_save_dir),
        "iverilog_compile_result_return_code": compile_result.return_code,
        "iverilog_compile_result_stdout": compile_result.stdout,
        "iverilog_compile_result_stderr": compile_result.stderr,
        "was_compile_success": was_compile_success,
    }
    return experiment_data


def load_problems() -> list[SimpleCodeGenProblem]:
    git_root = get_path_to_git_repo_root()
    verilog_eval_problems_path = (
        git_root.parent / "verilog-eval" / "descriptions" / "VerilogDescription_Human.jsonl"
    )

    problems: list[SimpleCodeGenProblem] = []
    with open(verilog_eval_problems_path, "r") as f:
        for line in f:
            problem_dict = orjson.loads(line)
            problem = SimpleCodeGenProblem(
                problem_id="verilog_eval__" + problem_dict["task_id"],
                problem_prompt=problem_dict["detail_description"],
            )
            problems.append(problem)

    return problems


def run_experiment_all_inputs():
    experiment_group_start_timestamp = datetime.now(timezone.utc)
    experiment_group_start_timestamp_str = get_file_date_str(
        experiment_group_start_timestamp, precision="datetime"
    )

    # Tool Setup
    iverilog_tool.install_and_init_tool()
    iverilog_tool.assert_is_usable()

    # LLM Setup
    llm_list: list[LlmBase] = [
        MockLlm(
            "mock_llm_no_preprogrammed_responses",
            config=MockLlmConfig(
                does_respond_to_test_queries=False,
            ),
        ),
        # OllamaLlm(
        #     "llama2_7b_no_randomness",
        #     config=solid_configs["llama2_7b_no_randomness"],
        # ),
        # OllamaLlm(
        #     "tinyllama_no_randomness",
        #     config=solid_configs["tinyllama_no_randomness"],
        # ),
    ]

    # Data Setup
    working_dir = make_data_dir(
        f"simple_code_gen_experiment_{experiment_group_start_timestamp_str}",
        append_date=False,
    )

    # Problems
    problems = load_problems()
    logger.info(f"Loaded {len(problems)} problems.")

    # experiment_space = itertools.product([llm_list, problems])

    for llm in llm_list:
        llm.init_model()
        logger.info(f"Initialized LLM: {llm}")

        for problem in problems:
            logger.info(f"Running experiment for problem: {problem}")

            experiment_data = do_experiment(
                llm=llm,
                problem=problem,
                working_dir=working_dir,
                logging_attributes={
                    "experiment_group_start_timestamp": experiment_group_start_timestamp
                },
            )

            with open(working_dir / "experiment_data.json", "ab") as f:
                f.write(orjson.dumps(experiment_data))
                f.write(b"\n")

        logger.info(f"Finished experiments for LLM: {llm}")

        # TODO: maybe enable this
        # llm.destroy_model()

    logger.info("Finished all experiments.")


if __name__ == "__main__":
    run_experiment_all_inputs()
