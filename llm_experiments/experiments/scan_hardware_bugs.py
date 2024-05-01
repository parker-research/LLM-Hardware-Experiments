"""\
A bug scanner, created for detecting bugs in hardware implementations of large hardware design
projects.
"""

from pathlib import Path
import uuid
from datetime import datetime
import traceback

import orjson
from loguru import logger
import polars as pl
import fire

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


from llm_experiments.intermediate_steps.split_large_files import (
    split_large_file_by_large_line_breaks,
)
from llm_experiments.intermediate_steps.line_numbering import add_line_numbers

from llm_experiments.llms.llm_base import LlmBase
from llm_experiments.llms.llm_types import LlmPrompt, LlmResponse
from llm_experiments.llms.models.mock_llm import MockLlm, MockLlmConfig  # noqa

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


def scan_file_for_bugs(
    llm: LlmBase,
    file_path: Path,
    working_dir: Path,
    logging_attributes: dict = {},
) -> dict:
    """
    Scans a single file for bugs.
    """

    experiment_execution_uuid = uuid.uuid4()

    # Prep the return data
    experiment_save_dir = (
        working_dir
        / f"bug_scan_{llm.configured_llm_name}"
        / f"{file_path.name}_{experiment_execution_uuid}"
    )
    (experiment_inputs_dir := experiment_save_dir / "inputs").mkdir(parents=True)
    (experiment_outputs_dir := experiment_save_dir / "outputs").mkdir(parents=True)

    file_contents = file_path.read_text()

    chunks = split_large_file_by_large_line_breaks(
        file_contents, min_lines_per_chunk=8, max_lines_per_chunk=20
    )
    start_of_chunk_line_number = 0
    for chunk in chunks:
        chunk_file_path = experiment_inputs_dir / f"{start_of_chunk_line_number}_{file_path.name}"
        chunk_file_path.write_text(chunk)
        start_of_chunk_line_number += len(chunk.split("\n"))

        file_contents_with_line_numbers = add_line_numbers(
            chunk, location="prefix", start_line_number=start_of_chunk_line_number
        )
        chunk_file_path_with_line_numbers = (
            experiment_inputs_dir
            / f"{start_of_chunk_line_number}_{file_path.stem}_numbered.{file_path.suffix}"
        )
        chunk_file_path_with_line_numbers.write_text(file_contents_with_line_numbers)

    # TODO: deal with long files

    # Step 1: Request that the LLM generates code from the problem prompt
    # TODO: assess prompt engineering strategies to improve performance here
    llm_prompt_text = (
        "You are the designer of the OpenTitan Hardware Root of Trust. "
        "A hacker has forked your repository, introduced several security-related mechanisms, and "
        "and has made changes to the code. In these changes, they introduced exploitable bugs, "
        "vulnerabilities, backdoors, etc.\n\n"
        "In the following code snippet, please identify and describe the bugs. Include the line "
        "number, and a brief description of the bug. Each line is prefixed with its line number. "
        "At the end of your response, you must include 'BUG FOUND' or 'BUG NOT FOUND'. \n\n"
        f"{file_contents_with_line_numbers}"
    )
    # TODO: could experiment with bug found/not found at the end of the prompt
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

    # log it
    logger.info(f"LLM Prompt: {llm_response.response_text}")

    if "bug not found" in llm_response.response_text.lower():
        logger.info(f"❌ LLM Response: {llm_response.response_text}")
        bug_found_status = "BUG NOT FOUND"
    elif "bug found" in llm_response.response_text.lower():
        logger.info(f"🐞 LLM Response: {llm_response.response_text}")
        bug_found_status = "BUG FOUND"
    else:
        logger.info(f"🤷 LLM Response: {llm_response.response_text}")
        bug_found_status = "UNKNOWN"

    experiment_data = {
        "experiment_group_start_timestamp": logging_attributes["experiment_group_start_timestamp"],
        "experiment_execution_uuid": str(experiment_execution_uuid),
        "base_llm_name": llm.base_llm_name,
        "configured_llm_name": llm.configured_llm_name,
        "target_file_path": str(file_path),
        "target_file_name": file_path.name,
        "llm_configuration": orjson.dumps(llm.config.to_dict()).decode(),
        "llm_prompt": llm_prompt_text,
        "llm_response": llm_response.response_text,
        "llm_response_metadata": orjson.dumps(llm_response.metadata).decode(),
        "experiment_save_dir": str(experiment_save_dir),
        "bug_found_status": bug_found_status,
    }

    return experiment_data


def run_scanner_as_experiment(project_dir_to_scan: Path, llama_model_name: str):
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
        OllamaLlm(
            llama_model_name.replace(":", "_"),
            config=OllamaLlmConfig(
                model_name=llama_model_name,
            ),
        ),
    ]

    for llm in llm_list:
        llm.init_model()
        logger.info(f"Initialized LLM: {llm}")

        target_files = list(project_dir_to_scan.glob("**/*.v")) + list(
            project_dir_to_scan.glob("**/*.sv")
        )
        for target_file_path in target_files:
            logger.info(f"Running experiment for {llm=}, {target_file_path=}")
            global_stats["total_experiment_count"] += 1

            try:
                experiment_data = scan_file_for_bugs(
                    llm=llm,
                    file_path=target_file_path,
                    working_dir=working_dir,
                    logging_attributes={
                        "experiment_group_start_timestamp": experiment_group_start_timestamp
                    },
                )
            except Exception as e:
                logger.error(f"Error in experiment: {target_file_path=}, {e=}")
                logger.error(traceback.format_exc())
                experiment_data = {
                    "experiment_group_start_timestamp": experiment_group_start_timestamp,
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
                    "target_file_name",
                    "bug_found_status",
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
    fire.Fire(run_scanner_as_experiment)