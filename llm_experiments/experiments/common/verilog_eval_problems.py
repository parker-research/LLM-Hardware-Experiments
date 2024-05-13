from pathlib import Path
import re
import orjson

import git
from loguru import logger
import polars as pl

from llm_experiments.experiments.common.simple_code_gen_problem import SimpleCodeGenProblem
from llm_experiments.util.path_helpers import make_data_dir


def get_verilog_eval_repo() -> Path:
    """Returns the path to the verilog-eval repo.
    If the repo is not cloned, clones it to <this_repo>/working/datasets/verilog-eval.
    """
    verilog_eval_git_root = make_data_dir("datasets/verilog-eval", append_date=False)

    if not (verilog_eval_git_root / ".git").is_dir():
        logger.info("Did not find verilog-eval repo. Cloning the verilog-eval repo.")

        # clone the repo
        repo = git.Repo.clone_from(
            # "https://github.com/NVlabs/verilog-eval",
            "https://github.com/parker-research/verilog-eval",
            to_path=verilog_eval_git_root,
        )
        logger.info(
            f"Cloned the verilog-eval repo. "
            f"Active branch: {repo.active_branch}. Latest commit: {repo.head.commit}"
        )
    else:
        logger.info("verilog-eval repo already exists.")
    return verilog_eval_git_root


def load_verilog_eval_problems() -> list[SimpleCodeGenProblem]:
    """Loads the problems from the verilog-eval repo.
    If the repo is not cloned, it clones it first to <this_repo>/../verilog-eval.
    """

    verilog_eval_git_root = get_verilog_eval_repo()

    df_desc = pl.read_ndjson(verilog_eval_git_root / "descriptions/VerilogDescription_Human.jsonl")
    df_solution = pl.read_ndjson(verilog_eval_git_root / "data/VerilogEval_Human.jsonl")
    assert set(df_desc.columns) == {"task_id", "detail_description"}
    assert set(df_solution.columns) == {"task_id", "prompt", "canonical_solution", "test"}

    logger.info(f"Loaded verilog-eval data: {len(df_desc)=:,}, {len(df_solution)=:,}")
    assert len(df_desc) == len(df_solution)

    df = df_desc.join(df_solution, on="task_id", validate="1:1", how="inner")

    problems: list[SimpleCodeGenProblem] = []
    for row in df.iter_rows(named=True):
        problem = SimpleCodeGenProblem(
            problem_id="verilog-eval_" + row["task_id"],
            problem_description=row["detail_description"],
            module_header=row["prompt"],
            # In source, "canonical_solution" doesn't include the module_header ("prompt")
            canonical_solution=(
                row["prompt"].strip()
                + "\n\n\t// BREAK between verilog_eval 'prompt' and 'canonical_solution'\n\n"
                + row["canonical_solution"].strip()
                + "\n"
            ),
            testbench_code=row["test"],
            code_language="systemverilog",
        )
        problems.append(problem)

    # log some stats
    num_problems = len(problems)
    logger.info(f"Loaded {num_problems:,} problems from verilog-eval.")
    df = pl.DataFrame([problem.to_dict() for problem in problems])
    logger.info(f"Problems, as a table: {df}")

    return problems


def parse_verilog_eval_testbench_output(testbench_output: str) -> dict:
    """Parses the output of the verilog-eval testbench.

    Args:
        testbench_output: The output of the testbench (execute_result.stdout).

    Returns:
        A dictionary with the following keys:
            - "testbench_stats": A JSON string, like:
                {"mismatch_sample_count": 0, "total_sample_count": 439}
            - "was_testbench_passed": A boolean, indicating if the testbench passed.
    """

    # Find "Hint: Total mismatched samples is 0 out of 439 samples"
    tb_match = re.search(
        r"Total mismatched samples is (?P<mismatch_sample_count>\d+) out of (?P<total_sample_count>\d+) samples",  # noqa
        testbench_output,
        re.IGNORECASE,
    )
    if not tb_match:
        raise ValueError("Could not find mismatched sample count in testbench output.")

    parse_output = {
        "testbench_stats": None,
        "was_testbench_passed": None,
    }
    mismatch_sample_count = int(tb_match.group("mismatch_sample_count"))
    total_sample_count = int(tb_match.group("total_sample_count"))
    parse_output["testbench_stats"] = orjson.dumps(
        {
            "mismatch_sample_count": mismatch_sample_count,
            "total_sample_count": total_sample_count,
        }
    ).decode()
    parse_output["was_testbench_passed"] = mismatch_sample_count == 0
    return parse_output
