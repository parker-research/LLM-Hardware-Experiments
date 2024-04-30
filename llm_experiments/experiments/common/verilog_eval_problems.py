from pathlib import Path

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
            # canonical_solution=row.get("canonical_solution"),
            # testbench_code=row.get("test"),
            module_header=row["prompt"],
            # currently, canonical_solution doesn't include the module header
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
    df = pl.DataFrame(problems)
    logger.info(f"Problems, as a table: {df}")

    return problems
