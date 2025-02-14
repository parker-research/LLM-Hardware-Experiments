"""\
This experiment aims to assess how capable various LLMs are at generating code for the select
Verilog problems, by experimenting with LangChain's RAG (Retriever-Answerer-Generator)
architecture/capabilities.

Play around with LangChain.
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
import langchain
from tqdm import tqdm
import yaml

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
from llm_experiments.util.path_helpers import get_path_to_git_repo_root


import pydash

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough


def load_documents() -> list[Document]:
    df = pl.read_parquet("/home/user/files/CISH/Verilog_GitHub/categorized.pq")
    documents = [
        Document(
            page_content=row["content"],
            metadata=(pydash.omit(row, "content")),
        )
        for row in df.iter_rows(named=True)
    ]
    return documents


iverilog_tool = IverilogTool(
    config=IverilogToolConfig(
        configured_tool_name="iverilog_v20240501",
        release_version_date=date(2024, 5, 1),
    )
)


def generate_solution(problem: str) -> str:
    # Load the OpenAI api key.
    secrets = yaml.safe_load((get_path_to_git_repo_root() / "secrets.yaml").read_text())
    openai_api_key = secrets["openai_api_key"]

    llm = ChatOpenAI(openai_api_key=openai_api_key)

    # Tutorial: https://python.langchain.com/v0.2/docs/tutorials/retrievers/

    documents = load_documents()

    vector_store = Chroma.from_documents(
        documents,
        embedding=OpenAIEmbeddings(),
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an A+ student demonstrating your knowledge of Verilog by solving problems.",
            ),
            (
                "human",
                """
                Solve the following problem by writing Verilog code which completes the template
                given below. If helpful, you may refer to the real-world code snippets, which may
                or may not be relevant to the problem.

                {question}

                Examples which may help you answer this problem:
                {context}
                """.strip(),
            ),
        ]
    )

    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

    invoke_output = chain.invoke({"question": "What is the color of the sky?"})
    logger.info(f"{type(invoke_output)}: {invoke_output}")


if __name__ == "__main__":
    basic_test()
