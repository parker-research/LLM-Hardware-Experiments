from pathlib import Path

from llm_experiments.llms.llm_types import LlmPrompt, LlmResponse


def write_llm_conversation_to_files(
    llm_history: list[LlmPrompt | LlmResponse], llm_log_folder: Path
) -> None:
    """Write the LLM conversation history to files in a log folder.
    If the files already exist, they will be wiped and rewritten.
    """

    assert all(isinstance(item, (LlmPrompt, LlmResponse)) for item in llm_history)
    assert isinstance(llm_log_folder, Path)
    assert llm_log_folder.is_dir()

    llm_history = sorted(llm_history, key=lambda x: x.timestamp)

    # write to unified log
    with open(llm_log_folder / "llm_log_all.txt", "w") as f:
        for item in llm_history:
            f.write(str(item) + "\n\n")
    with open(llm_log_folder / "llm_log_all.jsonl", "wb") as f:
        for item in llm_history:
            f.write(item.to_json() + b"\n")

    # write to per-agent logs
    for agent_name in set(item.agent_name for item in llm_history if item.agent_name):
        with open(llm_log_folder / f"llm_log_{agent_name}.txt", "w") as f:
            for item in llm_history:
                if item.agent_name == agent_name:
                    f.write(str(item) + "\n\n")
        with open(llm_log_folder / f"llm_log_{agent_name}.jsonl", "wb") as f:
            for item in llm_history:
                if item.agent_name == agent_name:
                    f.write(item.to_json() + b"\n")


def append_llm_conversation_to_files(
    llm_history: list[LlmPrompt | LlmResponse], llm_log_folder: Path
) -> None:
    """Write the LLM conversation history to files in a log folder.
    If the files already exist, they will be appended to.
    """

    assert all(isinstance(item, (LlmPrompt, LlmResponse)) for item in llm_history)
    assert isinstance(llm_log_folder, Path)
    assert llm_log_folder.is_dir()

    llm_history = sorted(llm_history, key=lambda x: x.timestamp)

    for item in llm_history:
        # write to per-agent logs
        with open(llm_log_folder / f"llm_log_{item.agent_name}.txt", "a") as f:
            f.write(str(item) + "\n\n")
        with open(llm_log_folder / f"llm_log_{item.agent_name}.jsonl", "ab") as f:
            f.write(item.to_json() + b"\n")
        # write to unified log
        with open(llm_log_folder / "llm_log_all.txt", "a") as f:
            f.write(str(item) + "\n\n")
        with open(llm_log_folder / "llm_log_all.jsonl", "ab") as f:
            f.write(item.to_json() + b"\n")
