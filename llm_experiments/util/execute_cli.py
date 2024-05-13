import subprocess
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from llm_experiments.util.strip_paths import strip_paths_from_text

# Tools for executing external CLI tools (e.g., verilator, iverilog, etc.).


@dataclass(kw_only=True)
class CommandResult:
    """The result of running a command."""

    stdout: str
    stderr: str
    return_code: int
    execution_duration: timedelta
    other_info: Optional[dict] = None
    command_step_name: Optional[str] = None  # like "{self.command_step_name}" or "execute"

    def write_to_files(self, folder_path: Path) -> None:
        """Write the stdout and stderr to files in the specified folder."""
        assert self.command_step_name is not None, "command_step_name must be set to write files"
        folder_path.mkdir(parents=True, exist_ok=True)
        (
            folder_path / f"cmd_{self.command_step_name}_stdout_ret{self.return_code}.txt"
        ).write_text(self.stdout)
        (
            folder_path / f"cmd_{self.command_step_name}_stderr_ret{self.return_code}.txt"
        ).write_text(self.stderr)

    def as_update_dict(self) -> dict:
        assert self.command_step_name is not None, "command_step_name must be set"
        return {
            f"{self.command_step_name}_result_return_code": (self.return_code),
            f"{self.command_step_name}_result_stdout": (self.stdout),
            f"{self.command_step_name}_result_stderr": (self.stderr),
            f"was_{self.command_step_name}_success": (self.return_code == 0),
        }

    @property
    def clean_stdout(self) -> str:
        return strip_paths_from_text(self.stdout)

    @property
    def clean_stderr(self) -> str:
        return strip_paths_from_text(self.stderr)

    def to_llm_text(self) -> str:
        """Return a text summary of the command result, for use in LLM responses."""
        text_parts = [f"Command return code: {self.return_code}"]

        if self.return_code != 0:
            text_parts[0] += " (failed)"
            # TODO: check if we can include more detail in the response with the return code

        if self.stdout:
            text_parts.append("Command stdout:\n" + self.clean_stdout)
        if self.stderr:
            text_parts.append("Command stderr:\n" + self.clean_stderr)

        return "\n\n".join(text_parts)


def run_command(command: list[str | Path]) -> CommandResult:
    """Run a command (e.g., iverilog) and return the output.
    Do not include the 'iverilog' command itself in the command list.
    """

    for command_part in command:
        assert isinstance(command_part, (str, Path)), f"Invalid command part: {command_part}"

    command = [str(cmd) for cmd in command]

    start_time = datetime.now()
    process_result = subprocess.run(
        command,
        capture_output=True,
    )

    result = CommandResult(
        stdout=process_result.stdout.decode("utf-8"),
        stderr=process_result.stderr.decode("utf-8"),
        return_code=process_result.returncode,
        execution_duration=datetime.now() - start_time,
    )
    return result
