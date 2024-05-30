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
    return_code: int | None  # None if timed out
    execution_duration: timedelta
    other_info: Optional[dict] = None
    command_step_name: Optional[str] = None  # like "{self.command_step_name}" or "execute"
    timed_out: bool

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
            if self.timed_out:
                text_parts[0] += " (failed, timed out)"
            else:
                text_parts[0] += " (failed)"
            # TODO: check if we can include more detail in the response with the return code
        else:
            text_parts[0] += " (success)"

        if self.stdout:
            text_parts.append("Command stdout:\n" + self.clean_stdout)
        if self.stderr:
            text_parts.append("Command stderr:\n" + self.clean_stderr)

        return "\n\n".join(text_parts)

    def as_dict(self) -> dict:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "execution_duration": str(self.execution_duration),
            "execution_duration_sec": self.execution_duration.total_seconds(),
            "timed_out": self.timed_out,
        }


def run_command(
    command: list[str | Path], max_execution_time: Optional[timedelta] = None
) -> CommandResult:
    """Run a command (e.g., iverilog) and return the output."""

    for command_part in command:
        assert isinstance(command_part, (str, Path)), f"Invalid command part: {command_part}"

    command = [str(cmd) for cmd in command]

    start_time = datetime.now()

    timeout_seconds = max_execution_time.total_seconds() if max_execution_time else None
    try:
        process_result = subprocess.run(
            command,
            capture_output=True,
            timeout=timeout_seconds,
        )
        timed_out = False
    except subprocess.TimeoutExpired as e:
        process_result = (
            e  # Note: this result has the same stdout/stderr/returncode as if it had completed
        )
        timed_out = True

    result = CommandResult(
        stdout=process_result.stdout.decode("utf-8") if process_result.stdout else "",
        stderr=process_result.stderr.decode("utf-8") if process_result.stderr else "",
        return_code=process_result.returncode if not timed_out else None,  # None if timed out
        execution_duration=datetime.now() - start_time,
        timed_out=timed_out,
    )
    return result
