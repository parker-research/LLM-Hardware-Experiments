import subprocess
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta

# Tools for executing external CLI tools (e.g., verilator, iverilog, etc.).


@dataclass
class CommandResult:
    """The result of running a command."""

    stdout: str
    stderr: str
    return_code: int
    execution_duration: timedelta
    other_info: dict = None


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
