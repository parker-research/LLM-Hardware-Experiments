from pathlib import Path
import re

from llm_experiments.feedback_eval_tools.feedback_eval_tool_base import (
    FeedbackEvalToolBase,
)
from llm_experiments.util.execute_cli import run_command, CommandResult
from llm_experiments.feedback_eval_tools.install_oss_cad import install_oss_cad


class IverilogTool(FeedbackEvalToolBase):
    """Implementation of access to the IVerilog tool."""

    def __init__(self, configured_tool_name: str, config: dict = {}):
        super().__init__(configured_tool_name, config)

    @classmethod
    def validate_configuration(cls, config: dict):
        assert isinstance(config, dict)

    def install_and_init_tool(self) -> None:
        install_oss_cad()

    def assert_is_usable(self):
        versions = self.get_iverilog_version()

        assert (
            len(versions) >= 4
        ), f"IVerilog is not installed ({len(versions)} out of 4 tools found)."

        iverilog_test = run_command(["iverilog", "-V"])  # -h returns exit>0
        assert iverilog_test.return_code == 0

        vvp_test = run_command(["vvp", "-V"])  # -V and -h are both good
        assert vvp_test.return_code == 0

    def evaluate(self, evaluation_info: dict) -> float:
        # FIXME: doesn't really make sense
        return 0.5

    def evaluate_query(self, query: dict) -> dict:
        # FIXME: doesn't really make sense
        return {"response": "mock response"}

    def run_iverilog_compile_command(
        self, verilog_file_list: list[Path], output_vvp_file_path: Path
    ) -> CommandResult:
        compile_command = ["iverilog", "-o", str(output_vvp_file_path)] + verilog_file_list
        result = run_command(compile_command)
        return result

    def run_iverilog_vvp_execute_command(self, vvp_file_path: Path) -> CommandResult:
        execute_command = ["vvp", str(vvp_file_path)]
        result = run_command(execute_command)
        return result

    def get_iverilog_version(self) -> list[dict[str, str]]:
        """Get the version of IVerilog."""
        result = run_command(["iverilog", "-V"])
        return self._extract_iverilog_version(result.stdout)

    @staticmethod
    def _extract_iverilog_version(version_command_stdout) -> list[dict[str, str]]:
        matches = re.finditer(
            (
                r"^\s*(?P<tool_name>.*Icarus.+?)\s+([vV]ersion\s+)?"
                r"(?P<full_version_id>(?P<version_num>\d+\.\d+(\.\d+)?).*)$"
            ),
            version_command_stdout,
            re.MULTILINE,  # multiline mode: ^ and $ match start/end of lines
        )

        versions: list[dict[str, str]] = []
        for m in matches:
            versions.append(m.groupdict())
        return versions
