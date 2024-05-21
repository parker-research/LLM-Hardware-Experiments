from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal, Optional
from datetime import date

from llm_experiments.feedback_eval_tools.feedback_eval_tool_base import (
    FeedbackEvalToolConfigBase,
    FeedbackEvalToolBase,
)
from llm_experiments.util.execute_cli import run_command, CommandResult
from llm_experiments.feedback_eval_tools.install_oss_cad import install_oss_cad_and_activate


@dataclass(kw_only=True)
class IverilogToolConfig(FeedbackEvalToolConfigBase):
    syntax_version: Literal["sv2012"] = "sv2012"  # TODO: maybe add support for others
    show_warnings: bool = True
    release_version_date: Optional[date]

    def to_command_args(self) -> list[str]:
        args = []

        if self.syntax_version == "sv2012":
            args.append("-g2012")
        else:
            raise ValueError(f"Unsupported syntax version: {self.syntax_version}")

        if self.show_warnings:
            args.append("-Wall")
        return args


class IverilogTool(FeedbackEvalToolBase):
    """Implementation of access to the IVerilog tool."""

    def __init__(self, config: IverilogToolConfig):
        self.config: IverilogToolConfig = config
        super().__init__()

    def install_and_init_tool(self) -> None:
        install_oss_cad_and_activate(self.config.release_version_date)

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
        raise NotImplementedError
        return 0.5

    def evaluate_query(self, query: dict) -> dict:
        # FIXME: come up with a reasonable way to implement this
        raise NotImplementedError
        return {"response": "mock response"}

    def run_iverilog_compile_command(
        self, verilog_file_list: list[Path], output_vvp_file_path: Path
    ) -> CommandResult:
        compile_command = (
            ["iverilog"]
            + self.config.to_command_args()
            + ["-o", str(output_vvp_file_path)]
            + verilog_file_list
        )
        result = run_command(compile_command)
        return result

    def run_iverilog_vvp_execute_command(self, vvp_file_path: Path) -> CommandResult:
        execute_command = ["vvp", str(vvp_file_path)]
        result = run_command(execute_command)
        return result

    def run_iverilog_compile_and_execute_commands(
        self, verilog_file_list: list[Path], output_vvp_file_path: Path
    ) -> dict:
        """Runs the IVerilog compile and IVerilog vvp (execute) commands.
        Returns a dict with keys:
            'compile_result': CommandResult
            'execute_result': CommandResult
            'status': Literal["success", "compile_error", "execute_error"]
        """
        result = {
            "compile_result": None,
            "execute_result": None,
            "status": "success",
        }
        result["compile_result"] = self.run_iverilog_compile_command(
            verilog_file_list, output_vvp_file_path
        )
        if result["compile_result"].return_code != 0:
            result["status"] = "compile_error"
            return result

        result["execute_result"] = self.run_iverilog_vvp_execute_command(output_vvp_file_path)
        if result["execute_result"].return_code != 0:
            result["status"] = "execute_error"
            return result

        # TODO: check testbench output
        return result

    def get_iverilog_version(self) -> list[dict[str, str]]:
        """Get the version of IVerilog."""
        result = run_command(["iverilog", "-V"])
        return self._extract_iverilog_version(result.stdout)

    @staticmethod
    def _extract_iverilog_version(version_command_stdout: str) -> list[dict[str, str]]:
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
