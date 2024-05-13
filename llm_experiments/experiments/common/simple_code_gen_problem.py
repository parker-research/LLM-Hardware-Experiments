from typing import Optional, Any, Literal
from pathlib import Path
import re


class SimpleCodeGenProblem:
    """A simple code generation problem."""

    def __init__(
        self,
        *,
        problem_id: str,
        problem_description: str,
        module_header: Optional[str] = None,
        testbench_code: Optional[str] = None,
        canonical_solution: Optional[str] = None,
        code_language: Literal["verilog", "systemverilog", "vhdl"] = "systemverilog",
        other_data: dict[str, Any] = {},
    ):
        self.problem_id = problem_id
        self.problem_description = problem_description
        self.module_header = module_header
        self._testbench_code = testbench_code
        self._canonical_solution = canonical_solution
        self.code_language = code_language
        self.other_data = other_data

    @staticmethod
    def _fully_qualify_file_names(vcd_folder_path: Path, source_code: str) -> str:
        """Fully qualify the file names in the code.
        Returns the source code with the file names fully qualified.
        """
        file_names = re.findall(r"\"(?P<file_name>\w+\.vcd)\"", source_code)
        for file_name in set(file_names):
            source_code = source_code.replace(
                f'"{file_name}"', '"' + str((vcd_folder_path.absolute() / file_name)) + '"'
            )
        return source_code

    def get_testbench_code(self, vcd_folder_path: Path) -> str | None:
        """Return the testbench code, fully qualified with the folder path."""
        if not self.has_testbench_code:
            return None
        return self._fully_qualify_file_names(vcd_folder_path, self._testbench_code)

    def get_canonical_solution(self, vcd_folder_path: Path) -> str | None:
        """Return the canonical solution, fully qualified with the folder path."""
        if not self.has_canonical_solution:
            return None
        return self._fully_qualify_file_names(vcd_folder_path, self._canonical_solution)

    @property
    def has_canonical_solution(self) -> bool:
        return self._canonical_solution is not None

    @property
    def has_testbench_code(self) -> bool:
        return self._testbench_code is not None

    @property
    def available_data_flags(self) -> list[str]:
        flags = []
        if self.has_canonical_solution:
            flags.append("has_canonical_solution")
        if self.has_testbench_code:
            flags.append("has_testbench_code")
        return flags

    def __repr__(self) -> str:
        args_part = ", ".join([self.problem_id] + self.available_data_flags)
        parts = ["SimpleCodeGenProblem(", args_part, ")"]
        return "".join(parts)

    def __str__(self) -> str:
        args_part = ", ".join([self.problem_id] + self.available_data_flags)
        parts = ["SimpleCodeGenProblem(", args_part, ")"]
        return "".join(parts)

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "problem_description": self.problem_description,
            "module_header": self.module_header,
            "testbench_code": self._testbench_code,
            "canonical_solution": self._canonical_solution,
            "code_language": self.code_language,
            "other_data": self.other_data,
        }
