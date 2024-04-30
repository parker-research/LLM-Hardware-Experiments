from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass(kw_only=True)
class SimpleCodeGenProblem:
    """A simple code generation problem."""

    problem_id: str
    problem_description: str

    module_header: Optional[str] = None
    canonical_solution: Optional[str] = None
    testbench_code: Optional[str] = None

    other_data: dict[str, Any] = field(default_factory=dict)  # default: empty dict

    # TODO: track the language of the code (e.g., "verilog", "vhdl", "systemverilog")

    @property
    def has_canonical_solution(self) -> bool:
        return self.canonical_solution is not None

    @property
    def has_testbench_code(self) -> bool:
        return self.testbench_code is not None

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
