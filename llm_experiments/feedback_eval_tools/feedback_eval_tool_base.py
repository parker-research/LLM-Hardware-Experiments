import abc


class FeedbackEvalToolBase(abc.ABC):
    """A base class for feedback evaluation tools. Implementations should inherit from this class.

    Examples of feedback evaluation tools include:
        - Icarus Verilog (Iverilog)
        - Verilator
        - JasperGold
        - A Language Server Protocol (LSP) server for syntax checking, highlighting, etc.
        - MyHDL (if using Python's MyHDL)
        - RustHDL (if using Rust's RustHDL)
        - Other synthesis tools
    """

    def __init__(self, configured_tool_name: str, config: dict = {}):
        self.configured_tool_name: str = configured_tool_name
        self.base_tool_name: str = self.__class__.__name__

        self.validate_configuration(config)
        self.config: dict = config

    def __repr__(self) -> str:
        return f"{self.base_tool_name}, acting as {self.configured_tool_name} ({self.config})"

    @classmethod
    @abc.abstractmethod
    def validate_configuration(cls, config: dict):
        """Check that the configuration options are valid. Raise an exception if not."""
        pass

    @abc.abstractmethod
    def install_and_init_tool(self) -> None:
        """Prep the tool for use right away.
        May involve downloading the tool's files, etc.
        Method is stable, and can be called multiple times consecutively.
        """
        pass

    @abc.abstractmethod
    def assert_is_usable(self) -> None:
        pass

    @abc.abstractmethod
    def evaluate(self, evaluation_info: dict) -> float:
        # FIXME: come up with a reasonable way to implement this
        pass

    @abc.abstractmethod
    def evaluate_query(self, query: dict) -> dict:
        pass
