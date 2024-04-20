from llm_experiments.feedback_eval_tools.feedback_eval_tool_base import (
    FeedbackEvalToolBase,
)


class MockFeedbackEvalTool(FeedbackEvalToolBase):
    def __init__(self, configured_tool_name: str, config: dict = {}):
        super().__init__(configured_tool_name, config)

    @classmethod
    def validate_configuration(cls, config: dict):
        assert isinstance(config, dict)

        if "mock_option" in config:
            assert isinstance(config["mock_option"], str)

    def check_is_connectable(self):
        return True

    def evaluate(self, evaluation_info: dict) -> float:
        # FIXME: doesn't really make sense
        return 0.5

    def evaluate_query(self, query: dict) -> dict:
        # FIXME: doesn't really make sense
        return {"response": "mock response"}
