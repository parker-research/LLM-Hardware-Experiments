from llm_experiments.llms.llm_types import LlmPrompt, LlmResponse
from llm_experiments.llms.models.mock_llm import MockLlm, MockLlmConfig


def test_construction_1():
    llm = MockLlm(
        configured_llm_name="MockLlm_Test_1",
        config=MockLlmConfig(does_respond_to_test_queries=True),
    )
    assert llm is not None
    assert llm.configured_llm_name == "MockLlm_Test_1"
    assert llm.base_llm_name == "MockLlm"
    assert llm.config.does_respond_to_test_queries


def test_construction_2():
    llm = MockLlm(
        configured_llm_name="MockLlm_Test_1",
        config=MockLlmConfig(does_respond_to_test_queries=False),
    )
    assert llm is not None
    assert llm.configured_llm_name == "MockLlm_Test_1"
    assert llm.base_llm_name == "MockLlm"
    assert not llm.config.does_respond_to_test_queries


def test_check_is_connectable():
    llm = MockLlm(
        configured_llm_name="MockLlm_Test_1",
        config=MockLlmConfig(does_respond_to_test_queries=True),
    )
    assert llm.check_is_connectable()


def test_query_llm():
    llm = MockLlm(
        configured_llm_name="MockLlm_Test_1",
        config=MockLlmConfig(does_respond_to_test_queries=True),
    )
    prompt = LlmPrompt("Write me 10 sentences ending with the word 'apple'.")
    response = llm.query_llm(prompt)
    assert response is not None
    assert isinstance(response, LlmResponse)
    assert response.response_text.startswith(
        "This is sentence 1 ending with apple. This is sentence 2 ending with apple."
    )
    assert response.response_text.endswith(
        "This is sentence 9 ending with apple. This is sentence 10 ending with apple."
    )
