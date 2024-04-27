import pytest

from llm_experiments.llms.llm_types import LlmPrompt, LlmResponse
from llm_experiments.llms.llm_base import LlmBase
from llm_experiments.llms.models.ollama_llm import (
    OllamaLlmConfig,
    OllamaLlm,
)
from llm_experiments.llms.models.chatgpt_llm import (
    ChatGptLlmConfig,
    ChatGptLlm,
)

llms = [
    # === These TinyLlama tests don't really prove anything because the model is so bad. === #
    # OllamaLlm(
    #     configured_llm_name="TinyLlama Default Config",
    #     config=OllamaLlmConfig(model_name="tinyllama:1.1b"),
    # ),
    # OllamaLlm(
    #     configured_llm_name="TinyLlama No Randomness",
    #     config=OllamaLlmConfig(
    #         is_stable=True,
    #         model_name="tinyllama:1.1b",
    #         option_seed=9865,  # any fixed number is good
    #         option_temperature=0,
    #     ),
    # ),
    OllamaLlm(
        configured_llm_name="Llama2-7B No Randomness",
        config=OllamaLlmConfig(
            is_stable=True,
            model_name="llama2:7b",
            option_seed=101,  # any fixed number is good
            option_temperature=0,
        ),
    ),
    ChatGptLlm(
        configured_llm_name="ChatGPT 3.5 Turbo (Default Config)",
        config=ChatGptLlmConfig(model_name="gpt-3.5-turbo"),
    ),
    ChatGptLlm(
        configured_llm_name="ChatGPT 3.5 Turbo (No Randomness)",
        config=ChatGptLlmConfig(
            # is_stable=True,
            model_name="gpt-3.5-turbo",
            option_seed=101,  # any fixed number is good
            option_temperature=0,
        ),
    ),
]
llms_ids = [llm.configured_llm_name for llm in llms]

stable_llms = [llm for llm in llms if llm.config.is_stable]
stable_llms_ids = [llm.configured_llm_name for llm in stable_llms]


@pytest.mark.parametrize("llm", llms, ids=llms_ids)
def test_construction_1(llm):
    assert llm is not None
    assert isinstance(llm, LlmBase)
    assert llm.base_llm_name.endswith("Llm")


@pytest.mark.parametrize("llm", llms, ids=llms_ids)
def test_init_model(llm):
    x = llm.init_model()
    assert x is None


@pytest.mark.parametrize("llm", llms, ids=llms_ids)
def test_query_llm_basic(llm):
    llm.init_model()

    prompt = LlmPrompt("What color is the sky?")
    response = llm.query_llm_basic(prompt)
    assert isinstance(response, LlmResponse)
    assert len(response.response_text) > 3
    assert "blue" in response.response_text.lower()


@pytest.mark.parametrize("llm", stable_llms, ids=stable_llms_ids)
def test_query_llm_basic_response_is_stable(llm):
    llm.init_model()

    prompt = LlmPrompt("What color is the sky?")
    response1 = llm.query_llm_basic(prompt)
    response2 = llm.query_llm_basic(prompt)
    assert response1.response_text == response2.response_text

    response3 = llm.query_llm_basic(prompt)
    assert response1.response_text == response3.response_text

    response4 = llm.query_llm_basic(prompt)
    assert response1.response_text == response4.response_text


@pytest.mark.parametrize("llm", llms, ids=llms_ids)
def test_query_llm_chat(llm):

    llm.init_model()

    prompt1 = LlmPrompt("What color is the sky?")
    chat_history = []
    response1 = llm.query_llm_chat(prompt1, chat_history)
    assert isinstance(response1, LlmResponse)
    assert len(response1.response_text) > 3
    assert "blue" in response1.response_text.lower()

    chat_history.extend([prompt1, response1])

    prompt2 = LlmPrompt("What was the last question I asked you?")
    response2 = llm.query_llm_chat(prompt2, chat_history)
    assert isinstance(response2, LlmResponse)
    assert len(response2.response_text) > 3

    assert "what" in response2.response_text.lower()
    assert "sky" in response2.response_text.lower()
    assert any(
        [
            "color" in response2.response_text.lower(),
            "colour" in response2.response_text.lower(),
        ]
    )
    assert response1.response_text != response2.response_text


@pytest.mark.parametrize("llm", llms, ids=llms_ids)
def test_query_llm_chat__isolation(llm):
    """Test that chat sequences are isolated (i.e., they don't
    have a "memory" effect on the model).

    Chat chain "A" should not affect chat chain "B".
    """
    llm.init_model()

    # Chat chain "A", query 1
    prompt_A_1 = LlmPrompt("What color is the sky?")
    chat_history_A = []
    response_A_1 = llm.query_llm_chat(prompt_A_1, chat_history_A)
    assert isinstance(response_A_1, LlmResponse)
    assert "blue" in response_A_1.response_text.lower()
    chat_history_A.extend([prompt_A_1, response_A_1])

    # Chat chain "B", query 1
    prompt_B_1 = LlmPrompt("What is the most popular programming language for data science?")
    chat_history_B = []
    response_B_1 = llm.query_llm_chat(prompt_B_1, chat_history_B)
    assert isinstance(response_B_1, LlmResponse)
    assert "python" in response_B_1.response_text.lower()
    assert "language" in response_B_1.response_text.lower()
    chat_history_B.extend([prompt_B_1, response_B_1])

    # Chat chain "A", query 2
    prompt_A_2 = LlmPrompt("What was the last question I asked you, and what was your response?")
    response_A_2 = llm.query_llm_chat(prompt_A_2, chat_history_A)
    assert isinstance(response_A_2, LlmResponse)
    assert "what" in response_A_2.response_text.lower()
    assert "sky" in response_A_2.response_text.lower()
    assert any(
        [
            "color" in response_A_2.response_text.lower(),
            "colour" in response_A_2.response_text.lower(),
        ]
    )
    assert "python" not in response_A_2.response_text.lower()
    assert "language" not in response_A_2.response_text.lower()

    # Chat chain "B", query 2
    prompt_B_2 = LlmPrompt("What was the last question I asked you, and what was your response?")
    response_B_2 = llm.query_llm_chat(prompt_B_2, chat_history_B)
    assert isinstance(response_B_2, LlmResponse)

    # Chat chain "B", query 2 - response is about Python
    assert "what" in response_B_2.response_text.lower()
    assert "data" in response_B_2.response_text.lower()
    assert "science" in response_B_2.response_text.lower()
    assert "python" in response_B_2.response_text.lower()
    assert "language" in response_B_2.response_text.lower()

    # Chat chain "B", query 2 - response is not about the sky
    assert "blue" not in response_B_2.response_text.lower()
    assert "color" not in response_B_2.response_text.lower()
    assert "colour" not in response_B_2.response_text.lower()
    assert "sky" not in response_B_2.response_text.lower()
