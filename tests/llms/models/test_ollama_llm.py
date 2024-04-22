from llm_experiments.llms.llm_types import LlmPrompt, LlmResponse
from llm_experiments.llms.models.ollama_llm import (
    OllamaLlmConfig,
    OllamaLlm,
    solid_configs,
    _convert_chat_history_to_ollama_api_dict,
)


def test_construction_1():
    llm = OllamaLlm(
        configured_llm_name="TinyLlama Default Config",
        config=OllamaLlmConfig(model_name="tinyllama:1.1b"),
    )
    assert llm is not None
    assert llm.configured_llm_name == "TinyLlama Default Config"
    assert llm.base_llm_name == "OllamaLlm"
    assert llm.config.model_name == "tinyllama:1.1b"


def test_ollama_init_model():
    config = solid_configs["tinyllama_no_randomness"]
    llm = OllamaLlm(configured_llm_name="tinyllama_no_randomness", config=config)
    x = llm.init_model()
    assert x is None


def test_query_llm_basic():
    config = solid_configs["tinyllama_no_randomness"]
    llm = OllamaLlm(configured_llm_name="tinyllama_no_randomness", config=config)
    llm.init_model()

    prompt = LlmPrompt("What color is the sky?")
    response = llm.query_llm_basic(prompt)
    assert isinstance(response, LlmResponse)
    assert len(response.response_text) > 3
    assert "blue" in response.response_text.lower()


def test_query_llm_basic_response_is_stable():
    config = solid_configs["tinyllama_no_randomness"]
    llm = OllamaLlm(configured_llm_name="tinyllama_no_randomness", config=config)
    llm.init_model()

    prompt = LlmPrompt("What color is the sky?")
    response1 = llm.query_llm_basic(prompt)
    response2 = llm.query_llm_basic(prompt)
    assert response1.response_text == response2.response_text

    response3 = llm.query_llm_basic(prompt)
    assert response1.response_text == response3.response_text

    response4 = llm.query_llm_basic(prompt)
    assert response1.response_text == response4.response_text


def test_query_llm_chat():
    # NOTE: this test doesn't work with tinyllama_no_randomness
    config = solid_configs["llama2_7b_no_randomness"]
    llm = OllamaLlm(configured_llm_name="llama2_7b_no_randomness", config=config)
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


def test_query_llm_chat__isolation():
    """Test that chat sequences are isolated (i.e., they don't
    have a "memory" effect on the model).

    Chat chain "A" should not affect chat chain "B".
    """
    # NOTE: this test doesn't work with tinyllama_no_randomness
    config = solid_configs["llama2_7b_no_randomness"]
    llm = OllamaLlm(configured_llm_name="llama2_7b_no_randomness", config=config)
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


def test__convert_chat_history_to_ollama_api_dict():
    prompt1 = LlmPrompt("What color is the sky?")
    fake_response_1 = LlmResponse("It's blue.")
    prompt2 = LlmPrompt("What was the last question I asked you?")

    chat_history = [prompt1, fake_response_1, prompt2]

    messages_query = _convert_chat_history_to_ollama_api_dict(chat_history)
    assert isinstance(messages_query, list)
    assert len(messages_query) == 3

    assert messages_query == [
        {
            "content": "What color is the sky?",
            "role": "user",
        },
        {
            "content": "It's blue.",
            "role": "assistant",
        },
        {
            "content": "What was the last question I asked you?",
            "role": "user",
        },
    ]
