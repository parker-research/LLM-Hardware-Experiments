from llm_experiments.llms.llm_types import LlmPrompt, LlmResponse
from llm_experiments.llms.models.chatgpt_llm import (
    ChatGptLlmConfig,
    ChatGptLlm,
    _convert_chat_history_to_openai_api_dict,
)


def test_construction_1():
    llm = ChatGptLlm(
        config=ChatGptLlmConfig(
            configured_llm_name="ChatGPT 3.5 Turbo (Default Config)",
            model_name="gpt-3.5-turbo",
        ),
    )
    assert llm is not None
    assert llm.configured_llm_name == "ChatGPT 3.5 Turbo (Default Config)"
    assert llm.llm_provider_name == "ChatGptLlm"
    assert llm.config.model_name == "gpt-3.5-turbo"


def test__convert_chat_history_to_openai_api_dict():
    prompt1 = LlmPrompt("What color is the sky?")
    fake_response_1 = LlmResponse("It's blue.")
    prompt2 = LlmPrompt("What was the last question I asked you?")

    chat_history = [prompt1, fake_response_1, prompt2]

    messages_query = _convert_chat_history_to_openai_api_dict(chat_history)
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
