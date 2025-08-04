import os
import pytest
from unittest.mock import patch, MagicMock
from aisuite.providers.openai_provider import OpenaiProvider
import respx
import httpx

OPENROUTER_API_URL="https://openrouter.ai/api/v1"
MOCK_ANSWER = "mocked-text-response-from-openrouter-model"


no_of_calls=0
def mock_fun(request):
    global no_of_calls
    if OPENROUTER_API_URL in str(request.url):
        response = {
            "choices": [
                { "message": {
                    "content": MOCK_ANSWER,
                    }
                }
            ]
        }
        no_of_calls += 1
        
        return httpx.Response(200, json=response)

    return ValueError( f"Unexpected request: {request.url}")


@pytest.fixture(autouse=True)
def set_api_url_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("OPENAI_API_URL", OPENROUTER_API_URL)
    monkeypatch.setenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "boo"))


def test_completion():
    """Test that completions request successfully."""

    user_greeting = "Howdy!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "mistralai/mistral-small-3.1-24b-instruct:free"
    chosen_temperature = 0.77
    response_text_content = "mocked-text-response-from-openrouter-model"

    openrouter = OpenaiProvider(base_url=OPENROUTER_API_URL)
    #mock_response = {"message": {"content": response_text_content}}


    with respx.mock:
        mock_get = respx.get().mock(side_effect=mock_fun)
        mock_post = respx.post().mock(side_effect=mock_fun)

        response = openrouter.chat_completions_create(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        assert no_of_calls == 1
        '''
        mock_post.assert_called_once_with(
            f"{OPENROUTER_API_URL}/chat/completion",
            json={
                "model": selected_model,
                "messages": message_history,
                "stream": False,
                "temperature": chosen_temperature,
            },
            timeout=30,
        )
        '''

        assert response.choices[0].message.content == response_text_content
