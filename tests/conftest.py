"""Shared fixtures: isolated credentials, a fake OpenRouter catalog, mock HTTP."""

import json

import pytest

from model_gate import get_all_models, providers, registry
from model_gate._http import httpx

FAKE_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "XAI_API_KEY",
    "OPENROUTER_API_KEY",
)

CATALOG = [
    {
        "id": "openai/gpt-5.4-mini",
        "name": "OpenAI: GPT-5.4 Mini",
        "context_length": 400000,
        "architecture": {
            "input_modalities": ["file", "image", "text"],
            "output_modalities": ["text"],
        },
        "pricing": {
            "prompt": "0.00000075",
            "completion": "0.0000045",
            "input_cache_read": "0.000000075",
        },
        "top_provider": {
            "context_length": 400000,
            "max_completion_tokens": 128000,
        },
        "supported_parameters": [
            "max_tokens",
            "reasoning",
            "reasoning_effort",
            "response_format",
            "structured_outputs",
            "tool_choice",
            "tools",
        ],
        "reasoning": {"mandatory": False, "default_enabled": True},
    },
    {
        "id": "anthropic/claude-sonnet-4.6",
        "name": "Anthropic: Claude Sonnet 4.6",
        "context_length": 1000000,
        "architecture": {
            "input_modalities": ["text", "image", "file"],
            "output_modalities": ["text"],
        },
        "pricing": {
            "prompt": "0.000003",
            "completion": "0.000015",
            "input_cache_read": "0.0000003",
            "input_cache_write": "0.00000375",
        },
        "top_provider": {
            "context_length": 1000000,
            "max_completion_tokens": 128000,
        },
        "supported_parameters": [
            "max_tokens",
            "reasoning",
            "structured_outputs",
            "temperature",
            "tool_choice",
            "tools",
            "top_k",
        ],
    },
    {
        "id": "anthropic/claude-sonnet-4.6:batch",
        "name": "Anthropic: Claude Sonnet 4.6 (batch)",
        "context_length": 1000000,
        "architecture": {"output_modalities": ["text"]},
        "pricing": {"prompt": "0.0000015", "completion": "0.0000075"},
        "top_provider": {"max_completion_tokens": 128000},
        "supported_parameters": [],
    },
    {
        "id": "google/gemini-2.5-pro",
        "name": "Google: Gemini 2.5 Pro",
        "context_length": 1048576,
        "architecture": {
            "input_modalities": ["text", "image", "audio"],
            "output_modalities": ["text"],
        },
        "pricing": {"prompt": "0.00000125", "completion": "0.00001"},
        "top_provider": {"max_completion_tokens": 65536},
        "supported_parameters": ["reasoning", "tools"],
    },
    {
        "id": "deepseek/deepseek-chat",
        "name": "DeepSeek: DeepSeek V3",
        "context_length": 163840,
        "architecture": {
            "input_modalities": ["text"],
            "output_modalities": ["text"],
        },
        "pricing": {"prompt": "0.0000002", "completion": "0.000001"},
        "top_provider": {
            "context_length": 128000,
            "max_completion_tokens": 16000,
        },
        "supported_parameters": [
            "tools",
            "tool_choice",
            "temperature",
        ],
        "reasoning": None,
    },
]


def _all_key_vars():
    """Every API-key and base-URL variable ModelGate reads."""
    names = set()
    for spec in providers.PROVIDERS.values():
        names.update(spec.key_env)
        names.update(spec.base_env)
    for source in get_all_models.SOURCES.values():
        names.update(source.key_env)
        names.update(source.base_env)
    return names


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    """Clear real credentials, set fake ones, and reset every cache."""
    for name in _all_key_vars():
        monkeypatch.delenv(name, raising=False)
    for name in FAKE_KEYS:
        monkeypatch.setenv(name, f"test-{name.lower()}")
    registry.clear_model_cache()
    registry._registered.clear()
    get_all_models.clear_cache()
    yield
    registry.clear_model_cache()
    registry._registered.clear()
    get_all_models.clear_cache()


@pytest.fixture(autouse=True)
def catalog(monkeypatch):
    """Serve the fake OpenRouter catalog instead of the network; count fetches."""
    calls = {"count": 0}

    def fake_fetch():
        calls["count"] += 1
        return CATALOG

    monkeypatch.setattr(registry, "_fetch", fake_fetch)
    return calls


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Make retry delays zero."""
    import model_gate.main as main

    monkeypatch.setattr(main, "_backoff", lambda attempt: 0)
    monkeypatch.setattr(
        main, "_retry_delay", lambda response, attempt: 0
    )


class Recorder:
    """Mock transport handler that records requests and replays responses.

    Responses are returned in order; the last one repeats once the rest are used.
    """

    def __init__(self, *responses):
        """Queue responses.

        Args:
            *responses: httpx Responses, or callables taking the request.
        """
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        """Record the request and return the next response.

        Args:
            request: The outgoing request.

        Returns:
            The queued response.
        """
        self.requests.append(request)
        if len(self.responses) > 1:
            response = self.responses.pop(0)
        else:
            response = self.responses[0]
        return response(request) if callable(response) else response

    @property
    def last_json(self):
        """The decoded body of the last request."""
        return json.loads(self.requests[-1].content)


def sse(events):
    """Encode (event, data) pairs as a server-sent-event body.

    Args:
        events: Iterable of event name and JSON-serializable data.

    Returns:
        bytes: The event stream.
    """
    return "".join(
        f"event: {event}\ndata: {json.dumps(data)}\n\n"
        for event, data in events
    ).encode()


@pytest.fixture
def mock_http():
    """Build sync or async httpx clients backed by a Recorder."""

    def build(*responses, is_async=False):
        recorder = Recorder(*responses)
        transport = httpx.MockTransport(recorder)
        cls = httpx.AsyncClient if is_async else httpx.Client
        return cls(transport=transport), recorder

    return build


@pytest.fixture
def mock_openai(mock_http):
    """Build an OpenAI SDK client whose traffic goes to a Recorder."""

    def build(*responses, is_async=False):
        import openai

        http_client, recorder = mock_http(
            *responses, is_async=is_async
        )
        cls = openai.AsyncOpenAI if is_async else openai.OpenAI
        client = cls(
            api_key="test",
            base_url="https://mock.local/v1",
            http_client=http_client,
            max_retries=0,
        )
        return client, recorder

    return build


def chat_completion(content="OK", **message):
    """An OpenAI chat.completion response body.

    Args:
        content: Reply text.
        **message: Extra message fields.

    Returns:
        dict: The response JSON.
    """
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.4-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                    **message,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "total_tokens": 6,
        },
    }
