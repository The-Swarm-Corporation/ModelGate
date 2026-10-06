"""embedding and aembedding."""

import openai
import pytest

import model_gate as mg
from model_gate._http import httpx


def embedding_body(*vectors):
    return {
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }


def test_embedding_returns_openai_response(mock_openai):
    client, recorder = mock_openai(httpx.Response(200, json=embedding_body([0.1, 0.2], [0.3, 0.4])))
    response = mg.embedding(model="text-embedding-3-small", input=["a", "b"], dimensions=2, client=client)
    assert isinstance(response, openai.types.CreateEmbeddingResponse)
    assert response.data[1].embedding == [0.3, 0.4]
    body = recorder.last_json
    assert body["model"] == "text-embedding-3-small"
    assert body["input"] == ["a", "b"]
    assert body["dimensions"] == 2


def test_embedding_strips_provider_prefix(mock_openai):
    client, recorder = mock_openai(httpx.Response(200, json=embedding_body([1.0])))
    mg.embedding(model="openai/text-embedding-3-small", input="a", client=client)
    assert recorder.last_json["model"] == "text-embedding-3-small"
    assert recorder.last_json["input"] == "a"


def test_embedding_passes_options(mock_openai):
    client, recorder = mock_openai(httpx.Response(200, json=embedding_body([1.0])))
    mg.embedding(
        model="text-embedding-3-small",
        input="a",
        user="u",
        encoding_format="float",
        extra_body={"x": 1},
        extra_headers={"h": "v"},
        client=client,
    )
    body = recorder.last_json
    assert body["user"] == "u"
    assert body["encoding_format"] == "float"
    assert body["x"] == 1
    assert recorder.requests[-1].headers["h"] == "v"


def test_embedding_uses_the_cached_client(monkeypatch, mock_openai):
    import model_gate.main as main

    client, _ = mock_openai(httpx.Response(200, json=embedding_body([1.0])))
    seen = {}

    def fake_openai_client(provider, api_key, api_base, **kwargs):
        seen.update(provider=provider, api_key=api_key, **kwargs)
        return client

    monkeypatch.setattr(main, "openai_client", fake_openai_client)
    mg.embedding(model="text-embedding-3-small", input="a", num_retries=4, ssl_verify=False)
    assert seen["provider"] == "openai"
    assert seen["api_key"] == "test-openai_api_key"
    assert seen["max_retries"] == 4
    assert seen["ssl_verify"] is False


def test_embedding_maps_errors(mock_openai):
    client, _ = mock_openai(httpx.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(mg.AuthenticationError):
        mg.embedding(model="text-embedding-3-small", input="a", client=client)


def test_anthropic_has_no_embeddings():
    with pytest.raises(mg.BadRequestError, match="no embeddings"):
        mg.embedding(model="claude-sonnet-4-6", input="a")


def test_embedding_needs_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(mg.AuthenticationError, match="OPENAI_API_KEY"):
        mg.embedding(model="text-embedding-3-small", input="a")


async def test_aembedding(mock_openai):
    client, recorder = mock_openai(httpx.Response(200, json=embedding_body([0.5])), is_async=True)
    response = await mg.aembedding(model="text-embedding-3-small", input="a", client=client)
    assert response.data[0].embedding == [0.5]
    assert recorder.last_json["model"] == "text-embedding-3-small"


async def test_aembedding_maps_errors(mock_openai):
    client, _ = mock_openai(httpx.Response(429, json={"error": {"message": "slow"}}), is_async=True)
    with pytest.raises(mg.RateLimitError):
        await mg.aembedding(model="text-embedding-3-small", input="a", client=client)


def test_own_client_needs_no_env_key(monkeypatch, mock_openai):
    monkeypatch.delenv("OPENAI_API_KEY")
    client, _ = mock_openai(httpx.Response(200, json=embedding_body([1.0])))
    assert mg.embedding(model="text-embedding-3-small", input="a", client=client).data[0].embedding == [1.0]
