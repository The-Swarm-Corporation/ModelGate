"""Model metadata from the OpenRouter catalog."""

import pytest

import model_gate as mg
from model_gate import registry
from model_gate._http import httpx

from conftest import CATALOG

# Captured at collection, before the autouse fixture swaps in the fake catalog.
REAL_FETCH = registry._fetch


class TestLookups:
    def test_maps_openrouter_fields(self):
        info = mg.get_model_info("gpt-5.4-mini")
        assert info["key"] == "openai/gpt-5.4-mini"
        assert info["litellm_provider"] == "openai"
        assert info["mode"] == "chat"
        assert info["max_input_tokens"] == 400000
        assert info["max_output_tokens"] == 128000
        assert info["input_cost_per_token"] == pytest.approx(0.00000075)
        assert info["cache_read_input_token_cost"] == pytest.approx(0.000000075)
        assert info["supports_vision"] is True
        assert info["supports_pdf_input"] is True
        assert info["supports_reasoning"] is True
        assert info["supports_function_calling"] is True
        assert info["supports_response_schema"] is True
        assert info["supports_prompt_caching"] is True

    @pytest.mark.parametrize(
        "name",
        [
            "claude-sonnet-4-6",
            "anthropic/claude-sonnet-4-6",
            "claude-sonnet-4-6-20260101",
            "anthropic/claude-sonnet-4.6",
            "openrouter/anthropic/claude-sonnet-4.6",
            "claude-sonnet-4.6",
        ],
    )
    def test_native_and_openrouter_names_resolve(self, name):
        assert mg.get_model_info(name)["key"] == "anthropic/claude-sonnet-4.6"

    def test_provider_prefix_maps_to_vendor(self):
        info = mg.get_model_info("gemini/gemini-2.5-pro")
        assert info["key"] == "google/gemini-2.5-pro"
        assert info["litellm_provider"] == "gemini"
        assert info["supports_audio_input"] is True

    def test_variants_need_their_full_id(self):
        assert mg.get_model_info("openrouter/anthropic/claude-sonnet-4.6:batch")["key"] == "anthropic/claude-sonnet-4.6:batch"
        assert mg.get_model_info("claude-sonnet-4-6")["key"] == "anthropic/claude-sonnet-4.6"

    def test_reasoning_false_when_not_reported(self):
        assert mg.supports_reasoning("deepseek/deepseek-chat") is False
        assert mg.supports_vision("deepseek/deepseek-chat") is False
        assert mg.supports_function_calling("deepseek/deepseek-chat") is True

    def test_unknown_model_raises_and_supports_is_false(self):
        with pytest.raises(mg.ModelNotMappedError):
            mg.get_model_info("no-such-model")
        with pytest.raises(mg.ModelNotMappedError):
            mg.get_max_tokens("no-such-model")
        assert mg.supports_vision("no-such-model") is False

    def test_get_max_tokens(self):
        assert mg.get_max_tokens("claude-sonnet-4-6") == 128000

    def test_utils_module_mirrors_litellm_utils(self):
        from model_gate.utils import get_model_info, supports_parallel_function_calling

        assert get_model_info("gpt-5.4-mini")["max_input_tokens"] == 400000
        assert supports_parallel_function_calling("gpt-5.4-mini") is True


class TestModelList:
    def test_contains_resolves_aliases(self):
        models = mg.model_list
        assert "openai/gpt-5.4-mini" in list(models)
        assert "claude-sonnet-4-6" in models
        assert "gpt-5.4-mini" in models
        assert "made-up-model" not in models

    def test_model_cost_is_keyed_by_openrouter_id(self):
        cost = mg.model_cost
        assert cost["anthropic/claude-sonnet-4.6"]["max_output_tokens"] == 128000
        assert set(cost) == {entry["id"] for entry in CATALOG}


class TestCaching:
    def test_catalog_is_fetched_once_per_ttl(self, catalog):
        mg.get_model_info("gpt-5.4-mini")
        mg.supports_vision("claude-sonnet-4-6")
        list(mg.model_list)
        assert catalog["count"] == 1

    def test_clear_model_cache_refetches(self, catalog):
        mg.get_model_info("gpt-5.4-mini")
        mg.clear_model_cache()
        mg.get_model_info("gpt-5.4-mini")
        assert catalog["count"] == 2

    def test_expired_cache_refetches(self, catalog, monkeypatch):
        mg.get_model_info("gpt-5.4-mini")
        monkeypatch.setattr(registry, "_expires_at", 0.0)
        mg.get_model_info("gpt-5.4-mini")
        assert catalog["count"] == 2

    def test_failed_refresh_keeps_stale_data(self, monkeypatch):
        mg.get_model_info("gpt-5.4-mini")
        monkeypatch.setattr(registry, "_fetch", lambda: [])
        monkeypatch.setattr(registry, "_expires_at", 0.0)
        assert mg.get_model_info("gpt-5.4-mini")["max_input_tokens"] == 400000

    def test_failed_first_fetch_means_unknown(self, monkeypatch):
        monkeypatch.setattr(registry, "_fetch", lambda: [])
        assert mg.supports_vision("gpt-5.4-mini") is False

    def test_fetch_handles_network_errors(self, monkeypatch):
        def offline(*args, **kwargs):
            raise httpx.ConnectError("offline")

        monkeypatch.setattr(httpx, "get", offline)
        assert REAL_FETCH() == []

    def test_fetch_parses_the_openrouter_response(self, monkeypatch):
        def serve(url, **kwargs):
            assert url == registry.OPENROUTER_MODELS_URL
            return httpx.Response(200, json={"data": CATALOG}, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "get", serve)
        assert [e["id"] for e in REAL_FETCH()] == [e["id"] for e in CATALOG]


class TestRegisterModel:
    def test_registered_models_override_and_never_expire(self):
        mg.register_model({"my-model": {"max_input_tokens": 8192, "max_output_tokens": 1024, "supports_function_calling": True}})
        info = mg.get_model_info("my-model")
        assert info["key"] == "my-model"
        assert info["max_output_tokens"] == 1024
        assert info["supports_vision"] is None
        assert mg.supports_function_calling("my-model") is True
        assert "my-model" in mg.model_list
        assert mg.model_cost["my-model"]["max_input_tokens"] == 8192

    def test_registration_merges_fields(self):
        mg.register_model({"m": {"max_input_tokens": 1}})
        mg.register_model({"m": {"max_output_tokens": 2}})
        info = mg.get_model_info("m")
        assert info["max_input_tokens"] == 1 and info["max_output_tokens"] == 2

    def test_registration_beats_the_catalog(self):
        mg.register_model({"gpt-5.4-mini": {"max_input_tokens": 1234}})
        assert mg.get_model_info("gpt-5.4-mini")["max_input_tokens"] == 1234

    def test_provider_prefixed_lookup_finds_registration(self):
        mg.register_model({"local-llama": {"max_input_tokens": 4096}})
        assert mg.get_model_info("ollama/local-llama")["max_input_tokens"] == 4096

    def test_register_from_json_file(self, tmp_path):
        path = tmp_path / "models.json"
        path.write_text('{"file-model": {"max_input_tokens": 77}}')
        mg.register_model(str(path))
        assert mg.get_model_info("file-model")["max_input_tokens"] == 77


def test_normalize_names():
    assert registry._normalize("claude-opus-4-7-20251001") == "claude-opus-4.7"
    assert registry._normalize("claude-3-5-haiku-latest") == "claude-3.5-haiku"
    assert registry._normalize("gpt-4o-2024-08-06") == "gpt-4o"
    assert registry._normalize("gemini-2.5-pro") == "gemini-2.5-pro"
