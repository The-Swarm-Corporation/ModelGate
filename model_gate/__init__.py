"""ModelGate: a fast, lightweight LLM gateway with a litellm-compatible API."""

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

_LAZY = {
    "completion": "model_gate.main",
    "acompletion": "model_gate.main",
    "embedding": "model_gate.main",
    "aembedding": "model_gate.main",
    "get_reasoning_efforts": "model_gate.main",
    "REASONING_EFFORTS": "model_gate.main",
    "get_llm_provider": "model_gate.providers",
    "PROVIDERS": "model_gate.providers",
    "get_model_info": "model_gate.registry",
    "get_max_tokens": "model_gate.registry",
    "register_model": "model_gate.registry",
    "model_cost": "model_gate.registry",
    "model_list": "model_gate.registry",
    "clear_model_cache": "model_gate.registry",
    "supports_vision": "model_gate.registry",
    "supports_reasoning": "model_gate.registry",
    "supports_function_calling": "model_gate.registry",
    "supports_parallel_function_calling": "model_gate.registry",
    "supports_response_schema": "model_gate.registry",
    "supports_prompt_caching": "model_gate.registry",
    "supports_tool_choice": "model_gate.registry",
    "supports_system_messages": "model_gate.registry",
    "supports_pdf_input": "model_gate.registry",
    "supports_audio_input": "model_gate.registry",
    "supports_web_search": "model_gate.registry",
    "supports_audio_output": "model_gate.registry",
    "supports_computer_use": "model_gate.registry",
    "supports_assistant_prefill": "model_gate.registry",
    "encode": "model_gate.tokenizer",
    "decode": "model_gate.tokenizer",
    "token_counter": "model_gate.tokenizer",
    "APIError": "model_gate.exceptions",
    "APIConnectionError": "model_gate.exceptions",
    "AuthenticationError": "model_gate.exceptions",
    "BadRequestError": "model_gate.exceptions",
    "ContentPolicyViolationError": "model_gate.exceptions",
    "ContextWindowExceededError": "model_gate.exceptions",
    "InternalServerError": "model_gate.exceptions",
    "ModelNotMappedError": "model_gate.registry",
    "NotFoundError": "model_gate.exceptions",
    "OpenAIError": "model_gate.exceptions",
    "PermissionDeniedError": "model_gate.exceptions",
    "RateLimitError": "model_gate.exceptions",
    "ServiceUnavailableError": "model_gate.exceptions",
    "Timeout": "model_gate.exceptions",
    "UnprocessableEntityError": "model_gate.exceptions",
    "UnsupportedParamsError": "model_gate.exceptions",
}

_UNCACHED = frozenset({"model_list", "model_cost"})

__all__ = sorted(_LAZY)

if TYPE_CHECKING:  # pragma: no cover
    from model_gate.exceptions import *  # noqa: F401,F403
    from model_gate.main import (  # noqa: F401
        REASONING_EFFORTS,
        acompletion,
        aembedding,
        completion,
        embedding,
        get_reasoning_efforts,
    )
    from model_gate.providers import (
        PROVIDERS,
        get_llm_provider,
    )  # noqa: F401
    from model_gate.registry import *  # noqa: F401,F403
    from model_gate.tokenizer import (
        decode,
        encode,
        token_counter,
    )  # noqa: F401


def __getattr__(name: str):
    """Load a public name from its submodule on first access.

    Args:
        name (str): The attribute being looked up.

    Returns:
        Any: The requested function, class or value.
    """
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(
            f"module 'model_gate' has no attribute {name!r}"
        )
    value = getattr(importlib.import_module(module), name)
    # model_list and model_cost refresh with the registry, so they are not pinned.
    if name not in _UNCACHED:
        globals()[name] = value
    return value


def __dir__() -> list:
    """List the module's public names, including lazy ones.

    Returns:
        list: Attribute names.
    """
    return sorted(set(globals()) | set(_LAZY))
