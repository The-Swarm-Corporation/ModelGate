"""The helpers litellm exposes under litellm.utils."""

from routehub.providers import get_llm_provider
from routehub.registry import (
    get_max_tokens,
    get_model_info,
    register_model,
    supports_assistant_prefill,
    supports_audio_input,
    supports_audio_output,
    supports_computer_use,
    supports_function_calling,
    supports_parallel_function_calling,
    supports_pdf_input,
    supports_prompt_caching,
    supports_reasoning,
    supports_response_schema,
    supports_system_messages,
    supports_tool_choice,
    supports_vision,
    supports_web_search,
)
from routehub.tokenizer import decode, encode, token_counter

__all__ = [
    "decode",
    "encode",
    "get_llm_provider",
    "get_max_tokens",
    "get_model_info",
    "register_model",
    "supports_assistant_prefill",
    "supports_audio_input",
    "supports_audio_output",
    "supports_computer_use",
    "supports_function_calling",
    "supports_parallel_function_calling",
    "supports_pdf_input",
    "supports_prompt_caching",
    "supports_reasoning",
    "supports_response_schema",
    "supports_system_messages",
    "supports_tool_choice",
    "supports_vision",
    "supports_web_search",
    "token_counter",
]
