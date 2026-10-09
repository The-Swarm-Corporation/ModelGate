"""A tour of RouteHub.

Each example runs only when the API key it needs is set; the rest are skipped.
Run with: uv run python example.py
"""

import asyncio
import json
import os

from pydantic import BaseModel

import routehub
from routehub.get_all_models import get_all_models, get_model

OPENAI_MODEL = os.environ.get("EXAMPLE_OPENAI_MODEL", "gpt-5.4-mini")
CLAUDE_MODEL = os.environ.get(
    "EXAMPLE_CLAUDE_MODEL", "claude-sonnet-4-6"
)
GEMINI_MODEL = os.environ.get(
    "EXAMPLE_GEMINI_MODEL", "gemini/gemini-3-flash-preview"
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def ask(text: str) -> list:
    """One user message.

    Args:
        text (str): The message.

    Returns:
        list: A messages list.
    """
    return [{"role": "user", "content": text}]


def mock_response() -> None:
    """A canned reply, with no network call and no API key."""
    response = routehub.completion(
        model=OPENAI_MODEL,
        messages=ask("Is the deployment approved?"),
        mock_response="Approved.",
    )
    print(response.choices[0].message.content)


def basic_completion() -> None:
    """One request, with the reply and token usage."""
    response = routehub.completion(
        model=OPENAI_MODEL,
        messages=ask("Name three prime numbers, comma separated."),
        max_tokens=100,
    )
    print(response.choices[0].message.content)
    print(
        "usage:",
        response.usage.prompt_tokens,
        "in,",
        response.usage.completion_tokens,
        "out",
    )


def streaming() -> None:
    """Print tokens as they arrive, then the usage chunk."""
    stream = routehub.completion(
        model=OPENAI_MODEL,
        messages=ask("Count from 1 to 10."),
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.usage:
            print(f"\nusage: {chunk.usage.total_tokens} tokens")
        elif chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)


def concurrent_async() -> None:
    """Send three requests at once with acompletion."""

    async def run() -> list:
        questions = [
            "Capital of Japan?",
            "Capital of Kenya?",
            "Capital of Peru?",
        ]
        responses = await asyncio.gather(
            *(
                routehub.acompletion(
                    model=OPENAI_MODEL,
                    messages=ask(f"{q} One word."),
                    max_tokens=20,
                )
                for q in questions
            )
        )
        return [r.choices[0].message.content for r in responses]

    print(asyncio.run(run()))


def tool_calling(model: str) -> None:
    """Let the model call a tool, return the result, and get the final answer.

    Args:
        model (str): The model to use.
    """
    messages = ask("What's the weather in Paris?")
    response = routehub.completion(
        model=model,
        messages=messages,
        tools=[WEATHER_TOOL],
        max_tokens=1024,
    )
    message = response.choices[0].message
    if not message.tool_calls:
        print("No tool call:", message.content)
        return
    call = message.tool_calls[0]
    print(
        "tool call:",
        call.function.name,
        json.loads(call.function.arguments),
    )
    messages += [
        {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [call.model_dump()],
        },
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": "Sunny, 21°C",
        },
    ]
    final = routehub.completion(
        model=model,
        messages=messages,
        tools=[WEATHER_TOOL],
        max_tokens=1024,
    )
    print(final.choices[0].message.content)


class Capital(BaseModel):
    country: str
    capital: str
    population_millions: float


def structured_output() -> None:
    """Get a reply that validates against a pydantic model."""
    response = routehub.completion(
        model=OPENAI_MODEL,
        messages=ask(
            "Give the capital of Canada and its population."
        ),
        response_format=Capital,
    )
    print(
        Capital.model_validate_json(
            response.choices[0].message.content
        )
    )


def claude_thinking_and_caching() -> None:
    """Cache a long system prompt and ask Claude to think first."""
    handbook = "Company travel policy. " + " ".join(
        f"Rule {i}: book economy for flights under {i + 4} hours."
        for i in range(300)
    )
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": handbook,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": "Can I book business class for a 6 hour flight?",
        },
    ]
    for attempt in (1, 2):
        response = routehub.completion(
            model=CLAUDE_MODEL,
            messages=messages,
            reasoning_effort="low",
            max_tokens=4000,
        )
        usage = response.usage
        print(
            f"call {attempt}: cache read={usage.prompt_tokens_details.cached_tokens} "
            f"cache write={getattr(usage, 'cache_creation_input_tokens', 0)}"
        )
    message = response.choices[0].message
    print(
        "thinking:",
        (getattr(message, "reasoning_content", "") or "")[:120],
    )
    print("answer:", message.content)


def embeddings() -> None:
    """Embed two texts."""
    response = routehub.embedding(
        model="text-embedding-3-small",
        input=["quarterly revenue report", "annual earnings summary"],
    )
    print(
        len(response.data),
        "vectors of",
        len(response.data[0].embedding),
        "dimensions",
    )


def model_catalog() -> None:
    """Look up limits, prices and capabilities from the live catalog."""
    for name in (OPENAI_MODEL, CLAUDE_MODEL):
        info = routehub.get_model_info(name)
        print(
            f"{name}: {info['max_input_tokens']:,} token context, "
            f"{info['max_output_tokens']:,} max output, "
            f"${info['input_cost_per_token'] * 1e6:.2f} per million input tokens, "
            f"reasoning={info['supports_reasoning']}, vision={info['supports_vision']}"
        )
    print(
        "claude-sonnet-4-6 known:",
        "claude-sonnet-4-6" in routehub.model_list,
    )


def list_all_models() -> None:
    """List models from every configured provider's own API."""
    models = get_all_models()
    by_provider = {}
    for entry in models:
        by_provider[entry["provider"]] = (
            by_provider.get(entry["provider"], 0) + 1
        )
    print(f"{len(models)} models:", by_provider)
    detail = get_model("openrouter", "anthropic/claude-sonnet-4.6")
    if detail:
        print(
            "hosts serving claude-sonnet-4.6 on OpenRouter:",
            len(detail["endpoints"] or []),
        )


def error_handling() -> None:
    """Catch a typed error from a bad key."""
    try:
        routehub.completion(
            model=OPENAI_MODEL,
            messages=ask("hi"),
            api_key="sk-invalid",
            num_retries=0,
        )
    except routehub.AuthenticationError as error:
        print(
            f"{type(error).__name__}: status={error.status_code} provider={error.llm_provider}"
        )


def token_counting() -> None:
    """Count tokens in a conversation before sending it."""
    messages = [
        {"role": "system", "content": "You are a concise analyst."},
        {
            "role": "user",
            "content": "Summarize the attached quarterly report.",
        },
    ]
    print(
        routehub.token_counter(model=OPENAI_MODEL, messages=messages),
        "tokens",
    )


EXAMPLES = [
    ("mock response", mock_response, None),
    ("token counting", token_counting, None),
    ("model catalog", model_catalog, None),
    ("list all models", list_all_models, None),
    ("basic completion", basic_completion, "OPENAI_API_KEY"),
    ("streaming", streaming, "OPENAI_API_KEY"),
    ("concurrent async", concurrent_async, "OPENAI_API_KEY"),
    (
        "tool calling (OpenAI)",
        lambda: tool_calling(OPENAI_MODEL),
        "OPENAI_API_KEY",
    ),
    ("structured output", structured_output, "OPENAI_API_KEY"),
    ("embeddings", embeddings, "OPENAI_API_KEY"),
    ("error handling", error_handling, None),
    (
        "tool calling (Gemini)",
        lambda: tool_calling(GEMINI_MODEL),
        "GEMINI_API_KEY",
    ),
    (
        "Claude thinking and prompt caching",
        claude_thinking_and_caching,
        "ANTHROPIC_API_KEY",
    ),
    (
        "tool calling (Claude)",
        lambda: tool_calling(CLAUDE_MODEL),
        "ANTHROPIC_API_KEY",
    ),
]


def main() -> None:
    """Run every example whose API key is set."""
    for title, run, key in EXAMPLES:
        print(f"\n== {title}")
        if key and not os.environ.get(key):
            print(f"skipped: set {key} to run it")
            continue
        try:
            run()
        except Exception as error:
            # One provider's outage or billing problem should not stop the tour.
            print(f"failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
