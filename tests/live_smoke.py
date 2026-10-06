"""Live smoke test against real providers; needs API keys in the environment."""

import asyncio
import json
import os
import sys
import time

from pydantic import BaseModel

import routehub as rh

RESULTS = []


def check(name, fn):
    """Run one live check and record pass or fail.

    Args:
        name (str): Check label.
        fn (Callable[[], str]): Runs the check and returns a short detail.
    """
    started = time.perf_counter()
    try:
        detail = fn()
        RESULTS.append(
            (name, True, detail, time.perf_counter() - started)
        )
    except Exception as error:
        RESULTS.append(
            (
                name,
                False,
                f"{type(error).__name__}: {str(error)[:160]}",
                time.perf_counter() - started,
            )
        )


def user(text):
    """One user message.

    Args:
        text (str): The message text.

    Returns:
        list: A messages list.
    """
    return [{"role": "user", "content": text}]


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


class Capital(BaseModel):
    country: str
    capital: str


def basic(model, max_tokens=64, **kw):
    def run():
        r = rh.completion(
            model=model,
            messages=user("Reply with the single word OK."),
            max_tokens=max_tokens,
            drop_params=True,
            **kw,
        )
        u = r.usage
        return f"{r.choices[0].message.content!r} usage={u.prompt_tokens}/{u.completion_tokens}"

    return run


def stream(model, max_tokens=64):
    def run():
        s = rh.completion(
            model=model,
            messages=user("Count from 1 to 5, comma separated."),
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            drop_params=True,
        )
        text, usage = "", None
        for chunk in s:
            if chunk.usage:
                usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
        assert text, "no streamed text"
        return (
            f"{text.strip()!r} usage={'yes' if usage else 'missing'}"
        )

    return run


def tools_roundtrip(model):
    def run():
        messages = user("What is the weather in Paris? Use the tool.")
        r = rh.completion(
            model=model,
            messages=messages,
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=1024,
            drop_params=True,
        )
        msg = r.choices[0].message
        assert (
            msg.tool_calls
        ), f"no tool call, content={msg.content!r}"
        call = msg.tool_calls[0]
        args = json.loads(call.function.arguments)
        messages = messages + [
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [call.model_dump()],
            },
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": "Sunny, 21C",
            },
        ]
        r2 = rh.completion(
            model=model,
            messages=messages,
            tools=[WEATHER_TOOL],
            max_tokens=256,
            drop_params=True,
        )
        return f"call={call.function.name}({args}) finish={r.choices[0].finish_reason} -> {r2.choices[0].message.content[:50]!r}"

    return run


def structured(model):
    def run():
        r = rh.completion(
            model=model,
            messages=user("What is the capital of France?"),
            response_format=Capital,
            max_tokens=512,
            drop_params=True,
        )
        parsed = Capital.model_validate_json(
            r.choices[0].message.content
        )
        return f"{parsed}"

    return run


def async_stream(model):
    async def go():
        s = await rh.acompletion(
            model=model,
            messages=user("Say hello in French, one word."),
            max_tokens=64,
            stream=True,
            drop_params=True,
        )
        text = ""
        async for chunk in s:
            if chunk.choices and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
        r = await rh.acompletion(
            model=model,
            messages=user("Reply OK"),
            max_tokens=32,
            drop_params=True,
        )
        return f"stream={text.strip()!r} non-stream={r.choices[0].message.content!r}"

    return lambda: asyncio.run(go())


def thinking(model):
    def run():
        r = rh.completion(
            model=model,
            messages=user("What is 17 * 23? Answer with the number."),
            reasoning_effort="low",
            max_tokens=4000,
            drop_params=True,
        )
        msg = r.choices[0].message
        has = bool(
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "thinking_blocks", None)
        )
        return f"{msg.content!r} reasoning_returned={has} reasoning_tokens={getattr(r.usage.completion_tokens_details, 'reasoning_tokens', None) if r.usage.completion_tokens_details else None}"

    return run


def prompt_cache(model):
    def run():
        system = "You are a meticulous assistant. " + " ".join(
            f"Rule {i}: always be precise and concise."
            for i in range(400)
        )
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": "Reply with OK."},
        ]
        r1 = rh.completion(
            model=model, messages=messages, max_tokens=16
        )
        r2 = rh.completion(
            model=model, messages=messages, max_tokens=16
        )
        u1, u2 = r1.usage, r2.usage
        return f"call1 write={getattr(u1, 'cache_creation_input_tokens', None)} read={u1.prompt_tokens_details.cached_tokens}; call2 read={u2.prompt_tokens_details.cached_tokens} prompt={u2.prompt_tokens}"

    return run


def bad_key(model):
    def run():
        try:
            rh.completion(
                model=model,
                messages=user("hi"),
                api_key="sk-invalid",
                num_retries=0,
            )
        except rh.AuthenticationError as e:
            return f"AuthenticationError status={e.status_code} provider={e.llm_provider}"
        raise AssertionError("no error raised")

    return run


def embed():
    r = rh.embedding(
        model="text-embedding-3-small", input=["hello world"]
    )
    return f"dims={len(r.data[0].embedding)}"


def list_models():
    from routehub.get_all_models import get_all_models

    entries = get_all_models()
    providers = sorted({e["provider"] for e in entries})
    return f"{len(entries)} models from {providers}"


OPENAI = os.environ.get("LIVE_OPENAI_MODEL", "gpt-5.4-mini")
CLAUDE = os.environ.get("LIVE_CLAUDE_MODEL", "claude-haiku-4-5")
CLAUDE_THINK = os.environ.get(
    "LIVE_CLAUDE_THINK_MODEL", "claude-sonnet-4-6"
)

check(f"openai basic ({OPENAI})", basic(OPENAI))
check("openai stream+usage", stream(OPENAI))
check("openai tools roundtrip", tools_roundtrip(OPENAI))
check("openai structured (pydantic)", structured(OPENAI))
check("openai async", async_stream(OPENAI))
check("openai reasoning_effort", thinking(OPENAI))
check("openai bad key", bad_key(OPENAI))
check("openai embedding", embed)
check("get_all_models", list_models)
check(f"anthropic basic ({CLAUDE})", basic(CLAUDE))
check("anthropic stream+usage", stream(CLAUDE))
check("anthropic tools roundtrip", tools_roundtrip(CLAUDE))
check("anthropic structured (pydantic)", structured(CLAUDE))
check("anthropic async", async_stream(CLAUDE))
check(f"anthropic thinking ({CLAUDE_THINK})", thinking(CLAUDE_THINK))
check(
    f"anthropic prompt cache ({CLAUDE_THINK})",
    prompt_cache(CLAUDE_THINK),
)
check("anthropic bad key", bad_key(CLAUDE))
check("groq basic", basic("groq/llama-3.3-70b-versatile"))
check("groq stream", stream("groq/llama-3.3-70b-versatile"))
check(
    "gemini basic",
    basic("gemini/gemini-3-flash-preview", max_tokens=2000),
)
check(
    "gemini stream",
    stream("gemini/gemini-3-flash-preview", max_tokens=2000),
)
check(
    "gemini tools roundtrip",
    tools_roundtrip("gemini/gemini-3-flash-preview"),
)
check(
    "together basic",
    basic("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"),
)
check("xai basic", basic("xai/grok-3-mini"))
check("openrouter basic", basic("openrouter/openai/gpt-4o-mini"))

width = max(len(r[0]) for r in RESULTS)
for name, ok, detail, seconds in RESULTS:
    print(
        f"{'PASS' if ok else 'FAIL'}  {name:{width}s}  {seconds:5.2f}s  {detail}"
    )
failed = sum(1 for r in RESULTS if not r[1])
print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} passed")
sys.exit(1 if failed else 0)
