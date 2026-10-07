"""Reusable SDK and HTTP clients, so connections stay pooled across calls."""

import asyncio
import importlib.util
import os
import threading
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Optional, Union

from routehub._json import _OFF_VALUES

_MAX_CLIENTS = 64
_lock = threading.Lock()
_clients: "OrderedDict[tuple, Any]" = OrderedDict()
_closers: "dict[int, Any]" = {}


def _loop_id(is_async: bool) -> Optional[Any]:
    """Identify the running event loop, since async clients are bound to one.

    Args:
        is_async (bool): Whether the client is asynchronous.

    Returns:
        Optional[Any]: The loop itself, or None for sync clients.
    """
    if not is_async:
        return None
    try:
        # The loop object, not its id, so a recycled id never matches a dead loop.
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _cached(key: tuple, build) -> Any:
    """Return the client for key, building and caching it when missing.

    Args:
        key (tuple): Cache key.
        build (Callable[[], Any]): Builds the client.

    Returns:
        Any: The cached client.
    """
    with _lock:
        client = _clients.get(key)
        if client is not None:
            _clients.move_to_end(key)
            return client
    client = build()
    with _lock:
        _clients[key] = client
        while len(_clients) > _MAX_CLIENTS:
            _clients.popitem(last=False)
    return client


@lru_cache(maxsize=None)
def _aiohttp_client() -> Any:
    """Return the OpenAI SDK's aiohttp-backed async client class.

    Used when the aiohttp extra is installed on openai 3.x, unless
    ROUTEHUB_USE_AIOHTTP is 0, false, no or off. Read once, on first use.

    Returns:
        Any: openai.DefaultAioHttpClient, or None to keep httpx.
    """
    setting = os.environ.get("ROUTEHUB_USE_AIOHTTP", "")
    if setting.strip().lower() in _OFF_VALUES:
        return None
    from routehub._http import httpx

    if httpx.__name__ != "httpx2":
        return None
    if importlib.util.find_spec("aiohttp") is None:
        return None
    import openai

    return openai.DefaultAioHttpClient


async def _close_on_shutdown(client: Any) -> Any:
    """Close client when its event loop shuts down async generators.

    aiohttp warns about every session still open at exit, and
    asyncio.run closes pending async generators before its loop.

    Args:
        client (Any): An aiohttp-backed async client.

    Yields:
        None: Once, so the generator stays registered with the loop.
    """
    try:
        yield
    finally:
        _closers.pop(id(client), None)
        await client.aclose()


def _close_with_loop(client: Any) -> Any:
    """Arrange for client to be closed when the running loop ends.

    Args:
        client (Any): An aiohttp-backed async client.

    Returns:
        Any: The same client.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return client
    closer = _close_on_shutdown(client)
    _closers[id(client)] = closer
    asyncio.ensure_future(closer.__anext__())
    return client


def _limits(keepalive_expiry: float) -> Any:
    """Build pool limits that keep idle connections open between calls.

    Args:
        keepalive_expiry (float): Seconds an idle connection stays pooled.

    Returns:
        Any: An httpx-compatible Limits.
    """
    from routehub._http import httpx

    return httpx.Limits(
        max_connections=1000,
        max_keepalive_connections=100,
        keepalive_expiry=keepalive_expiry,
    )


def openai_client(
    provider: str,
    api_key: Optional[str],
    api_base: Optional[str],
    *,
    is_async: bool,
    max_retries: int,
    ssl_verify: Union[bool, str],
    api_version: Optional[str] = None,
    organization: Optional[str] = None,
    keepalive_expiry: float = 60.0,
) -> Any:
    """Return a cached OpenAI or Azure OpenAI client.

    Args:
        provider (str): Provider name; "azure" builds an Azure client.
        api_key (Optional[str]): API key.
        api_base (Optional[str]): Base URL, or the Azure endpoint.
        is_async (bool): Whether to build the async client.
        max_retries (int): Retries the SDK performs on retryable errors.
        ssl_verify (Union[bool, str]): TLS verification flag or CA bundle path.
        api_version (Optional[str]): Azure API version.
        organization (Optional[str]): OpenAI organization id.
        keepalive_expiry (float): Seconds an idle connection stays pooled.

    Returns:
        Any: An OpenAI SDK client.
    """
    aiohttp_client = _aiohttp_client() if is_async else None
    key = (
        "openai",
        provider == "azure",
        is_async,
        _loop_id(is_async),
        api_key,
        api_base,
        api_version,
        organization,
        max_retries,
        ssl_verify,
        keepalive_expiry,
        aiohttp_client is not None,
    )

    def build() -> Any:
        import openai

        factory = aiohttp_client or (
            openai.DefaultAsyncHttpxClient
            if is_async
            else openai.DefaultHttpxClient
        )
        http_client = factory(
            verify=ssl_verify, limits=_limits(keepalive_expiry)
        )
        if aiohttp_client is not None:
            _close_with_loop(http_client)
        if provider == "azure":
            cls = (
                openai.AsyncAzureOpenAI
                if is_async
                else openai.AzureOpenAI
            )
            return cls(
                api_key=api_key,
                azure_endpoint=api_base,
                api_version=api_version,
                max_retries=max_retries,
                http_client=http_client,
            )
        cls = openai.AsyncOpenAI if is_async else openai.OpenAI
        return cls(
            api_key=api_key,
            base_url=api_base,
            organization=organization,
            max_retries=max_retries,
            http_client=http_client,
        )

    return _cached(key, build)


def http_client(
    *,
    is_async: bool,
    ssl_verify: Union[bool, str],
    keepalive_expiry: float = 60.0,
) -> Any:
    """Return a cached raw HTTP client for native provider adapters.

    Args:
        is_async (bool): Whether to build the async client.
        ssl_verify (Union[bool, str]): TLS verification flag or CA bundle path.
        keepalive_expiry (float): Seconds an idle connection stays pooled.

    Returns:
        Any: An httpx-compatible Client or AsyncClient.
    """
    aiohttp_client = _aiohttp_client() if is_async else None
    key = (
        "http",
        is_async,
        _loop_id(is_async),
        ssl_verify,
        keepalive_expiry,
        aiohttp_client is not None,
    )

    def build() -> Any:
        from routehub._http import httpx

        cls = aiohttp_client or (
            httpx.AsyncClient if is_async else httpx.Client
        )
        client = cls(
            verify=ssl_verify,
            follow_redirects=True,
            limits=_limits(keepalive_expiry),
        )
        if aiohttp_client is not None:
            _close_with_loop(client)
        return client

    return _cached(key, build)
