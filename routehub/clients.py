"""Reusable SDK and HTTP clients, so connections stay pooled across calls."""

import asyncio
import threading
from collections import OrderedDict
from typing import Any, Optional, Union

_MAX_CLIENTS = 64
_lock = threading.Lock()
_clients: "OrderedDict[tuple, Any]" = OrderedDict()


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

    Returns:
        Any: An OpenAI SDK client.
    """
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
    )

    def build() -> Any:
        import openai

        http_client = None
        if ssl_verify is not True:
            factory = (
                openai.DefaultAsyncHttpxClient
                if is_async
                else openai.DefaultHttpxClient
            )
            http_client = factory(verify=ssl_verify)
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
    *, is_async: bool, ssl_verify: Union[bool, str]
) -> Any:
    """Return a cached raw HTTP client for native provider adapters.

    Args:
        is_async (bool): Whether to build the async client.
        ssl_verify (Union[bool, str]): TLS verification flag or CA bundle path.

    Returns:
        Any: An httpx-compatible Client or AsyncClient.
    """
    key = ("http", is_async, _loop_id(is_async), ssl_verify)

    def build() -> Any:
        from routehub._http import httpx

        cls = httpx.AsyncClient if is_async else httpx.Client
        return cls(verify=ssl_verify, follow_redirects=True)

    return _cached(key, build)
