"""Stream wrappers that yield ChatCompletionChunks and map provider errors."""

from typing import Any, AsyncIterator, Callable, Iterator, Optional

from routehub.exceptions import map_exception


class ChatStream:
    """Iterator over ChatCompletionChunk objects from any provider."""

    def __init__(
        self,
        chunks: Iterator[Any],
        close: Optional[Callable[[], None]],
        llm_provider: str,
        model: str,
    ) -> None:
        """Wrap a chunk iterator.

        Args:
            chunks (Iterator[Any]): The provider's chunk iterator.
            close (Optional[Callable[[], None]]): Releases the connection.
            llm_provider (str): Provider name, for error mapping.
            model (str): Model name, for error mapping.
        """
        self._chunks = chunks
        self._close = close
        self._closed = False
        self.llm_provider = llm_provider
        self.model = model

    def __iter__(self) -> "ChatStream":
        """Return the stream itself.

        Returns:
            ChatStream: This stream.
        """
        return self

    def __next__(self) -> Any:
        """Return the next chunk.

        Returns:
            Any: A ChatCompletionChunk.
        """
        try:
            return next(self._chunks)
        except StopIteration:
            self.close()
            raise
        except Exception as error:
            self.close()
            mapped = map_exception(
                error, self.llm_provider, self.model
            )
            if mapped is error:
                raise
            raise mapped from error

    def close(self) -> None:
        """Release the underlying connection."""
        if not self._closed:
            self._closed = True
            if self._close is not None:
                self._close()

    def __enter__(self) -> "ChatStream":
        """Enter a with block.

        Returns:
            ChatStream: This stream.
        """
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the stream when the with block ends.

        Args:
            *exc_info (Any): Exception details, if any.
        """
        self.close()


class AsyncChatStream:
    """Async iterator over ChatCompletionChunk objects from any provider."""

    def __init__(
        self,
        chunks: AsyncIterator[Any],
        close: Optional[Callable[[], Any]],
        llm_provider: str,
        model: str,
    ) -> None:
        """Wrap an async chunk iterator.

        Args:
            chunks (AsyncIterator[Any]): The provider's chunk iterator.
            close (Optional[Callable[[], Any]]): Awaitable that releases the
                connection.
            llm_provider (str): Provider name, for error mapping.
            model (str): Model name, for error mapping.
        """
        self._chunks = chunks
        self._close = close
        self._closed = False
        self.llm_provider = llm_provider
        self.model = model

    def __aiter__(self) -> "AsyncChatStream":
        """Return the stream itself.

        Returns:
            AsyncChatStream: This stream.
        """
        return self

    async def __anext__(self) -> Any:
        """Return the next chunk.

        Returns:
            Any: A ChatCompletionChunk.
        """
        try:
            return await self._chunks.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise
        except Exception as error:
            await self.aclose()
            mapped = map_exception(
                error, self.llm_provider, self.model
            )
            if mapped is error:
                raise
            raise mapped from error

    async def aclose(self) -> None:
        """Release the underlying connection."""
        if not self._closed:
            self._closed = True
            if self._close is not None:
                await self._close()

    async def __aenter__(self) -> "AsyncChatStream":
        """Enter an async with block.

        Returns:
            AsyncChatStream: This stream.
        """
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        """Close the stream when the async with block ends.

        Args:
            *exc_info (Any): Exception details, if any.
        """
        await self.aclose()
