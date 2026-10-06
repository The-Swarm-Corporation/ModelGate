"""The HTTP library the installed OpenAI SDK is built on."""

from importlib.metadata import PackageNotFoundError, version


def _load():
    """Import httpx2 for openai 3.x and httpx for earlier versions.

    Returns:
        module: The httpx-compatible module.
    """
    try:
        major = int(version("openai").split(".")[0])
    except (PackageNotFoundError, ValueError):
        major = 0
    if major >= 3:
        import httpx2

        return httpx2
    try:
        import httpx

        return httpx
    except ImportError:
        import httpx2

        return httpx2


httpx = _load()
