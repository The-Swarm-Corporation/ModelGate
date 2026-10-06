"""JSON encoding with orjson when installed, falling back to the stdlib.

Set MODEL_GATE_USE_ORJSON to 0, false, no or off to always use the stdlib json
module. The variable is read once, on first use.
"""

import json
import os
from typing import Any, Union

_OFF_VALUES = frozenset({"0", "false", "no", "off"})


def _load_orjson() -> Any:
    """Import orjson unless it is missing or switched off.

    Returns:
        Any: The orjson module, or None to use the stdlib.
    """
    setting = os.environ.get("MODEL_GATE_USE_ORJSON", "")
    if setting.strip().lower() in _OFF_VALUES:
        return None
    try:
        import orjson
    except ImportError:
        return None
    return orjson


_orjson = _load_orjson()
USING_ORJSON = _orjson is not None


def loads(data: Union[str, bytes]) -> Any:
    """Decode JSON text.

    Args:
        data (Union[str, bytes]): JSON as text or UTF-8 bytes.

    Returns:
        Any: The decoded value.
    """
    if _orjson is not None:
        return _orjson.loads(data)
    return json.loads(data)


def dumps(value: Any) -> bytes:
    """Encode a value as compact UTF-8 JSON.

    Args:
        value (Any): A JSON-serializable value.

    Returns:
        bytes: The encoded JSON.
    """
    if _orjson is not None:
        try:
            return _orjson.dumps(
                value, option=_orjson.OPT_NON_STR_KEYS
            )
        except TypeError:
            # orjson rejects some values the stdlib accepts, such as integers past 64 bits.
            pass
    return json.dumps(value, separators=(",", ":")).encode()


def dumps_str(value: Any) -> str:
    """Encode a value as compact JSON text.

    Args:
        value (Any): A JSON-serializable value.

    Returns:
        str: The encoded JSON.
    """
    return dumps(value).decode()
