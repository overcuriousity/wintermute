"""
Shared utilities for parsing LLM text output.

LLMs frequently wrap JSON in markdown code fences or add prose around it.
The helpers here centralise the extraction logic so individual modules don't
each implement their own slightly-different variant.
"""

import json
import re
from typing import Any


def strip_fences(text: str) -> str:
    """Remove markdown code fences (```json … ```) from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"(?:\r?\n)?```\s*$", "", text)
    return text.strip()


def parse_json_from_llm(text: str, expected_type: type) -> Any:
    """Extract a JSON value of *expected_type* from LLM output.

    Tries in order:
    1. Direct JSON parse.
    2. Strip markdown code fences, then parse.
    3. Scan for the outermost delimiters (``{…}`` or ``[…]``), then parse.

    For ``expected_type=dict`` a bare single-element list wrapping a dict is
    also accepted (some models wrap objects in an array).

    Raises ``ValueError`` if no valid JSON of the expected type is found.
    Raises ``TypeError`` if *expected_type* is not ``dict`` or ``list``.
    """
    if expected_type not in (dict, list):
        raise TypeError(
            f"parse_json_from_llm: expected_type must be dict or list, got {expected_type!r}"
        )
    type_name = "object" if expected_type is dict else "array"
    open_char, close_char = ("{", "}") if expected_type is dict else ("[", "]")

    text = text.strip()

    # 1. Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, expected_type):
            return result
        # dict: accept a single-element list wrapping a dict
        if (
            expected_type is dict
            and isinstance(result, list)
            and result
            and isinstance(result[0], dict)
        ):
            return result[0]
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences, then parse
    fenced = strip_fences(text)
    try:
        result = json.loads(fenced)
        if isinstance(result, expected_type):
            return result
    except json.JSONDecodeError:
        pass

    # 3. Outermost delimiter scan
    start = text.find(open_char)
    end = text.rfind(close_char)
    if start != -1 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            if isinstance(result, expected_type):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No JSON {type_name} found in response: {text[:200]!r}")
