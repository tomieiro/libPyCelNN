"""Small helper utilities for documentation-facing messages."""

from __future__ import annotations


def optional_dependency_message(package: str, extra: str, feature: str) -> str:
    """Build a consistent optional dependency error message."""
    return (
        f"{feature} requires the optional dependency '{package}'. "
        f"Install it with `pip install celnn[{extra}]`."
    )
