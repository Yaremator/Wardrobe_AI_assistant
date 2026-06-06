from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Iterator

from langchain_core.runnables import RunnableConfig


def _langfuse_base_url() -> str | None:
    return (
        os.getenv("LANGFUSE_BASE_URL", "").strip()
        or os.getenv("LANGFUSE_HOST", "").strip()
        or None
    )


def is_langfuse_enabled() -> bool:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    return bool(public_key and secret_key)


def _normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, str] | None:
    if not metadata:
        return None

    normalized: dict[str, str] = {}
    for key, value in metadata.items():
        text = str(value)
        if len(text) > 200:
            text = text[:197] + "..."
        normalized[str(key)] = text
    return normalized


@lru_cache(maxsize=1)
def _ensure_langfuse_client() -> bool:
    from langfuse import Langfuse

    kwargs: dict[str, Any] = {
        "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", "").strip(),
        "secret_key": os.getenv("LANGFUSE_SECRET_KEY", "").strip(),
    }
    base_url = _langfuse_base_url()
    if base_url:
        kwargs["host"] = base_url

    Langfuse(**kwargs)
    return True


def _get_langfuse_client():
    if not is_langfuse_enabled():
        raise RuntimeError("Langfuse is not configured.")
    _ensure_langfuse_client()
    from langfuse import get_client

    return get_client()


@lru_cache(maxsize=1)
def _get_callback_handler():
    _ensure_langfuse_client()
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def flush_langfuse() -> None:
    if not is_langfuse_enabled():
        return

    _get_langfuse_client().flush()


def verify_langfuse_connection() -> bool:
    if not is_langfuse_enabled():
        return False

    try:
        return _get_langfuse_client().auth_check()
    except Exception:
        return False


@dataclass
class LangfuseTraceRun:
    config: RunnableConfig
    _set_output: Callable[[Any], None] = field(repr=False)

    def set_output(self, output: Any) -> None:
        self._set_output(output)


def _build_langchain_config(
    *,
    handler: Any,
    name: str,
    user_id: int | str | None,
    session_id: str | None,
    tags: list[str] | None,
) -> RunnableConfig:
    config: RunnableConfig = {"callbacks": [handler], "run_name": name}
    config_metadata: dict[str, Any] = {}
    if user_id is not None:
        config_metadata["langfuse_user_id"] = str(user_id)
    if session_id:
        config_metadata["langfuse_session_id"] = session_id
    if tags:
        config_metadata["langfuse_tags"] = tags
    if config_metadata:
        config["metadata"] = config_metadata
    return config


@contextmanager
def langfuse_trace(
    *,
    name: str,
    user_id: int | str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    trace_input: Any = None,
) -> Iterator[LangfuseTraceRun | None]:
    """Wrap a LangChain/LangGraph run with a named trace and explicit input/output."""
    if not is_langfuse_enabled():
        yield None
        return

    from langfuse import propagate_attributes

    client = _get_langfuse_client()
    handler = _get_callback_handler()

    propagate_kwargs: dict[str, Any] = {"trace_name": name}
    if user_id is not None:
        propagate_kwargs["user_id"] = str(user_id)
    if session_id:
        propagate_kwargs["session_id"] = session_id
    if tags:
        propagate_kwargs["tags"] = tags
    normalized_metadata = _normalize_metadata(metadata)
    if normalized_metadata:
        propagate_kwargs["metadata"] = normalized_metadata

    with client.start_as_current_observation(
        as_type="span",
        name=name,
        input=trace_input,
    ) as span:
        with propagate_attributes(**propagate_kwargs):
            config = _build_langchain_config(
                handler=handler,
                name=name,
                user_id=user_id,
                session_id=session_id,
                tags=tags,
            )

            def set_output(output: Any) -> None:
                span.update(output=output)

            try:
                yield LangfuseTraceRun(config=config, _set_output=set_output)
            finally:
                flush_langfuse()


@contextmanager
def observe_operation(
    name: str,
    *,
    input_data: Any = None,
    user_id: int | str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
) -> Iterator[Any]:
    """Trace non-LangChain operations with explicit, minimal input data."""
    if not is_langfuse_enabled():
        yield None
        return

    from langfuse import propagate_attributes

    client = _get_langfuse_client()

    propagate_kwargs: dict[str, Any] = {}
    if user_id is not None:
        propagate_kwargs["user_id"] = str(user_id)
    if session_id:
        propagate_kwargs["session_id"] = session_id
    if tags:
        propagate_kwargs["tags"] = tags

    with client.start_as_current_observation(
        as_type="span",
        name=name,
        input=input_data,
    ) as observation:
        with propagate_attributes(**propagate_kwargs):
            try:
                yield observation
            finally:
                flush_langfuse()
