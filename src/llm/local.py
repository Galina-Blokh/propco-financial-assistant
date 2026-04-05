"""LLM abstraction: supports OpenAI (default) and optional local GGUF via llama-cpp-python.

Set LLM_BACKEND=local in .env and LOCAL_MODEL_PATH=/path/to/model.gguf to use a local model.
Defaults to OpenAI if no local model is configured.

Thread safety: llama-cpp-python is not async-native. All blocking calls are run via
asyncio.to_thread(). An asyncio.Lock serializes concurrent LLM calls across requests.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

_LOCAL_LLM: Any = None
_LLM_LOCK: asyncio.Lock | None = None
_LLM_LOCK_LOOP_ID: int | None = None


def _get_llm_lock() -> asyncio.Lock:
    """Return an asyncio.Lock bound to the *current* event loop.

    ``asyncio.Lock`` is tied to the loop it was created on.  Because each
    Streamlit query spawns a new loop via ``asyncio.run()``, we must recreate
    the lock whenever the loop identity changes.
    """
    global _LLM_LOCK, _LLM_LOCK_LOOP_ID
    loop_id = id(asyncio.get_running_loop())
    if _LLM_LOCK is None or _LLM_LOCK_LOOP_ID != loop_id:
        _LLM_LOCK = asyncio.Lock()
        _LLM_LOCK_LOOP_ID = loop_id
    return _LLM_LOCK


def _get_local_llm() -> Any:
    """Lazy-load the local GGUF model singleton."""
    global _LOCAL_LLM
    if _LOCAL_LLM is not None:
        return _LOCAL_LLM

    from src.utils.config import settings

    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError:
        raise RuntimeError(
            "llama-cpp-python is not installed. "
            "Run: pip install llama-cpp-python  (or set LLM_BACKEND=openai)"
        )

    model_path = settings.local_model_path
    if not model_path:
        raise RuntimeError("LOCAL_MODEL_PATH must be set when LLM_BACKEND=local")

    logger.info("Loading local GGUF model from %s ...", model_path)
    _LOCAL_LLM = Llama(
        model_path=model_path,
        n_ctx=settings.local_n_ctx,
        n_threads=settings.local_n_threads,
        n_gpu_layers=settings.local_n_gpu_layers,
        verbose=False,
    )
    logger.info("Local LLM loaded.")
    return _LOCAL_LLM


async def prewarm_llm() -> None:
    """Run a no-op inference to load model weights before the first real query."""
    from src.utils.config import settings
    if settings.llm_backend != "local":
        return
    try:
        llm = _get_local_llm()
        await asyncio.to_thread(
            lambda: llm.create_chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
            )
        )
        logger.info("LLM pre-warm complete.")
    except Exception as exc:
        logger.warning("LLM pre-warm failed (non-fatal): %s", exc)


async def chat_complete(
    system: str,
    user: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    json_mode: bool = False,
    model: str | None = None,
) -> str:
    """Unified async chat completion — routes to local or OpenAI backend.

    ``model`` overrides the default model for OpenAI calls (ignored for local).
    """
    from src.utils.config import settings

    if settings.llm_backend == "local":
        return await _local_complete_async(system, user, max_tokens, temperature)
    return await _openai_complete(system, user, max_tokens, temperature, json_mode, model)


async def chat_complete_stream(
    system: str,
    user: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    model: str | None = None,
) -> AsyncIterator[str]:
    """Async generator that yields token strings as they arrive from the LLM.

    For OpenAI: uses ``stream=True``.
    For local: falls back to non-streaming (yields the full response at once).
    """
    from src.utils.config import settings

    if settings.llm_backend == "local":
        full = await _local_complete_async(system, user, max_tokens, temperature)
        yield full
        return

    from src.utils.llm_client import get_llm_client

    kwargs: dict[str, Any] = {
        "model": model or settings.openai_model_reasoning,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    response = await get_llm_client().chat.completions.create(**kwargs)
    async for chunk in response:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


def _local_complete_sync(
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Blocking inference on the local GGUF model (call via asyncio.to_thread)."""
    llm = _get_local_llm()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return output["choices"][0]["message"]["content"].strip()


async def _local_complete_async(
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Run local GGUF inference off the event loop with serialization lock."""
    lock = _get_llm_lock()
    async with lock:
        return await asyncio.to_thread(
            _local_complete_sync, system, user, max_tokens, temperature
        )


async def _openai_complete(
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    model: str | None = None,
) -> str:
    """Run inference via OpenAI API."""
    from src.utils.llm_client import get_llm_client
    from src.utils.config import settings

    kwargs: dict[str, Any] = {
        "model": model or settings.openai_model_reasoning,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await get_llm_client().chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()
