"""Layer 1 join — synchronisation barrier after fan-out 1.

LangGraph calls this node only after both ``extractor`` and ``prefetcher``
have returned.  No heavy computation — just validates the merge and logs.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def join_layer1(state: dict) -> dict:
    """Log the merged extraction + prefetch results; pass through."""
    intent = state.get("intent")
    filters = state.get("filters", {})
    prefetch_keys = list(state.get("prefetch_results", {}).keys())

    logger.info(
        "layer1_joined: intent=%s filters=%s prefetch_keys=%s",
        intent,
        filters,
        prefetch_keys,
    )
    return {}
