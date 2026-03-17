"""Keyword Resolution Aggregate Root (ADR-019 Phase 5).

Orchestrates BDD prefix stripping, embedded argument matching,
and normal keyword lookup in a single resolution pipeline.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

from robotmcp.models.library_models import KeywordInfo
from .services import BddPrefixService, EmbeddedMatcherService
from .value_objects import BddPrefix, EmbeddedMatch

logger = logging.getLogger(__name__)


class KeywordResolver:
    """Aggregate root for keyword resolution.

    Orchestrates:
    1. BDD prefix stripping
    2. Normal keyword lookup (which includes embedded matching as fallback)

    This aggregate does NOT own the keyword cache -- it delegates to
    KeywordDiscovery via find_keyword_fn. It provides a clean API for
    server.py and keyword_executor.py to resolve keywords through all
    resolution layers.
    """

    @staticmethod
    def resolve(
        keyword_name: str,
        find_keyword_fn: Callable[..., Optional[KeywordInfo]],
        **find_kwargs,
    ) -> Tuple[Optional[KeywordInfo], BddPrefix, Optional[EmbeddedMatch]]:
        """Resolve a keyword through BDD stripping, then normal/embedded lookup.

        Args:
            keyword_name: Raw keyword name (may have BDD prefix)
            find_keyword_fn: Callable that does the actual keyword lookup
            **find_kwargs: Extra kwargs for find_keyword_fn (active_library, etc.)

        Returns:
            (keyword_info, bdd_result, embedded_match)
        """
        # Step 1: Strip BDD prefix
        bdd_result = BddPrefixService.strip_prefix(keyword_name)
        effective_name = bdd_result.stripped_name

        if bdd_result.has_prefix:
            logger.debug(
                f"KeywordResolver: stripped BDD prefix '{bdd_result.prefix_type.value}' "
                f"from '{bdd_result.original_name}' -> '{effective_name}'"
            )

        # Step 2: Normal keyword lookup (includes embedded matching as fallback)
        keyword_info = find_keyword_fn(effective_name, **find_kwargs)

        # Step 3: Extract embedded match if present
        embedded_match = (
            getattr(keyword_info, '_embedded_match', None)
            if keyword_info else None
        )

        if embedded_match:
            logger.debug(
                f"KeywordResolver: embedded match '{embedded_match.template_name}' "
                f"with args {embedded_match.extracted_args}"
            )

        return keyword_info, bdd_result, embedded_match
