"""Integration script for the enhanced serialization system.

This module initializes the enhanced serialization system and integrates
it with the execution coordinator in the server.py module.
"""

import logging

from robotmcp.utils.enhanced_serialization_integration import (
    apply_enhanced_serialization,
)

logger = logging.getLogger(__name__)


def initialize_enhanced_serialization(execution_engine):
    """
    Initialize enhanced serialization for the MCP server.

    This function should be called during server initialization to set up
    the enhanced serialization system.

    Args:
        execution_engine: The execution coordinator instance from the server
    """
    logger.info("Initializing enhanced serialization system...")
    apply_enhanced_serialization(execution_engine)
    logger.info("Enhanced serialization system initialized")
