"""Instruction Domain Adapters.

This module contains anti-corruption layer adapters that translate
between the instruction domain model and external systems.
"""

from .fastmcp_adapter import FastMCPInstructionAdapter, InstructionTemplateType

__all__ = ["FastMCPInstructionAdapter", "InstructionTemplateType"]
