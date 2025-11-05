"""Example DocTest plugins packaged for rf-mcp."""

from .visual import DocTestVisualPlugin
from .pdf import DocTestPdfPlugin
from .print_jobs import DocTestPrintJobPlugin
from .ai import DocTestAiPlugin

__all__ = [
    "DocTestVisualPlugin",
    "DocTestPdfPlugin",
    "DocTestPrintJobPlugin",
    "DocTestAiPlugin",
]
