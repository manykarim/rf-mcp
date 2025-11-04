"""Example plugins for robotframework-doctestlibrary submodules."""

from .visual_test_plugin import DocTestVisualPlugin
from .pdf_test_plugin import DocTestPdfPlugin
from .print_job_plugin import DocTestPrintJobPlugin
from .ai_plugin import DocTestAiPlugin

__all__ = [
    "DocTestVisualPlugin",
    "DocTestPdfPlugin",
    "DocTestPrintJobPlugin",
    "DocTestAiPlugin",
]
