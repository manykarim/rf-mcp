from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import pytest

from robotmcp.core.keyword_discovery import KeywordDiscovery
from robotmcp.core.library_manager import LibraryManager
from robotmcp.utils.rf_libdoc_integration import get_rf_doc_storage

PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "examples" / "plugins" / "doctest_plugin"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

import rfmcp_doctest_plugin.ai as ai_module  # noqa: E402
import rfmcp_doctest_plugin.pdf as pdf_module  # noqa: E402
import rfmcp_doctest_plugin.print_jobs as print_module  # noqa: E402
import rfmcp_doctest_plugin.visual as visual_module  # noqa: E402
from rfmcp_doctest_plugin.ai import DocTestAiPlugin  # noqa: E402
from rfmcp_doctest_plugin.pdf import DocTestPdfPlugin  # noqa: E402
from rfmcp_doctest_plugin.print_jobs import DocTestPrintJobPlugin  # noqa: E402
from rfmcp_doctest_plugin.visual import DocTestVisualPlugin  # noqa: E402

from DocTest.VisualTest import VisualTest  # noqa: E402

class DummySession:
    def __init__(self) -> None:
        self.session_id = "dummy"
        self.variables: dict[str, Any] = {}
        self.imports: List[tuple[str, bool]] = []

    def import_library(self, name: str, force: bool = False) -> None:
        self.imports.append((name, force))


def _raise_print_job_failure() -> None:
    reference_print_job = SimpleNamespace(
        jobtype="pcl",
        properties=[{"property": "copies", "value": ["1"]}],
    )
    test_print_job = SimpleNamespace(
        jobtype="pcl",
        properties=[{"property": "copies", "value": ["2"]}],
    )
    list_difference = [
        {"file": "reference", "property": "copies", "value": "1"},
        {"file": "test", "property": "copies", "value": "2"},
    ]
    # Ensure variables stay referenced for traceback inspection
    _ = reference_print_job, test_print_job, list_difference
    raise AssertionError("The compared print jobs are different.")


def _raise_ai_failure() -> None:
    document = "invoice.pdf"
    expected = "Company logo"

    class DummyDecision:
        def __init__(self) -> None:
            self.reason = "vision model rejected"

        def model_dump(self) -> dict[str, Any]:
            return {"decision": "reject", "reason": self.reason}

    decision = DummyDecision()
    _ = document, expected, decision
    raise AssertionError("Expected object 'Company logo' not found in 'invoice.pdf'.")


def _run_override(
    handler,
    session: DummySession,
    keyword_name: str,
    args: Sequence[str],
):
    return handler(session, keyword_name, list(args), None)


def test_visual_plugin_aliases():
    plugin = DocTestVisualPlugin()
    mapping = plugin.get_keyword_library_map()
    assert mapping["doctest.visualtest.compare images"] == "DocTest.VisualTest"
    overrides = plugin.get_keyword_overrides()
    assert "doctest.visualtest.compare images" in overrides


def test_keyword_discovery_respects_robot_auto_keywords():
    discovery = KeywordDiscovery()
    info = discovery.extract_library_info("DocTest.VisualTest", VisualTest())
    assert "Compare Images" in info.keywords
    assert "Any" not in info.keywords


@pytest.mark.asyncio
async def test_print_plugin_failure_captures_differences(monkeypatch: pytest.MonkeyPatch):
    plugin = DocTestPrintJobPlugin()
    session = DummySession()

    class RaisingBuiltIn:
        def run_keyword(self, keyword_name: str, args: Sequence[str]) -> None:
            _raise_print_job_failure()

    monkeypatch.setattr(print_module, "BuiltIn", lambda: RaisingBuiltIn())

    handler = plugin.get_keyword_overrides()["compare print jobs"]
    result = await _run_override(
        handler,
        session,
        "Compare Print Jobs",
        ["pcl", "reference.pcl", "candidate.pcl"],
    )

    assert result["success"] is False
    assert ("DocTest.PrintJobTests", False) in session.imports
    summary = session.variables["_doctest_print_result"]
    assert summary["status"] == "failed"
    assert summary["differences"][0]["property"] == "copies"

    state_provider = plugin.get_state_provider()
    assert state_provider is not None
    state = await state_provider.get_application_state(session)
    assert state["print"]["status"] == "failed"


@pytest.mark.asyncio
async def test_visual_plugin_handles_runtime_error(monkeypatch: pytest.MonkeyPatch):
    plugin = DocTestVisualPlugin()
    session = DummySession()

    class RaisingBuiltIn:
        def run_keyword(self, keyword_name: str, args: Sequence[str]) -> None:
            raise ValueError("Cannot load image from candidate.png")

    monkeypatch.setattr(visual_module, "BuiltIn", lambda: RaisingBuiltIn())

    handler = plugin.get_keyword_overrides()["doctest.visualtest.compare images"]
    result = await _run_override(
        handler,
        session,
        "DocTest.VisualTest.Compare Images",
        ["reference.png", "candidate.png"],
    )

    assert result["success"] is False
    assert ("DocTest.VisualTest", False) in session.imports
    summary = session.variables["_doctest_visual_result"]
    assert summary["exception"] == "ValueError"
    assert "candidate.png" in summary["message"]


@pytest.mark.asyncio
async def test_print_plugin_success_records_summary(monkeypatch: pytest.MonkeyPatch):
    plugin = DocTestPrintJobPlugin()
    session = DummySession()

    class SuccessBuiltIn:
        def run_keyword(self, keyword_name: str, args: Sequence[str]) -> None:
            return None

    monkeypatch.setattr(print_module, "BuiltIn", lambda: SuccessBuiltIn())

    handler = plugin.get_keyword_overrides()["compare print jobs"]
    result = await _run_override(
        handler,
        session,
        "Compare Print Jobs",
        ["pcl", "reference.pcl", "candidate.pcl"],
    )

    assert result["success"] is True
    summary = session.variables["_doctest_print_result"]
    assert summary["status"] == "passed"
    assert result["state_updates"]["doctest"]["print"]["status"] == "passed"


@pytest.mark.asyncio
async def test_pdf_plugin_handles_exception(monkeypatch: pytest.MonkeyPatch):
    plugin = DocTestPdfPlugin()
    session = DummySession()

    class RaisingBuiltIn:
        def run_keyword(self, keyword_name: str, args: Sequence[str]) -> None:
            raise RuntimeError("LLM timeout")

    monkeypatch.setattr(pdf_module, "BuiltIn", lambda: RaisingBuiltIn())

    handler = plugin.get_keyword_overrides()["doctest.pdftest.compare pdf documents"]
    result = await _run_override(
        handler,
        session,
        "DocTest.PdfTest.Compare Pdf Documents",
        ["reference.pdf", "candidate.pdf"],
    )

    assert result["success"] is False
    assert ("DocTest.PdfTest", False) in session.imports
    summary = session.variables["_doctest_pdf_result"]
    assert summary["exception"] == "RuntimeError"
    assert summary["message"] == "LLM timeout"


def test_libdoc_storage_loads_visual_on_demand():
    storage = get_rf_doc_storage()
    storage.refresh_library("DocTest.VisualTest")
    lib = storage.get_library_documentation("DocTest.VisualTest")
    assert lib is not None
    assert len(lib.keywords) >= 15


def test_library_manager_extracts_visual_keywords():
    manager = LibraryManager()
    discovery = KeywordDiscovery()
    assert manager.try_import_library("DocTest.VisualTest", discovery) is True
    keywords = manager.libraries["DocTest.VisualTest"].keywords
    assert "Compare Images" in keywords
    assert len(keywords) >= 15


def test_visual_plugin_normalises_windows_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    plugin = DocTestVisualPlugin()
    session = DummySession()

    ref = tmp_path / "ref.png"
    cand = tmp_path / "cand.png"
    ref.write_bytes(b"")
    cand.write_bytes(b"")

    recorded: Dict[str, Any] = {}

    class BuiltInStub:
        def run_keyword(self, keyword_name: str, args: Sequence[str]) -> None:
            recorded["args"] = list(args)
            raise AssertionError("boom")

    monkeypatch.setattr(visual_module, "BuiltIn", lambda: BuiltInStub())

    win_ref = str(ref).replace("/", "\\")
    win_cand = str(cand).replace("/", "\\")

    plugin._execute_compare_images(session, "Compare Images", [win_ref, win_cand])

    normalised = recorded["args"]
    assert normalised[0].startswith(ref.as_posix())
    assert "\\" not in normalised[0]
    assert "\r" not in normalised[0]


def test_pdf_plugin_normalises_windows_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    plugin = DocTestPdfPlugin()
    session = DummySession()

    ref = tmp_path / "ref.pdf"
    cand = tmp_path / "cand.pdf"
    ref.write_bytes(b"%PDF-1.4")
    cand.write_bytes(b"%PDF-1.4")

    recorded: Dict[str, Any] = {}

    class BuiltInStub:
        def run_keyword(self, keyword_name: str, args: Sequence[str]) -> None:
            recorded["args"] = list(args)
            raise RuntimeError("boom")

    monkeypatch.setattr(pdf_module, "BuiltIn", lambda: BuiltInStub())

    win_ref = str(ref).replace("/", "\\")
    win_cand = str(cand).replace("/", "\\")

    plugin._execute_pdf_keyword(session, "Compare Pdf Documents", [win_ref, win_cand])

    normalised = recorded["args"]
    assert normalised[0].startswith(ref.as_posix())
    assert normalised[1].startswith(cand.as_posix())


def test_print_plugin_normalises_windows_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    plugin = DocTestPrintJobPlugin()
    session = DummySession()

    ref = tmp_path / "ref.pcl"
    cand = tmp_path / "cand.pcl"
    ref.write_bytes(b"")
    cand.write_bytes(b"")

    recorded: Dict[str, Any] = {}

    class BuiltInStub:
        def run_keyword(self, keyword_name: str, args: Sequence[str]) -> None:
            recorded["args"] = list(args)
            raise AssertionError("boom")

    monkeypatch.setattr(print_module, "BuiltIn", lambda: BuiltInStub())

    win_ref = str(ref).replace("/", "\\")
    win_cand = str(cand).replace("/", "\\")

    plugin._execute_compare_print_jobs(
        session,
        "Compare Print Jobs",
        ["pcl", win_ref, win_cand],
    )

    normalised = recorded["args"]
    assert normalised[1].startswith(ref.as_posix())
    assert normalised[2].startswith(cand.as_posix())


@pytest.mark.asyncio
async def test_ai_plugin_failure_includes_context(monkeypatch: pytest.MonkeyPatch):
    plugin = DocTestAiPlugin()
    session = DummySession()

    class RaisingBuiltIn:
        def run_keyword(self, keyword_name: str, args: Sequence[str]) -> None:
            _raise_ai_failure()

    monkeypatch.setattr(ai_module, "BuiltIn", lambda: RaisingBuiltIn())

    handler = plugin.get_keyword_overrides()["image should contain"]
    result = await _run_override(
        handler,
        session,
        "Image Should Contain",
        ["invoice.pdf", "Company logo"],
    )

    assert result["success"] is False
    summary = session.variables["_doctest_ai_result"]
    assert summary["status"] == "failed"
    assert summary["context"]["document"] == "invoice.pdf"
    assert "decision" in summary["context"]


@pytest.mark.asyncio
async def test_ai_plugin_success_stores_llm_output(monkeypatch: pytest.MonkeyPatch):
    plugin = DocTestAiPlugin()
    session = DummySession()

    class SuccessBuiltIn:
        def run_keyword(self, keyword_name: str, args: Sequence[str]) -> str:
            return "LLM response text"

    monkeypatch.setattr(ai_module, "BuiltIn", lambda: SuccessBuiltIn())

    handler = plugin.get_keyword_overrides()["get text with llm"]
    result = await _run_override(
        handler,
        session,
        "Get Text With LLM",
        ["document.pdf"],
    )

    assert result["success"] is True
    assert result["return_value"] == "LLM response text"
    summary = session.variables["_doctest_ai_result"]
    assert summary["status"] == "passed"
    assert summary["result"] == "LLM response text"
