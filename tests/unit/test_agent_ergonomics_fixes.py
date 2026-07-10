"""Agent-ergonomics fixes (change: agent-ergonomics-fixes).

Cluster 1 of the 2026-07-10 Docker/agent capability spikes: agents lost turns to
unhelpful failures, not missing capability.
- F2: execute_batch leaked a bare KeyError('keyword') on a malformed step.
- F5: session profiles blocked standard utility libraries (e.g. OperatingSystem
      in an api_testing session), forcing an import_library detour.
- F4/docs: the batch BDD limitation and safe suite persistence were undocumented.
"""

from __future__ import annotations

import pytest

from robotmcp.domains.batch_execution.aggregates import BatchExecution
from robotmcp.models.session_models import ExecutionSession, SessionType


class TestBatchStepValidation:
    def test_missing_keyword_gives_actionable_error_not_keyerror(self):
        with pytest.raises(ValueError) as ei:
            BatchExecution.create(session_id="s", steps_data=[{"arguments": ["x"]}])
        msg = str(ei.value)
        assert "Step 0" in msg and "keyword" in msg  # names index + the field

    def test_empty_keyword_rejected(self):
        with pytest.raises(ValueError):
            BatchExecution.create(session_id="s", steps_data=[{"keyword": "   "}])

    def test_non_dict_step_rejected(self):
        with pytest.raises(ValueError):
            BatchExecution.create(session_id="s", steps_data=["Log"])

    def test_valid_batch_still_builds(self):
        b = BatchExecution.create(
            session_id="s",
            steps_data=[
                {"keyword": "Log", "arguments": ["hi"]},
                {"keyword": "No Operation"},
            ],
        )
        assert [st.keyword for st in b.steps] == ["Log", "No Operation"]

    def test_dict_arguments_rejected_not_resolved_to_keys(self):
        # Spike §3.2: list(dict) yields the dict's KEYS; execute_batch used to
        # accept that silently and run with garbage args. Must reject like
        # execute_step does.
        with pytest.raises(ValueError) as ei:
            BatchExecution.create(
                session_id="s",
                steps_data=[{"keyword": "Start Process", "arguments": {"item": "calc"}}],
            )
        msg = str(ei.value)
        assert "Step 0" in msg and "must be a list" in msg

    def test_string_arguments_rejected(self):
        with pytest.raises(ValueError):
            BatchExecution.create(
                session_id="s",
                steps_data=[{"keyword": "Log", "arguments": "hello"}],
            )

    def test_list_arguments_still_accepted(self):
        b = BatchExecution.create(
            session_id="s", steps_data=[{"keyword": "Log", "arguments": ["hello"]}]
        )
        assert b.steps[0].args == ["hello"]

    def test_second_step_index_reported(self):
        with pytest.raises(ValueError) as ei:
            BatchExecution.create(
                session_id="s",
                steps_data=[{"keyword": "Log", "arguments": ["ok"]}, {"arguments": []}],
            )
        assert "Step 1" in str(ei.value)


class TestUtilityLibrariesAllowed:
    def _session(self, stype: SessionType) -> ExecutionSession:
        s = ExecutionSession(session_id="s")
        s.session_type = stype
        return s

    def test_api_session_allows_operatingsystem_and_utils(self):
        allowed = self._session(SessionType.API_TESTING)._get_allowed_libraries_for_session_type()
        for lib in ("BuiltIn", "OperatingSystem", "Collections", "String", "DateTime", "Process"):
            assert lib in allowed, f"{lib} should be allowed in api_testing"
        assert "RequestsLibrary" in allowed  # profile core still present

    def test_utils_allowed_across_session_types(self):
        for stype in (SessionType.WEB_AUTOMATION, SessionType.XML_PROCESSING, SessionType.DESKTOP_TESTING):
            allowed = self._session(stype)._get_allowed_libraries_for_session_type()
            assert {"OperatingSystem", "Collections", "String", "DateTime", "Process"} <= allowed

    def test_web_libraries_still_excluded_for_nonweb(self):
        excluded = self._session(SessionType.API_TESTING).get_excluded_libraries_for_session()
        assert "Browser" in excluded and "SeleniumLibrary" in excluded


class TestAgentFacingDocs:
    def test_execute_batch_docstring_notes_bdd_limitation(self):
        import robotmcp.server as srv

        desc = getattr(srv.execute_batch, "description", None) or (srv.execute_batch.__doc__ or "")
        assert "bdd_group" in desc

    def test_workflow_guide_notes_output_path_persistence(self):
        from robotmcp.domains.instruction.value_objects import InstructionTemplate

        content = InstructionTemplate.standard().content
        assert "output_path" in content
        assert "Create File" in content  # warns against the corrupting round-trip
