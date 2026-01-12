"""Tests for listener module."""

from unittest.mock import MagicMock, PropertyMock

import pytest

from robotmcp.lib.listener import RecordingListener
from robotmcp.lib.recorder import ExecutionRecorder, StepType


class MockKeywordData:
    """Mock for Robot Framework keyword data object."""

    def __init__(
        self,
        name: str,
        args: list = None,
        libname: str = None,
        kw_type: str = None,
        assign: tuple = None,
    ):
        self.name = name
        self.args = args or []
        self.libname = libname
        self.type = kw_type
        self.assign = assign  # Variable assignment like ('${product_name}',)


class MockKeywordResult:
    """Mock for Robot Framework keyword result object."""

    def __init__(self, passed: bool = True, message: str = ""):
        self.passed = passed
        self.message = message


class TestRecordingListener:
    """Tests for RecordingListener class."""

    def test_listener_api_version(self):
        """Test listener uses API version 3."""
        listener = RecordingListener()
        assert listener.ROBOT_LISTENER_API_VERSION == 3

    def test_listener_with_custom_recorder(self):
        """Test listener can use custom recorder."""
        recorder = ExecutionRecorder()
        listener = RecordingListener(recorder=recorder)
        assert listener.recorder is recorder

    def test_listener_excludes_recording_keywords(self):
        """Test listener excludes Start Recording, Stop Recording, etc."""
        recorder = ExecutionRecorder()
        recorder.start_recording()
        listener = RecordingListener(recorder=recorder)

        # These keywords should be excluded
        excluded = ["start recording", "stop recording", "export test suite", "get recorded steps"]
        for kw in excluded:
            assert kw in listener._excluded_keywords

    def test_listener_excludes_ai_wrapper_keywords(self):
        """Test listener excludes Do, Check, Ask wrapper keywords.

        These are excluded because the AILibrary handles their recording
        via start_ai_step/end_ai_step. The actual executed sub-keywords
        are what should be recorded, not the wrapper keywords themselves.
        """
        recorder = ExecutionRecorder()
        recorder.start_recording()
        listener = RecordingListener(recorder=recorder)

        # AI wrapper keywords should be excluded
        ai_keywords = ["do", "check", "ask"]
        for kw in ai_keywords:
            assert kw in listener._excluded_keywords

    def test_listener_records_regular_keywords(self):
        """Test listener records regular RF keywords."""
        recorder = ExecutionRecorder()
        recorder.start_recording()
        listener = RecordingListener(recorder=recorder)

        # Simulate keyword execution
        data = MockKeywordData("Click", args=["#button"], libname="Browser")
        result = MockKeywordResult(passed=True)

        listener.start_keyword(data, None)
        listener.end_keyword(data, result)

        steps = recorder.get_steps()
        assert len(steps) == 1
        assert steps[0].keyword == "Click"
        assert steps[0].args == ["#button"]
        assert steps[0].library == "Browser"

    def test_listener_skips_excluded_keywords(self):
        """Test listener does not record excluded keywords."""
        recorder = ExecutionRecorder()
        recorder.start_recording()
        listener = RecordingListener(recorder=recorder)

        # Test excluded keywords
        excluded_keywords = ["Log", "Start Recording", "Do", "Check", "Ask", "Comment"]

        for kw_name in excluded_keywords:
            data = MockKeywordData(kw_name, args=["arg1"], libname="AILibrary")
            result = MockKeywordResult(passed=True)

            listener.start_keyword(data, None)
            listener.end_keyword(data, result)

        # None of the excluded keywords should be recorded
        steps = recorder.get_steps()
        assert len(steps) == 0

    def test_listener_records_sub_keywords_during_ai_step(self):
        """Test listener records sub-keywords that are executed during an AI step.

        When AILibrary calls start_ai_step, the recorder enters AI step mode.
        Keywords recorded during this time are added as sub-steps to the AI step.
        """
        recorder = ExecutionRecorder()
        recorder.start_recording()
        listener = RecordingListener(recorder=recorder)

        # Simulate AILibrary calling start_ai_step
        ai_step = recorder.start_ai_step(prompt="Login as user", ai_keyword="Do")

        # Simulate sub-keywords being executed (listener records these)
        keywords = [
            ("Fill Text", ["#username", "testuser"], "Browser"),
            ("Fill Text", ["#password", "secret"], "Browser"),
            ("Click", ["#login"], "Browser"),
        ]

        for kw_name, args, lib in keywords:
            data = MockKeywordData(kw_name, args=args, libname=lib)
            result = MockKeywordResult(passed=True)
            listener.start_keyword(data, None)
            listener.end_keyword(data, result)

        # End the AI step
        recorder.end_ai_step(success=True, attempts=1)

        # Verify: Only one AI step should be in main steps
        steps = recorder.get_steps()
        assert len(steps) == 1
        assert steps[0].step_type == StepType.AI
        assert steps[0].keyword == "Do"
        assert steps[0].prompt == "Login as user"

        # The sub-keywords should be inside the AI step
        assert len(steps[0].executed_keywords) == 3
        sub_kws = [s.keyword for s in steps[0].executed_keywords]
        assert sub_kws == ["Fill Text", "Fill Text", "Click"]

    def test_listener_records_mixed_regular_and_ai_steps(self):
        """Test recording session with both regular keywords and AI steps."""
        recorder = ExecutionRecorder()
        recorder.start_recording()
        listener = RecordingListener(recorder=recorder)

        # 1. Regular keyword (before AI)
        data1 = MockKeywordData("New Browser", args=["chromium"], libname="Browser")
        listener.start_keyword(data1, None)
        listener.end_keyword(data1, MockKeywordResult(passed=True))

        # 2. AI step: "Do" is handled by AILibrary, not the listener
        recorder.start_ai_step(prompt="Enter credentials", ai_keyword="Do")
        # Sub-keywords within the AI step
        fill1 = MockKeywordData("Fill Text", args=["#user", "admin"], libname="Browser")
        listener.start_keyword(fill1, None)
        listener.end_keyword(fill1, MockKeywordResult(passed=True))
        recorder.end_ai_step(success=True)

        # 3. Regular keyword (after AI)
        data2 = MockKeywordData("Take Screenshot", args=[], libname="Browser")
        listener.start_keyword(data2, None)
        listener.end_keyword(data2, MockKeywordResult(passed=True))

        recorder.stop_recording()

        steps = recorder.get_steps()
        # Should have: New Browser (regular), AI step (with 1 sub-step), Take Screenshot (regular)
        assert len(steps) == 3

        # Verify step types
        assert steps[0].keyword == "New Browser"
        assert steps[0].step_type == StepType.REGULAR

        assert steps[1].keyword == "Do"
        assert steps[1].step_type == StepType.AI
        assert len(steps[1].executed_keywords) == 1

        assert steps[2].keyword == "Take Screenshot"
        assert steps[2].step_type == StepType.REGULAR

    def test_listener_handles_failed_keywords(self):
        """Test listener records failure status correctly."""
        recorder = ExecutionRecorder()
        recorder.start_recording()
        listener = RecordingListener(recorder=recorder)

        data = MockKeywordData("Click", args=["#nonexistent"], libname="Browser")
        result = MockKeywordResult(passed=False, message="Element not found")

        listener.start_keyword(data, None)
        listener.end_keyword(data, result)

        steps = recorder.get_steps()
        assert len(steps) == 1
        assert steps[0].success is False
        assert steps[0].error == "Element not found"

    def test_listener_parses_kwargs(self):
        """Test listener correctly separates kwargs from args."""
        recorder = ExecutionRecorder()
        recorder.start_recording()
        listener = RecordingListener(recorder=recorder)

        data = MockKeywordData(
            "New Browser",
            args=["chromium", "headless=true", "slow_mo=100"],
            libname="Browser",
        )
        result = MockKeywordResult(passed=True)

        listener.start_keyword(data, None)
        listener.end_keyword(data, result)

        steps = recorder.get_steps()
        assert len(steps) == 1
        assert steps[0].args == ["chromium"]
        assert steps[0].kwargs == {"headless": "true", "slow_mo": "100"}

    def test_listener_infers_library_for_browser_keywords(self):
        """Test listener can infer Browser library for common keywords."""
        listener = RecordingListener()

        browser_keywords = ["new browser", "new page", "click", "fill text", "get text"]
        for kw in browser_keywords:
            lib = listener._infer_library(kw)
            assert lib == "Browser", f"Expected 'Browser' for {kw}, got {lib}"

    def test_listener_not_recording_when_inactive(self):
        """Test listener does not record when recorder is not active."""
        recorder = ExecutionRecorder()
        # NOT starting recording
        listener = RecordingListener(recorder=recorder)

        data = MockKeywordData("Click", args=["#button"], libname="Browser")
        result = MockKeywordResult(passed=True)

        listener.start_keyword(data, None)
        listener.end_keyword(data, result)

        steps = recorder.get_steps()
        assert len(steps) == 0

    def test_exclude_library(self):
        """Test excluding a library from recording."""
        recorder = ExecutionRecorder()
        recorder.start_recording()
        listener = RecordingListener(recorder=recorder)

        listener.exclude_library("BuiltIn")

        # Try recording a BuiltIn keyword
        data = MockKeywordData("Log", args=["Hello"], libname="BuiltIn")
        listener.start_keyword(data, None)
        listener.end_keyword(data, MockKeywordResult(passed=True))

        # Try recording a Browser keyword
        data2 = MockKeywordData("Click", args=["#btn"], libname="Browser")
        listener.start_keyword(data2, None)
        listener.end_keyword(data2, MockKeywordResult(passed=True))

        steps = recorder.get_steps()
        # Only Click should be recorded (Log is from excluded BuiltIn)
        # But wait - Log is also in excluded keywords! Let's use Set Variable instead
        assert len(steps) == 1
        assert steps[0].keyword == "Click"

    def test_include_keyword(self):
        """Test including a previously excluded keyword."""
        recorder = ExecutionRecorder()
        recorder.start_recording()
        listener = RecordingListener(recorder=recorder)

        # Verify log is excluded by default
        assert "log" in listener._excluded_keywords

        # Include log
        listener.include_keyword("log")
        assert "log" not in listener._excluded_keywords

    def test_listener_captures_variable_assignment_for_ask(self):
        """Test listener captures ${var}= Ask ... variable assignment."""
        recorder = ExecutionRecorder()
        recorder.start_recording()
        listener = RecordingListener(recorder=recorder)

        # Simulate: ${product_name}= Ask What is the name of the first product?
        ask_data = MockKeywordData(
            name="Ask",
            args=["What is the name of the first product?"],
            libname="AILibrary",
            assign=("${product_name}",),  # RF's assign attribute
        )

        # When listener sees Ask with assignment, it should store it on recorder
        listener.start_keyword(ask_data, None)

        # Check that the pending assignment was captured
        pending = recorder.get_and_clear_pending_assignment()
        assert pending is not None
        assert "${product_name}" in pending

    def test_listener_captures_variable_assignment_for_do(self):
        """Test listener captures variable assignment for Do keyword too."""
        recorder = ExecutionRecorder()
        recorder.start_recording()
        listener = RecordingListener(recorder=recorder)

        # Simulate: ${result}= Do Some action that returns value
        do_data = MockKeywordData(
            name="Do",
            args=["Some action"],
            libname="AILibrary",
            assign=("${result}",),
        )

        listener.start_keyword(do_data, None)

        pending = recorder.get_and_clear_pending_assignment()
        assert pending is not None
        assert "${result}" in pending

    def test_listener_clears_assignment_when_no_assign(self):
        """Test listener clears stale assignment when Ask has no assign."""
        recorder = ExecutionRecorder()
        recorder.start_recording()
        listener = RecordingListener(recorder=recorder)

        # First set a pending assignment manually
        recorder.set_pending_assignment(["${old_var}"])

        # Now simulate Ask WITHOUT assignment (clears stale assignment)
        ask_data = MockKeywordData(
            name="Ask",
            args=["What is the title?"],
            libname="AILibrary",
            assign=None,  # No assignment
        )

        listener.start_keyword(ask_data, None)

        # Pending assignment should now be None (cleared)
        pending = recorder.get_and_clear_pending_assignment()
        assert pending is None


class TestListenerVariableAssignmentIntegration:
    """Integration tests for variable assignment with complete workflow."""

    def test_ask_variable_assignment_end_to_end(self):
        """Test complete flow: ${var}= Ask ... -> Export with ${var}=."""
        recorder = ExecutionRecorder()
        listener = RecordingListener(recorder=recorder)
        recorder.start_recording()

        # 1. Listener sees Ask with assignment
        ask_data = MockKeywordData(
            name="Ask",
            args=["What is the product name?"],
            libname="AILibrary",
            assign=("${product_name}",),
        )
        listener.start_keyword(ask_data, None)

        # 2. AILibrary would call this to get and use the assignment
        pending = recorder.get_and_clear_pending_assignment()
        var_name = None
        if pending:
            var = pending[0]
            if var.startswith("${") and var.endswith("}"):
                var_name = var[2:-1]

        assert var_name == "product_name"

        # 3. AILibrary starts AI step with the variable
        recorder.start_ai_step(
            prompt="What is the product name?",
            ai_keyword="Ask",
            assigned_variable=var_name,
        )

        # 4. Sub-keyword executes (captured by listener)
        sub_data = MockKeywordData(
            name="Get Text",
            args=[".product-name >> nth=0"],
            libname="Browser",
        )
        listener.start_keyword(sub_data, None)
        listener.end_keyword(sub_data, MockKeywordResult(passed=True))

        # 5. AI step ends
        recorder.end_ai_step(success=True, result="Test Product", attempts=1)

        listener.end_keyword(ask_data, MockKeywordResult(passed=True))
        recorder.stop_recording()

        # 6. Verify the AI step has the variable assignment
        steps = recorder.get_steps()
        assert len(steps) == 1
        assert steps[0].assigned_variable == "product_name"
        assert steps[0].ai_keyword == "Ask"

        # 7. Export and verify variable assignment appears
        from robotmcp.lib.exporter import TestSuiteExporter
        import tempfile

        exporter = TestSuiteExporter(recorder)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".robot", delete=False) as f:
            exporter.export(f.name, format="robot", flatten_ai_only=True)
            with open(f.name) as rf:
                content = rf.read()

        # The exported file should have ${product_name}= before Get Text
        assert "${product_name}=" in content
        assert "Get Text" in content

    def test_multiple_ask_variables_end_to_end(self):
        """Test multiple Ask keywords with different variables."""
        recorder = ExecutionRecorder()
        listener = RecordingListener(recorder=recorder)
        recorder.start_recording()

        asks = [
            ("${name}", "What is the name?", ".name"),
            ("${price}", "What is the price?", ".price"),
            ("${desc}", "What is the description?", ".description"),
        ]

        for var, prompt, selector in asks:
            # Listener sees Ask
            ask_data = MockKeywordData(
                name="Ask", args=[prompt], libname="AILibrary", assign=(var,)
            )
            listener.start_keyword(ask_data, None)

            # Get and use assignment
            pending = recorder.get_and_clear_pending_assignment()
            var_name = pending[0][2:-1] if pending else None

            # Start AI step
            recorder.start_ai_step(prompt, "Ask", assigned_variable=var_name)

            # Sub-keyword
            sub_data = MockKeywordData("Get Text", args=[selector], libname="Browser")
            listener.start_keyword(sub_data, None)
            listener.end_keyword(sub_data, MockKeywordResult(passed=True))

            recorder.end_ai_step(success=True, attempts=1)
            listener.end_keyword(ask_data, MockKeywordResult(passed=True))

        recorder.stop_recording()

        # Verify all steps have their variables
        steps = recorder.get_steps()
        assert len(steps) == 3
        assert steps[0].assigned_variable == "name"
        assert steps[1].assigned_variable == "price"
        assert steps[2].assigned_variable == "desc"

        # Export and verify
        from robotmcp.lib.exporter import TestSuiteExporter
        import tempfile

        exporter = TestSuiteExporter(recorder)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".robot", delete=False) as f:
            exporter.export(f.name, format="robot", flatten_ai_only=True)
            with open(f.name) as rf:
                content = rf.read()

        assert "${name}=" in content
        assert "${price}=" in content
        assert "${desc}=" in content


class TestListenerWithAILibraryIntegration:
    """Tests for listener integration with AILibrary workflow."""

    def test_complete_ai_workflow_recording(self):
        """Test complete workflow: Start Recording -> Do -> Check -> Ask -> Export.

        This simulates what happens in a real Robot Framework test when:
        1. User calls Start Recording
        2. User calls Do "Login as user"
        3. AILibrary translates to Fill Text + Click
        4. User calls Check "User is logged in"
        5. User calls Export Test Suite
        """
        recorder = ExecutionRecorder()
        listener = RecordingListener(recorder=recorder)

        # 1. Start Recording (AILibrary calls recorder.start_recording())
        recorder.start_recording()

        # 2. Browser setup (regular keywords caught by listener)
        setup_keywords = [
            ("New Browser", ["chromium"], "Browser"),
            ("New Page", ["https://example.com"], "Browser"),
        ]
        for kw, args, lib in setup_keywords:
            data = MockKeywordData(kw, args=args, libname=lib)
            listener.start_keyword(data, None)
            listener.end_keyword(data, MockKeywordResult(passed=True))

        # 3. Do "Login as user" - AILibrary handles this:
        #    a) Listener sees "Do" keyword but EXCLUDES it
        #    b) AILibrary calls recorder.start_ai_step()
        #    c) AI generates Fill Text + Click
        #    d) Listener captures these sub-keywords
        #    e) AILibrary calls recorder.end_ai_step()

        # Simulate the "Do" keyword being called (should be excluded)
        do_data = MockKeywordData("Do", args=["Login as user"], libname="AILibrary")
        listener.start_keyword(do_data, None)  # Should push None to stack

        # AILibrary starts AI step
        recorder.start_ai_step("Login as user", "Do")

        # AI-generated sub-keywords (listener records these into AI step)
        ai_sub_keywords = [
            ("Fill Text", ["#username", "testuser"], "Browser"),
            ("Fill Text", ["#password", "secret"], "Browser"),
            ("Click", ["#login-btn"], "Browser"),
        ]
        for kw, args, lib in ai_sub_keywords:
            data = MockKeywordData(kw, args=args, libname=lib)
            listener.start_keyword(data, None)
            listener.end_keyword(data, MockKeywordResult(passed=True))

        # AILibrary ends AI step
        recorder.end_ai_step(success=True, attempts=1)

        listener.end_keyword(do_data, MockKeywordResult(passed=True))  # Pops None

        # 4. Check "User is logged in"
        check_data = MockKeywordData("Check", args=["User is logged in"], libname="AILibrary")
        listener.start_keyword(check_data, None)

        recorder.start_ai_step("User is logged in", "Check")

        check_sub = MockKeywordData("Get Text", args=[".welcome-message"], libname="Browser")
        listener.start_keyword(check_sub, None)
        listener.end_keyword(check_sub, MockKeywordResult(passed=True))

        recorder.end_ai_step(success=True, attempts=1)

        listener.end_keyword(check_data, MockKeywordResult(passed=True))

        # 5. Stop recording
        recorder.stop_recording()

        # Verify the recorded steps
        steps = recorder.get_steps()

        # Expected structure:
        # [0] New Browser (regular)
        # [1] New Page (regular)
        # [2] AI step "Do" with 3 sub-keywords
        # [3] AI step "Check" with 1 sub-keyword
        assert len(steps) == 4

        assert steps[0].keyword == "New Browser"
        assert steps[0].step_type == StepType.REGULAR

        assert steps[1].keyword == "New Page"
        assert steps[1].step_type == StepType.REGULAR

        assert steps[2].keyword == "Do"
        assert steps[2].step_type == StepType.AI
        assert steps[2].prompt == "Login as user"
        assert len(steps[2].executed_keywords) == 3

        assert steps[3].keyword == "Check"
        assert steps[3].step_type == StepType.AI
        assert steps[3].prompt == "User is logged in"
        assert len(steps[3].executed_keywords) == 1

        # Verify sub-keywords are correctly captured
        login_subs = [s.keyword for s in steps[2].executed_keywords]
        assert login_subs == ["Fill Text", "Fill Text", "Click"]

        # The exported test should contain the actual keywords, not Do/Check/Ask
        from robotmcp.lib.exporter import TestSuiteExporter
        import tempfile

        exporter = TestSuiteExporter(recorder)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".robot", delete=False) as f:
            exporter.export(f.name, format="robot", flatten_ai_only=True)
            with open(f.name) as rf:
                content = rf.read()

        # The exported file should contain actual keywords
        assert "Fill Text" in content
        assert "Click" in content
        assert "Get Text" in content
        # AI wrapper keywords should appear only in comments
        lines = content.split("\n")
        non_comment_lines = [l for l in lines if not l.strip().startswith("#")]
        non_comment_content = "\n".join(non_comment_lines)
        # Do, Check, Ask should not be in the actual keywords (only in comments)
        # We need to be careful - "Documentation" contains "Do"
        keyword_lines = [l for l in non_comment_lines if l.strip() and not l.strip().startswith("[")]
        for line in keyword_lines:
            # Skip lines that are just keyword calls
            parts = line.strip().split()
            if parts:
                first_word = parts[0]
                # The first word should not be "Do", "Check", or "Ask" as standalone keywords
                if first_word in ("Do", "Check", "Ask"):
                    pytest.fail(f"Found AI wrapper keyword in export: {line}")
