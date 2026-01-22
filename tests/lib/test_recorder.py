"""Tests for recorder module."""

import pytest

from robotmcp.lib.recorder import ExecutionRecorder, RecordedStep, StepType


class TestRecordedStep:
    """Tests for RecordedStep class."""

    def test_basic_step(self):
        """Test creating a basic recorded step."""
        step = RecordedStep(
            keyword="Click",
            args=["button#submit"],
            library="Browser",
        )

        assert step.keyword == "Click"
        assert step.args == ["button#submit"]
        assert step.library == "Browser"
        assert step.step_type == StepType.REGULAR
        assert step.success is True

    def test_ai_step(self):
        """Test creating an AI step."""
        step = RecordedStep(
            keyword="Do",
            args=["Login as user1"],
            step_type=StepType.AI,
            prompt="Login as user1",
            ai_keyword="Do",
        )

        assert step.step_type == StepType.AI
        assert step.prompt == "Login as user1"
        assert step.ai_keyword == "Do"

    def test_to_dict_regular(self):
        """Test converting regular step to dict."""
        step = RecordedStep(
            keyword="Fill Text",
            args=["id=username", "testuser"],
            library="Browser",
        )

        data = step.to_dict()

        assert data["keyword"] == "Fill Text"
        assert data["args"] == ["id=username", "testuser"]
        assert data["library"] == "Browser"
        assert data["type"] == "regular"
        assert data["success"] is True

    def test_to_dict_ai(self):
        """Test converting AI step to dict."""
        sub_step = RecordedStep(
            keyword="Fill Text",
            args=["id=username", "user1"],
            library="Browser",
        )

        step = RecordedStep(
            keyword="Do",
            args=["Login as user1"],
            step_type=StepType.AI,
            prompt="Login as user1",
            ai_keyword="Do",
            executed_keywords=[sub_step],
            attempts=2,
        )

        data = step.to_dict()

        assert data["type"] == "ai"
        assert data["prompt"] == "Login as user1"
        assert data["ai_keyword"] == "Do"
        assert data["attempts"] == 2
        assert len(data["executed_keywords"]) == 1


class TestExecutionRecorder:
    """Tests for ExecutionRecorder class."""

    def test_start_stop_recording(self):
        """Test starting and stopping recording."""
        recorder = ExecutionRecorder()

        assert not recorder.is_recording

        recorder.start_recording()
        assert recorder.is_recording

        recorder.stop_recording()
        assert not recorder.is_recording

    def test_record_step(self):
        """Test recording a step."""
        recorder = ExecutionRecorder()
        recorder.start_recording()

        step = recorder.record_step(
            keyword="Click",
            args=["button"],
            library="Browser",
        )

        assert step is not None
        assert step.keyword == "Click"

        steps = recorder.get_steps()
        assert len(steps) == 1

        recorder.stop_recording()

    def test_record_step_not_recording(self):
        """Test that recording is skipped when not active."""
        recorder = ExecutionRecorder()

        step = recorder.record_step(
            keyword="Click",
            args=["button"],
        )

        assert step is None
        assert len(recorder.get_steps()) == 0

    def test_ai_step_recording(self):
        """Test recording AI steps with sub-steps."""
        recorder = ExecutionRecorder()
        recorder.start_recording()

        # Start AI step
        ai_step = recorder.start_ai_step(
            prompt="Login as user1",
            ai_keyword="Do",
        )

        assert ai_step is not None
        assert recorder._current_ai_step is not None

        # Record sub-steps (these go into the AI step)
        recorder.record_step(
            keyword="Fill Text",
            args=["id=username", "user1"],
            library="Browser",
        )
        recorder.record_step(
            keyword="Click",
            args=["id=login"],
            library="Browser",
        )

        # End AI step
        completed = recorder.end_ai_step(success=True, attempts=1)

        assert completed is not None
        assert len(completed.executed_keywords) == 2
        assert recorder._current_ai_step is None

        # Check that AI step was added to main steps
        steps = recorder.get_steps()
        assert len(steps) == 1
        assert steps[0].step_type == StepType.AI

        recorder.stop_recording()

    def test_libraries_tracking(self):
        """Test tracking of used libraries."""
        recorder = ExecutionRecorder()
        recorder.start_recording()

        recorder.record_step(keyword="Click", library="Browser")
        recorder.record_step(keyword="Click Element", library="SeleniumLibrary")
        recorder.record_step(keyword="Log", library="BuiltIn")
        recorder.record_step(keyword="Click", library="Browser")  # Duplicate

        libs = recorder.get_libraries_used()
        assert len(libs) == 3
        assert "Browser" in libs
        assert "SeleniumLibrary" in libs
        assert "BuiltIn" in libs

        recorder.stop_recording()

    def test_metadata(self):
        """Test recording metadata."""
        recorder = ExecutionRecorder()
        recorder.start_recording()

        recorder.record_step(keyword="Click", library="Browser")
        recorder.record_step(keyword="Fill Text", library="Browser")

        metadata = recorder.get_metadata()
        assert "started_at" in metadata
        assert metadata["step_count"] == 2
        assert "Browser" in metadata["libraries"]

        recorder.stop_recording()
        assert "stopped_at" in recorder.get_metadata()

    def test_clear(self):
        """Test clearing recorded steps."""
        recorder = ExecutionRecorder()
        recorder.start_recording()

        recorder.record_step(keyword="Click", library="Browser")
        assert len(recorder.get_steps()) == 1

        recorder.clear()
        assert len(recorder.get_steps()) == 0
        assert len(recorder.get_libraries_used()) == 0

    def test_to_dict(self):
        """Test converting recording to dict."""
        recorder = ExecutionRecorder()
        recorder.start_recording()

        recorder.record_step(keyword="Click", library="Browser")

        data = recorder.to_dict()
        assert "metadata" in data
        assert "steps" in data
        assert len(data["steps"]) == 1

        recorder.stop_recording()

    def test_ai_step_with_assigned_variable(self):
        """Test recording AI step with variable assignment (for Ask keyword)."""
        recorder = ExecutionRecorder()
        recorder.start_recording()

        # Start Ask AI step with variable assignment
        ai_step = recorder.start_ai_step(
            prompt="What is the name of the first product?",
            ai_keyword="Ask",
            assigned_variable="product_name",
        )

        assert ai_step is not None
        assert ai_step.assigned_variable == "product_name"

        # Record the executed keyword
        recorder.record_step(
            keyword="Get Text",
            args=[".inventory_item_name >> nth=0"],
            library="Browser",
        )

        # End AI step with result
        completed = recorder.end_ai_step(
            success=True, result="Sauce Labs Backpack", attempts=1
        )

        assert completed.assigned_variable == "product_name"
        assert completed.result == "Sauce Labs Backpack"

        # Check that it's in the steps
        steps = recorder.get_steps()
        assert len(steps) == 1
        assert steps[0].assigned_variable == "product_name"

        recorder.stop_recording()

    def test_ai_step_assigned_variable_in_to_dict(self):
        """Test that assigned_variable is included in to_dict output."""
        step = RecordedStep(
            keyword="Ask",
            args=["What is the price?"],
            step_type=StepType.AI,
            prompt="What is the price?",
            ai_keyword="Ask",
            assigned_variable="price",
        )

        data = step.to_dict()

        assert data["type"] == "ai"
        assert data["assigned_variable"] == "price"

    def test_ai_step_no_assigned_variable_not_in_dict(self):
        """Test that assigned_variable is not in dict when not set."""
        step = RecordedStep(
            keyword="Do",
            args=["Click the button"],
            step_type=StepType.AI,
            prompt="Click the button",
            ai_keyword="Do",
        )

        data = step.to_dict()

        assert data["type"] == "ai"
        assert "assigned_variable" not in data
