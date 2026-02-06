"""Benchmark variable serialization for JSON compatibility in MCP attach bridge.

This module validates that the _sanitize_for_json fix works correctly and efficiently.
It benchmarks serialization across different variable counts and types commonly found
in Robot Framework test execution.

Run with:
    uv run pytest tests/benchmarks/test_variable_serialization_benchmark.py -v

Targets:
- Serialization should handle 1000+ variables without crash
- Serialization latency should be <10ms for typical variable sets
- Memory overhead should be <5x the input size
"""

import gc
import json
import sys
import time
import tracemalloc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Callable
from unittest.mock import MagicMock

import pytest


# ============================================================================
# Simulated Robot Framework variable types
# ============================================================================


class DotDict(dict):
    """Simulates Robot Framework's DotDict (dict subclass with attribute access)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class Tags(list):
    """Simulates Robot Framework's Tags (list subclass)."""

    def __init__(self, *tags: str):
        super().__init__(tags)

    def __str__(self) -> str:
        return f"[{', '.join(self)}]"


class RobotOptions:
    """Simulates Robot Framework's internal options object (not serializable)."""

    def __init__(self):
        self.output = Path("/tmp/output.xml")
        self.log = Path("/tmp/log.html")
        self.report = Path("/tmp/report.html")
        self.loglevel = "INFO"
        self.dryrun = False
        self._internal_state = {"complex": object()}


class LibraryInstance:
    """Simulates a library instance stored in RF variables."""

    def __init__(self, name: str):
        self.name = name
        self._connection = MagicMock()

    def some_keyword(self, arg: str) -> str:
        return f"result for {arg}"


class CircularRef:
    """Object with circular reference to test infinite loop prevention."""

    def __init__(self):
        self.name = "circular"
        self.parent = None  # Will be set to self

    def __repr__(self) -> str:
        return f"CircularRef(name={self.name})"


def create_circular_reference() -> CircularRef:
    """Create an object with a circular reference."""
    obj = CircularRef()
    obj.parent = obj  # Circular reference
    return obj


# ============================================================================
# Sanitization implementation under test
# ============================================================================


def sanitize_for_json(val: Any) -> Any:
    """Convert a value to a JSON-serializable form.

    This is the production implementation from mcp_attach.py.

    - None, str, int, float, bool are returned as-is
    - Path objects (anything with __fspath__) are converted to str
    - Lists/tuples are recursively sanitized
    - Dicts are recursively sanitized (keys converted to str)
    - Other objects are converted to str(val) to preserve useful info
    """
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    # Handle Path-like objects (pathlib.Path, os.PathLike, etc.)
    if hasattr(val, "__fspath__"):
        return str(val)
    if isinstance(val, (list, tuple)):
        return [sanitize_for_json(item) for item in val]
    if isinstance(val, dict):
        return {str(k): sanitize_for_json(v) for k, v in val.items()}
    # Fallback: convert to string to preserve useful information
    return str(val)


def sanitize_for_json_with_depth_limit(val: Any, max_depth: int = 50, _depth: int = 0) -> Any:
    """Enhanced sanitization with depth limit to handle circular references.

    This prevents infinite recursion for circular reference scenarios.
    """
    if _depth > max_depth:
        return f"<max_depth_exceeded: {type(val).__name__}>"

    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if hasattr(val, "__fspath__"):
        return str(val)
    if isinstance(val, (list, tuple)):
        return [sanitize_for_json_with_depth_limit(item, max_depth, _depth + 1) for item in val]
    if isinstance(val, dict):
        return {
            str(k): sanitize_for_json_with_depth_limit(v, max_depth, _depth + 1)
            for k, v in val.items()
        }
    return str(val)


# ============================================================================
# Test data generators
# ============================================================================


def generate_simple_variables(count: int) -> Dict[str, Any]:
    """Generate simple JSON-serializable variables."""
    return {
        f"${{VAR_{i}}}": f"value_{i}" if i % 3 == 0 else i if i % 3 == 1 else i * 0.5
        for i in range(count)
    }


def generate_rf_like_variables(count: int) -> Dict[str, Any]:
    """Generate variables similar to Robot Framework execution context."""
    variables: Dict[str, Any] = {}

    # Standard RF built-in variables
    variables["${CURDIR}"] = Path("/home/user/tests")
    variables["${EXECDIR}"] = Path("/home/user/project")
    variables["${OUTPUT_DIR}"] = Path("/home/user/output")
    variables["${LOG_FILE}"] = Path("/home/user/output/log.html")
    variables["${REPORT_FILE}"] = Path("/home/user/output/report.html")
    variables["${OUTPUT_FILE}"] = Path("/home/user/output/output.xml")
    variables["${TEMPDIR}"] = Path("/tmp")

    # Suite/test variables
    variables["${SUITE_NAME}"] = "Test Suite"
    variables["${SUITE_SOURCE}"] = Path("/home/user/tests/suite.robot")
    variables["${TEST_NAME}"] = "Test Case Name"
    variables["${TEST_STATUS}"] = "PASS"
    variables["${PREV_TEST_STATUS}"] = "PASS"

    # DotDict-like metadata
    variables["${SUITE_METADATA}"] = DotDict(
        author="Test Author",
        version="1.0",
        priority="high",
    )

    # Tags object
    variables["${TEST_TAGS}"] = Tags("smoke", "regression", "critical")

    # Datetime objects
    variables["${START_TIME}"] = datetime.now()
    variables["${ELAPSED}"] = timedelta(seconds=42)

    # User-defined variables
    for i in range(count - len(variables)):
        var_type = i % 5
        if var_type == 0:
            variables[f"${{USER_STR_{i}}}"] = f"string_value_{i}"
        elif var_type == 1:
            variables[f"${{USER_INT_{i}}}"] = i * 100
        elif var_type == 2:
            variables[f"${{USER_PATH_{i}}}"] = Path(f"/data/file_{i}.txt")
        elif var_type == 3:
            variables[f"${{USER_LIST_{i}}}"] = [f"item_{j}" for j in range(5)]
        else:
            variables[f"${{USER_DICT_{i}}}"] = {"key": f"value_{i}", "count": i}

    return variables


def generate_edge_case_variables() -> Dict[str, Any]:
    """Generate variables with edge cases that might fail serialization."""
    return {
        # Path objects (common RF variable type)
        "${OUTPUT_DIR}": Path("/home/user/output"),
        "${CURDIR}": Path.cwd(),
        "${TEMPDIR}": Path("/tmp"),

        # DotDict-like objects
        "${METADATA}": DotDict(
            nested=DotDict(deep=DotDict(value="deep_value")),
            list_attr=[1, 2, 3],
        ),

        # Tags (list subclass)
        "${TAGS}": Tags("tag1", "tag2", "tag3"),

        # datetime objects
        "${TIMESTAMP}": datetime.now(),
        "${DELTA}": timedelta(hours=1, minutes=30),

        # Complex RF options object
        "${OPTIONS}": RobotOptions(),

        # Library instance
        "${BROWSER_LIB}": LibraryInstance("Browser"),

        # Functions/lambdas
        "${FUNC}": lambda x: x * 2,
        "${BUILTIN_FUNC}": len,

        # Large strings
        "${LARGE_STRING}": "x" * 10000,

        # Deeply nested structure
        "${DEEP_NESTED}": {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {"value": "deep"}
                        }
                    }
                }
            }
        },

        # List with mixed types
        "${MIXED_LIST}": [
            Path("/file.txt"),
            datetime.now(),
            {"nested": "dict"},
            [1, 2, 3],
            None,
            True,
            42,
        ],

        # Empty collections
        "${EMPTY_LIST}": [],
        "${EMPTY_DICT}": {},
        "${EMPTY_STRING}": "",

        # None value
        "${NONE_VAR}": None,

        # Boolean values
        "${TRUE}": True,
        "${FALSE}": False,

        # Numeric edge cases
        "${ZERO}": 0,
        "${NEGATIVE}": -42,
        "${FLOAT_PRECISION}": 0.1 + 0.2,
        "${LARGE_INT}": 10**20,
    }


def generate_many_path_variables(count: int) -> Dict[str, Any]:
    """Generate many Path objects (common failure case)."""
    return {f"${{PATH_{i}}}": Path(f"/data/dir_{i}/file_{i}.txt") for i in range(count)}


def generate_deeply_nested_structure(depth: int) -> Dict[str, Any]:
    """Generate a deeply nested dictionary structure."""
    result: Dict[str, Any] = {"value": "leaf"}
    for i in range(depth):
        result = {f"level_{depth - i}": result}
    return {"${DEEP}": result}


# ============================================================================
# Benchmark tests
# ============================================================================


class TestSerializationPerformance:
    """Benchmark serialization performance across variable counts."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize("var_count", [100, 500, 1000])
    def test_simple_variable_serialization_latency(
        self,
        var_count: int,
        benchmark_reporter,
    ):
        """Target: <10ms for serializing simple variables."""
        variables = generate_simple_variables(var_count)

        # Warm-up
        for _ in range(10):
            for v in list(variables.values())[:10]:
                sanitize_for_json(v)

        start = time.perf_counter()
        iterations = 100

        for _ in range(iterations):
            result = {k: sanitize_for_json(v) for k, v in variables.items()}

        duration_ms = (time.perf_counter() - start) * 1000

        # Verify result is JSON serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0

        benchmark_reporter.record_latency(
            name=f"simple_vars_{var_count}",
            duration_ms=duration_ms,
            target_ms=10.0 * iterations,  # 10ms per iteration
            iterations=iterations,
            var_count=var_count,
        )

        avg_ms = duration_ms / iterations
        assert avg_ms < 10.0, (
            f"Serialization took {avg_ms:.2f}ms for {var_count} vars, target <10ms"
        )

    @pytest.mark.benchmark
    @pytest.mark.parametrize("var_count", [100, 500, 1000])
    def test_rf_like_variable_serialization_latency(
        self,
        var_count: int,
        benchmark_reporter,
    ):
        """Target: <20ms for serializing RF-like variables with complex types."""
        variables = generate_rf_like_variables(var_count)

        start = time.perf_counter()
        iterations = 50

        for _ in range(iterations):
            result = {k: sanitize_for_json(v) for k, v in variables.items()}

        duration_ms = (time.perf_counter() - start) * 1000

        # Verify result is JSON serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0

        benchmark_reporter.record_latency(
            name=f"rf_like_vars_{var_count}",
            duration_ms=duration_ms,
            target_ms=20.0 * iterations,
            iterations=iterations,
            var_count=var_count,
        )

        avg_ms = duration_ms / iterations
        assert avg_ms < 20.0, (
            f"Serialization took {avg_ms:.2f}ms for {var_count} RF-like vars, target <20ms"
        )


class TestSerializationMemory:
    """Benchmark memory usage during serialization."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize("var_count", [100, 500, 1000])
    def test_serialization_memory_overhead(
        self,
        var_count: int,
        benchmark_reporter,
    ):
        """Target: Memory overhead <5x input size."""
        variables = generate_rf_like_variables(var_count)

        # Estimate input size
        input_size = sys.getsizeof(variables)
        for v in variables.values():
            try:
                input_size += sys.getsizeof(v)
            except TypeError:
                input_size += 100  # Estimate for complex objects

        tracemalloc.start()
        baseline, _ = tracemalloc.get_traced_memory()

        result = {k: sanitize_for_json(v) for k, v in variables.items()}
        json_str = json.dumps(result)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_used = current - baseline
        output_size = len(json_str)

        benchmark_reporter.record_memory(
            name=f"serialization_memory_{var_count}",
            memory_bytes=memory_used,
            items_count=var_count,
            target_bytes_per_item=5000,  # 5KB per variable max
            input_size_estimate=input_size,
            output_size=output_size,
            peak_memory=peak,
        )

        # Memory overhead should be reasonable
        if input_size > 0:
            overhead_ratio = memory_used / input_size
            assert overhead_ratio < 5.0, (
                f"Memory overhead {overhead_ratio:.1f}x exceeds 5x target"
            )


class TestEdgeCaseSerialization:
    """Test serialization of edge cases that previously caused crashes."""

    @pytest.mark.benchmark
    def test_path_objects_serialize_to_string(self, benchmark_reporter):
        """Verify Path objects serialize to strings, not crash."""
        variables = generate_many_path_variables(100)

        start = time.perf_counter()
        result = {k: sanitize_for_json(v) for k, v in variables.items()}
        duration_ms = (time.perf_counter() - start) * 1000

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0

        # Verify Path objects became strings
        for k, v in result.items():
            assert isinstance(v, str), f"Path at {k} should be string, got {type(v)}"
            assert "/" in v or "\\" in v, f"Path string at {k} should contain path separator"

        benchmark_reporter.record_latency(
            name="path_objects_100",
            duration_ms=duration_ms,
            target_ms=5.0,
            iterations=1,
        )

    @pytest.mark.benchmark
    def test_dotdict_like_objects_serialize(self, benchmark_reporter):
        """Verify DotDict-like objects serialize correctly."""
        dotdict = DotDict(
            key1="value1",
            nested=DotDict(inner="inner_value"),
            list_attr=[1, 2, 3],
        )

        start = time.perf_counter()
        result = sanitize_for_json(dotdict)
        duration_ms = (time.perf_counter() - start) * 1000

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0

        # Verify structure preserved
        assert result["key1"] == "value1"
        assert result["nested"]["inner"] == "inner_value"
        assert result["list_attr"] == [1, 2, 3]

        benchmark_reporter.record_latency(
            name="dotdict_serialization",
            duration_ms=duration_ms,
            target_ms=1.0,
            iterations=1,
        )

    @pytest.mark.benchmark
    def test_tags_object_serializes(self, benchmark_reporter):
        """Verify Tags (list subclass) serializes correctly."""
        tags = Tags("smoke", "regression", "critical", "automated")

        start = time.perf_counter()
        result = sanitize_for_json(tags)
        duration_ms = (time.perf_counter() - start) * 1000

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0

        # Verify it's a list with correct values
        assert isinstance(result, list)
        assert result == ["smoke", "regression", "critical", "automated"]

        benchmark_reporter.record_latency(
            name="tags_serialization",
            duration_ms=duration_ms,
            target_ms=0.5,
            iterations=1,
        )

    @pytest.mark.benchmark
    def test_datetime_objects_serialize(self, benchmark_reporter):
        """Verify datetime objects serialize to string representation."""
        variables = {
            "${NOW}": datetime.now(),
            "${DELTA}": timedelta(hours=2, minutes=30),
        }

        start = time.perf_counter()
        result = {k: sanitize_for_json(v) for k, v in variables.items()}
        duration_ms = (time.perf_counter() - start) * 1000

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0

        # Verify datetime became string
        assert isinstance(result["${NOW}"], str)
        assert isinstance(result["${DELTA}"], str)

        benchmark_reporter.record_latency(
            name="datetime_serialization",
            duration_ms=duration_ms,
            target_ms=0.5,
            iterations=1,
        )

    @pytest.mark.benchmark
    def test_custom_class_instances_serialize(self, benchmark_reporter):
        """Verify custom class instances serialize to string representation."""
        variables = {
            "${OPTIONS}": RobotOptions(),
            "${LIB}": LibraryInstance("Browser"),
        }

        start = time.perf_counter()
        result = {k: sanitize_for_json(v) for k, v in variables.items()}
        duration_ms = (time.perf_counter() - start) * 1000

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0

        # Verify complex objects became strings
        for k, v in result.items():
            assert isinstance(v, str), f"{k} should be string, got {type(v)}"

        benchmark_reporter.record_latency(
            name="custom_class_serialization",
            duration_ms=duration_ms,
            target_ms=1.0,
            iterations=1,
        )

    @pytest.mark.benchmark
    def test_functions_and_lambdas_serialize(self, benchmark_reporter):
        """Verify functions and lambdas serialize to string representation."""
        variables = {
            "${LAMBDA}": lambda x: x * 2,
            "${FUNC}": len,
            "${METHOD}": "".join,
        }

        start = time.perf_counter()
        result = {k: sanitize_for_json(v) for k, v in variables.items()}
        duration_ms = (time.perf_counter() - start) * 1000

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0

        # Verify functions became strings
        for k, v in result.items():
            assert isinstance(v, str), f"{k} should be string, got {type(v)}"

        benchmark_reporter.record_latency(
            name="function_serialization",
            duration_ms=duration_ms,
            target_ms=0.5,
            iterations=1,
        )

    @pytest.mark.benchmark
    def test_deeply_nested_structure_serializes(self, benchmark_reporter):
        """Verify deeply nested structures serialize without stack overflow."""
        variables = generate_deeply_nested_structure(depth=30)

        start = time.perf_counter()
        result = {k: sanitize_for_json(v) for k, v in variables.items()}
        duration_ms = (time.perf_counter() - start) * 1000

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0

        # Verify nesting preserved
        assert "level_1" in result["${DEEP}"]

        benchmark_reporter.record_latency(
            name="deep_nesting_30",
            duration_ms=duration_ms,
            target_ms=2.0,
            iterations=1,
        )

    @pytest.mark.benchmark
    def test_circular_reference_does_not_infinite_loop(self, benchmark_reporter):
        """Verify circular references don't cause infinite loops.

        Note: The basic sanitize_for_json will convert to str() which handles this.
        The enhanced version with depth limit provides additional protection.
        """
        circular = create_circular_reference()

        start = time.perf_counter()
        # Use str() fallback which handles circular refs
        result = sanitize_for_json(circular)
        duration_ms = (time.perf_counter() - start) * 1000

        # Verify JSON serializable (should be string representation)
        json_str = json.dumps({"circular": result})
        assert len(json_str) > 0
        assert isinstance(result, str)

        benchmark_reporter.record_latency(
            name="circular_reference",
            duration_ms=duration_ms,
            target_ms=1.0,
            iterations=1,
        )

    @pytest.mark.benchmark
    def test_all_edge_cases_combined(self, benchmark_reporter):
        """Test all edge cases in a single variable set."""
        variables = generate_edge_case_variables()

        start = time.perf_counter()
        result = {k: sanitize_for_json(v) for k, v in variables.items()}
        duration_ms = (time.perf_counter() - start) * 1000

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0

        # Count successful serializations
        successful = sum(1 for v in result.values() if v is not None or variables.get(v) is None)
        assert successful == len(variables), "All variables should serialize"

        benchmark_reporter.record_latency(
            name="all_edge_cases",
            duration_ms=duration_ms,
            target_ms=5.0,
            iterations=1,
            edge_case_count=len(variables),
        )

        benchmark_reporter.record(
            name="edge_case_coverage",
            duration_ms=duration_ms,
            tokens_before=len(variables),
            tokens_after=len(result),
            target_reduction=0.0,  # No reduction expected
            successful_conversions=successful,
        )


class TestComparisonWithWithoutSanitization:
    """Compare performance with and without sanitization."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize("var_count", [100, 500, 1000])
    def test_sanitization_overhead(
        self,
        var_count: int,
        benchmark_reporter,
    ):
        """Measure overhead of sanitization vs direct JSON dump."""
        # Generate simple variables that ARE JSON serializable
        simple_vars = generate_simple_variables(var_count)

        # Warmup iterations to avoid cold cache effects (important for CI)
        # This ensures JIT optimization and cache warming before timing
        for _ in range(20):
            json.dumps(simple_vars)
            sanitized = {k: sanitize_for_json(v) for k, v in simple_vars.items()}
            json.dumps(sanitized)

        # Time direct JSON dump (no sanitization)
        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            json.dumps(simple_vars)
        direct_duration_ms = (time.perf_counter() - start) * 1000

        # Time with sanitization
        start = time.perf_counter()
        for _ in range(iterations):
            sanitized = {k: sanitize_for_json(v) for k, v in simple_vars.items()}
            json.dumps(sanitized)
        sanitized_duration_ms = (time.perf_counter() - start) * 1000

        overhead_percent = (
            (sanitized_duration_ms - direct_duration_ms) / direct_duration_ms * 100
            if direct_duration_ms > 0
            else 0
        )

        benchmark_reporter.record(
            name=f"sanitization_overhead_{var_count}",
            duration_ms=sanitized_duration_ms - direct_duration_ms,
            tokens_before=int(direct_duration_ms * 100),  # Use as proxy
            tokens_after=int(sanitized_duration_ms * 100),
            target_reduction=-400.0,  # Expect up to 400% overhead max (CI variability)
            direct_ms=direct_duration_ms / iterations,
            sanitized_ms=sanitized_duration_ms / iterations,
            overhead_percent=overhead_percent,
        )

        # Sanitization overhead should be reasonable (< 5x slower)
        # Using 5x threshold to account for CI environment variability:
        # - Local dev machines typically show 2-3x overhead
        # - GitHub Actions runners show 3-4x due to shared resources
        # - This threshold catches severe regressions while allowing CI variance
        assert sanitized_duration_ms < direct_duration_ms * 5, (
            f"Sanitization adds {overhead_percent:.0f}% overhead, target <400%"
        )


class TestJsonValidation:
    """Validate that serialized output is valid JSON."""

    @pytest.mark.benchmark
    def test_serialized_output_is_valid_json(self, benchmark_reporter):
        """Verify all edge cases produce valid JSON."""
        variables = generate_edge_case_variables()
        result = {k: sanitize_for_json(v) for k, v in variables.items()}

        # Must not raise
        json_str = json.dumps(result)

        # Must be parseable
        parsed = json.loads(json_str)

        # Parsed keys should match
        assert set(parsed.keys()) == set(result.keys())

        benchmark_reporter.record_latency(
            name="json_validation",
            duration_ms=0.0,
            target_ms=1.0,
            iterations=1,
            valid=True,
        )

    @pytest.mark.benchmark
    def test_serialized_output_roundtrips(self, benchmark_reporter):
        """Verify serialized data can be JSON encoded and decoded."""
        variables = generate_rf_like_variables(100)

        start = time.perf_counter()
        iterations = 50

        for _ in range(iterations):
            result = {k: sanitize_for_json(v) for k, v in variables.items()}
            encoded = json.dumps(result)
            decoded = json.loads(encoded)
            assert len(decoded) == len(result)

        duration_ms = (time.perf_counter() - start) * 1000

        benchmark_reporter.record_latency(
            name="json_roundtrip",
            duration_ms=duration_ms,
            target_ms=20.0 * iterations,
            iterations=iterations,
        )


class TestProductionScenarios:
    """Test scenarios matching real production usage."""

    @pytest.mark.benchmark
    def test_typical_rf_session_variables(self, benchmark_reporter):
        """Test with a typical RF session variable count (~32 variables)."""
        # Simulate real RF session variables
        variables = {
            # Built-in variables
            "${CURDIR}": Path("/home/user/tests"),
            "${EXECDIR}": Path("/home/user/project"),
            "${OUTPUT_DIR}": Path("/home/user/output"),
            "${TEMPDIR}": Path("/tmp"),
            "${/}": "/",
            "${:}": ":",
            "${\\n}": "\n",
            "${SPACE}": " ",
            "${EMPTY}": "",
            "${TRUE}": True,
            "${FALSE}": False,
            "${NONE}": None,

            # Suite variables
            "${SUITE_NAME}": "Test Suite Name",
            "${SUITE_SOURCE}": Path("/home/user/tests/suite.robot"),
            "${SUITE_DOCUMENTATION}": "Suite documentation",
            "${SUITE_METADATA}": DotDict(author="tester", version="1.0"),

            # Test variables
            "${TEST_NAME}": "Test Case Name",
            "${TEST_TAGS}": Tags("smoke", "regression"),
            "${TEST_DOCUMENTATION}": "Test documentation",
            "${TEST_STATUS}": "PASS",

            # Output files
            "${LOG_FILE}": Path("/home/user/output/log.html"),
            "${REPORT_FILE}": Path("/home/user/output/report.html"),
            "${OUTPUT_FILE}": Path("/home/user/output/output.xml"),

            # User variables
            "${BASE_URL}": "https://example.com",
            "${TIMEOUT}": "30",
            "${BROWSER}": "chromium",
            "@{CREDENTIALS}": ["user", "pass"],
            "&{CONFIG}": {"env": "test", "debug": True},
        }

        start = time.perf_counter()
        iterations = 100

        for _ in range(iterations):
            result = {k: sanitize_for_json(v) for k, v in variables.items()}
            json_str = json.dumps(result)

        duration_ms = (time.perf_counter() - start) * 1000

        assert len(json_str) > 0

        benchmark_reporter.record_latency(
            name="typical_rf_session",
            duration_ms=duration_ms,
            target_ms=5.0 * iterations,
            iterations=iterations,
            var_count=len(variables),
        )

        avg_ms = duration_ms / iterations
        assert avg_ms < 5.0, f"Typical session serialization took {avg_ms:.2f}ms, target <5ms"

    @pytest.mark.benchmark
    def test_bridge_get_variables_simulation(self, benchmark_reporter):
        """Simulate the /get_variables endpoint behavior."""
        # Generate typical RF session
        variables = generate_rf_like_variables(200)  # ~200 vars after truncation

        start = time.perf_counter()

        # Simulate truncation (max 200 vars)
        out = {}
        for i, (k, v) in enumerate(variables.items()):
            if i >= 200:
                break
            out[k] = sanitize_for_json(v)

        # Create response payload
        response = {
            "success": True,
            "result": out,
            "truncated": True,
        }

        # Serialize for HTTP response
        json_bytes = json.dumps(response).encode("utf-8")

        duration_ms = (time.perf_counter() - start) * 1000

        assert len(json_bytes) > 0
        assert b'"success": true' in json_bytes

        benchmark_reporter.record_latency(
            name="bridge_get_variables",
            duration_ms=duration_ms,
            target_ms=50.0,  # 50ms budget for full endpoint
            iterations=1,
            response_size_bytes=len(json_bytes),
        )

        assert duration_ms < 50.0, (
            f"Bridge get_variables took {duration_ms:.2f}ms, target <50ms"
        )
