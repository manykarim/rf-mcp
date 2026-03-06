"""Tests for Artifact Output Domain Value Objects (ADR-015)."""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import patch

import pytest

from robotmcp.domains.artifact_output.value_objects import (
    ArtifactId,
    ArtifactPolicy,
    ArtifactReference,
    ExternalizationResult,
    ExternalizationRule,
    OutputMode,
)


# ---------------------------------------------------------------------------
# OutputMode
# ---------------------------------------------------------------------------

class TestOutputMode:
    """Tests for OutputMode enum."""

    def test_inline_value(self):
        assert OutputMode.INLINE.value == "inline"

    def test_file_value(self):
        assert OutputMode.FILE.value == "file"

    def test_auto_value(self):
        assert OutputMode.AUTO.value == "auto"

    def test_from_env_default_is_auto(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ROBOTMCP_OUTPUT_MODE", None)
            assert OutputMode.from_env() == OutputMode.AUTO

    def test_from_env_inline(self):
        with patch.dict(os.environ, {"ROBOTMCP_OUTPUT_MODE": "inline"}):
            assert OutputMode.from_env() == OutputMode.INLINE

    def test_from_env_file(self):
        with patch.dict(os.environ, {"ROBOTMCP_OUTPUT_MODE": "file"}):
            assert OutputMode.from_env() == OutputMode.FILE

    def test_from_env_auto(self):
        with patch.dict(os.environ, {"ROBOTMCP_OUTPUT_MODE": "auto"}):
            assert OutputMode.from_env() == OutputMode.AUTO

    def test_from_env_case_insensitive(self):
        with patch.dict(os.environ, {"ROBOTMCP_OUTPUT_MODE": "INLINE"}):
            assert OutputMode.from_env() == OutputMode.INLINE

    def test_from_env_strips_whitespace(self):
        with patch.dict(os.environ, {"ROBOTMCP_OUTPUT_MODE": "  file  "}):
            assert OutputMode.from_env() == OutputMode.FILE

    def test_from_env_invalid_returns_auto(self):
        with patch.dict(os.environ, {"ROBOTMCP_OUTPUT_MODE": "bogus"}):
            assert OutputMode.from_env() == OutputMode.AUTO


# ---------------------------------------------------------------------------
# ArtifactId
# ---------------------------------------------------------------------------

class TestArtifactId:
    """Tests for ArtifactId value object."""

    def test_valid_id(self):
        aid = ArtifactId("art_0123456789ab")
        assert aid.value == "art_0123456789ab"

    def test_str_returns_value(self):
        aid = ArtifactId("art_aabbccddeeff")
        assert str(aid) == "art_aabbccddeeff"

    def test_frozen(self):
        aid = ArtifactId("art_0123456789ab")
        with pytest.raises(AttributeError):
            aid.value = "art_000000000000"  # type: ignore[misc]

    def test_invalid_no_prefix(self):
        with pytest.raises(ValueError, match="Invalid ArtifactId"):
            ArtifactId("0123456789ab")

    def test_invalid_too_short(self):
        with pytest.raises(ValueError, match="Invalid ArtifactId"):
            ArtifactId("art_0123")

    def test_invalid_too_long(self):
        with pytest.raises(ValueError, match="Invalid ArtifactId"):
            ArtifactId("art_0123456789abcdef")

    def test_invalid_uppercase_hex(self):
        with pytest.raises(ValueError, match="Invalid ArtifactId"):
            ArtifactId("art_0123456789AB")

    def test_invalid_non_hex(self):
        with pytest.raises(ValueError, match="Invalid ArtifactId"):
            ArtifactId("art_0123456789gz")

    def test_generate_format(self):
        aid = ArtifactId.generate()
        assert aid.value.startswith("art_")
        assert len(aid.value) == 16  # "art_" + 12 hex chars

    def test_generate_unique(self):
        ids = {ArtifactId.generate().value for _ in range(100)}
        assert len(ids) == 100

    def test_equality(self):
        a = ArtifactId("art_aabbccddeeff")
        b = ArtifactId("art_aabbccddeeff")
        assert a == b

    def test_inequality(self):
        a = ArtifactId("art_aabbccddeeff")
        b = ArtifactId("art_112233445566")
        assert a != b


# ---------------------------------------------------------------------------
# ArtifactReference
# ---------------------------------------------------------------------------

class TestArtifactReference:
    """Tests for ArtifactReference value object."""

    def _make_ref(self, **kwargs):
        defaults = {
            "artifact_id": "art_aabbccddeeff",
            "file_path": ".artifacts/art_aabbccddeeff",
            "content_hash": "abc123def456",
            "byte_size": 1024,
            "token_estimate": 256,
            "mime_type": "text/plain",
        }
        defaults.update(kwargs)
        return ArtifactReference(**defaults)

    def test_construction(self):
        ref = self._make_ref()
        assert ref.artifact_id == "art_aabbccddeeff"
        assert ref.byte_size == 1024
        assert ref.token_estimate == 256

    def test_frozen(self):
        ref = self._make_ref()
        with pytest.raises(AttributeError):
            ref.byte_size = 999  # type: ignore[misc]

    def test_negative_byte_size_rejected(self):
        with pytest.raises(ValueError, match="byte_size cannot be negative"):
            self._make_ref(byte_size=-1)

    def test_negative_token_estimate_rejected(self):
        with pytest.raises(ValueError, match="token_estimate cannot be negative"):
            self._make_ref(token_estimate=-1)

    def test_zero_sizes_ok(self):
        ref = self._make_ref(byte_size=0, token_estimate=0)
        assert ref.byte_size == 0

    def test_created_at_default(self):
        ref = self._make_ref()
        assert isinstance(ref.created_at, datetime)

    def test_to_inline_dict(self):
        ref = self._make_ref(content_hash="abcdef1234567890")
        d = ref.to_inline_dict()
        assert d["artifact_id"] == "art_aabbccddeeff"
        assert d["file"] == ".artifacts/art_aabbccddeeff"
        assert d["size"] == 1024
        assert d["tokens"] == 256
        assert d["hash"] == "abcdef12"  # first 8 chars

    def test_to_inline_dict_hash_truncation(self):
        ref = self._make_ref(content_hash="1234567890abcdef")
        assert ref.to_inline_dict()["hash"] == "12345678"


# ---------------------------------------------------------------------------
# ArtifactPolicy
# ---------------------------------------------------------------------------

class TestArtifactPolicy:
    """Tests for ArtifactPolicy value object."""

    def test_defaults(self):
        p = ArtifactPolicy()
        assert p.max_inline_tokens == 500
        assert p.artifact_dir == ".robotmcp_artifacts"
        assert p.retention_ttl_seconds == 3600
        assert p.max_artifacts == 100

    def test_custom_values(self):
        p = ArtifactPolicy(max_inline_tokens=100, artifact_dir="/tmp/arts", retention_ttl_seconds=60, max_artifacts=10)
        assert p.max_inline_tokens == 100
        assert p.artifact_dir == "/tmp/arts"

    def test_frozen(self):
        p = ArtifactPolicy()
        with pytest.raises(AttributeError):
            p.max_inline_tokens = 999  # type: ignore[misc]

    def test_negative_max_inline_tokens_rejected(self):
        with pytest.raises(ValueError, match="max_inline_tokens cannot be negative"):
            ArtifactPolicy(max_inline_tokens=-1)

    def test_negative_retention_ttl_rejected(self):
        with pytest.raises(ValueError, match="retention_ttl_seconds cannot be negative"):
            ArtifactPolicy(retention_ttl_seconds=-1)

    def test_zero_max_artifacts_rejected(self):
        with pytest.raises(ValueError, match="max_artifacts must be >= 1"):
            ArtifactPolicy(max_artifacts=0)

    def test_from_env_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            for k in ("ROBOTMCP_MAX_INLINE_TOKENS", "ROBOTMCP_ARTIFACT_DIR", "ROBOTMCP_ARTIFACT_TTL"):
                os.environ.pop(k, None)
            p = ArtifactPolicy.from_env()
            assert p.max_inline_tokens == 500
            assert p.artifact_dir == ".robotmcp_artifacts"
            assert p.retention_ttl_seconds == 3600

    def test_from_env_custom(self):
        with patch.dict(os.environ, {
            "ROBOTMCP_MAX_INLINE_TOKENS": "200",
            "ROBOTMCP_ARTIFACT_DIR": "/custom",
            "ROBOTMCP_ARTIFACT_TTL": "120",
        }):
            p = ArtifactPolicy.from_env()
            assert p.max_inline_tokens == 200
            assert p.artifact_dir == "/custom"
            assert p.retention_ttl_seconds == 120

    def test_should_externalize_below_threshold(self):
        p = ArtifactPolicy(max_inline_tokens=10)
        # 10 tokens * 4 chars = 40 chars threshold
        assert not p.should_externalize("x" * 39)

    def test_should_externalize_at_threshold(self):
        p = ArtifactPolicy(max_inline_tokens=10)
        # Exactly at threshold: 40 chars / 4.0 = 10.0, not > 10
        assert not p.should_externalize("x" * 40)

    def test_should_externalize_above_threshold(self):
        p = ArtifactPolicy(max_inline_tokens=10)
        assert p.should_externalize("x" * 41)

    def test_should_externalize_empty_string(self):
        p = ArtifactPolicy(max_inline_tokens=0)
        # 0/4.0 = 0.0, not > 0
        assert not p.should_externalize("")


# ---------------------------------------------------------------------------
# ExternalizationRule
# ---------------------------------------------------------------------------

class TestExternalizationRule:
    """Tests for ExternalizationRule value object."""

    def test_construction(self):
        r = ExternalizationRule(tool_name="my_tool", field_path="output")
        assert r.tool_name == "my_tool"
        assert r.field_path == "output"
        assert "artifact" in r.summary_template.lower() or "{artifact_id}" in r.summary_template

    def test_custom_template(self):
        r = ExternalizationRule(tool_name="t", field_path="f", summary_template="custom {artifact_id}")
        assert r.summary_template == "custom {artifact_id}"

    def test_empty_tool_name_rejected(self):
        with pytest.raises(ValueError, match="tool_name cannot be empty"):
            ExternalizationRule(tool_name="", field_path="output")

    def test_empty_field_path_rejected(self):
        with pytest.raises(ValueError, match="field_path cannot be empty"):
            ExternalizationRule(tool_name="tool", field_path="")

    def test_frozen(self):
        r = ExternalizationRule(tool_name="t", field_path="f")
        with pytest.raises(AttributeError):
            r.tool_name = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ExternalizationResult
# ---------------------------------------------------------------------------

class TestExternalizationResult:
    """Tests for ExternalizationResult value object."""

    def _make_ref(self):
        return ArtifactReference(
            artifact_id="art_aabbccddeeff",
            file_path=".artifacts/art_aabbccddeeff",
            content_hash="abc123",
            byte_size=1024,
            token_estimate=256,
            mime_type="text/plain",
        )

    def test_construction(self):
        r = ExternalizationResult(summary="done", artifact_ref=None, original_token_estimate=100, saved_tokens=0)
        assert r.summary == "done"
        assert r.artifact_ref is None

    def test_negative_saved_tokens_rejected(self):
        with pytest.raises(ValueError, match="saved_tokens cannot be negative"):
            ExternalizationResult(summary="x", artifact_ref=None, original_token_estimate=10, saved_tokens=-1)

    def test_to_dict_without_artifact(self):
        r = ExternalizationResult(summary="s", artifact_ref=None, original_token_estimate=50, saved_tokens=0)
        d = r.to_dict()
        assert d["summary"] == "s"
        assert d["original_tokens"] == 50
        assert d["saved_tokens"] == 0
        assert "artifact" not in d

    def test_to_dict_with_artifact(self):
        ref = self._make_ref()
        r = ExternalizationResult(summary="s", artifact_ref=ref, original_token_estimate=256, saved_tokens=200)
        d = r.to_dict()
        assert "artifact" in d
        assert d["artifact"]["artifact_id"] == "art_aabbccddeeff"
        assert d["saved_tokens"] == 200

    def test_frozen(self):
        r = ExternalizationResult(summary="s", artifact_ref=None, original_token_estimate=10, saved_tokens=0)
        with pytest.raises(AttributeError):
            r.summary = "x"  # type: ignore[misc]
