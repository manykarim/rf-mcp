"""Unit tests for inline workspace file synchronization and path-traversal guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from robotmcp.runner.workspace_sync import sync_inline_files


def test_sync_inline_files_writes_nested_files(tmp_path: Path) -> None:
    written = sync_inline_files(
        tmp_path,
        {
            "robot.toml": 'paths = ["tests"]\n',
            "tests/login.robot": "*** Test Cases ***\nDemo\n    Log    Hi\n",
        },
    )
    assert written == ["robot.toml", "tests/login.robot"]
    assert (tmp_path / "robot.toml").is_file()
    assert (tmp_path / "tests" / "login.robot").is_file()


def test_sync_inline_files_rejects_path_traversal_and_absolute_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Path traversal"):
        sync_inline_files(tmp_path, {"../escape.robot": "bad"})

    with pytest.raises(ValueError, match="Absolute paths"):
        sync_inline_files(tmp_path, {str(tmp_path / "abs.robot"): "bad"})
