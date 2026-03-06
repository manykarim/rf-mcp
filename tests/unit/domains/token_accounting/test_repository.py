"""Tests for Token Accounting Repository (ADR-017)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotmcp.domains.token_accounting.repository import (
    JsonFileBaselineRepository,
)


@pytest.fixture
def repo_path(tmp_path: Path) -> str:
    return str(tmp_path / "baselines" / "token_baselines.json")


@pytest.fixture
def repo(repo_path: str) -> JsonFileBaselineRepository:
    return JsonFileBaselineRepository(file_path=repo_path)


class TestJsonFileBaselineRepository:
    """Tests for JsonFileBaselineRepository."""

    def test_get_baseline_missing_file(self, repo: JsonFileBaselineRepository):
        assert repo.get_baseline("nonexistent") is None

    def test_save_and_get_baseline(self, repo: JsonFileBaselineRepository):
        repo.save_baseline("full", 1500)
        assert repo.get_baseline("full") == 1500

    def test_save_overwrites(self, repo: JsonFileBaselineRepository):
        repo.save_baseline("full", 1500)
        repo.save_baseline("full", 2000)
        assert repo.get_baseline("full") == 2000

    def test_multiple_profiles(self, repo: JsonFileBaselineRepository):
        repo.save_baseline("full", 1500)
        repo.save_baseline("compact", 800)
        repo.save_baseline("minimal", 300)
        assert repo.get_baseline("full") == 1500
        assert repo.get_baseline("compact") == 800
        assert repo.get_baseline("minimal") == 300

    def test_get_all_empty(self, repo: JsonFileBaselineRepository):
        assert repo.get_all() == {}

    def test_get_all(self, repo: JsonFileBaselineRepository):
        repo.save_baseline("a", 100)
        repo.save_baseline("b", 200)
        result = repo.get_all()
        assert result == {"a": 100, "b": 200}

    def test_creates_directories(self, repo_path: str):
        repo = JsonFileBaselineRepository(file_path=repo_path)
        repo.save_baseline("test", 100)
        assert Path(repo_path).exists()

    def test_file_format(self, repo: JsonFileBaselineRepository, repo_path: str):
        repo.save_baseline("test", 500)
        data = json.loads(Path(repo_path).read_text())
        assert "baselines" in data
        assert data["baselines"]["test"] == 500
        assert "meta" in data
        assert "version" in data["meta"]
        assert data["meta"]["version"] == 1
        assert "updated" in data["meta"]

    def test_file_has_trailing_newline(self, repo: JsonFileBaselineRepository, repo_path: str):
        repo.save_baseline("test", 100)
        content = Path(repo_path).read_text()
        assert content.endswith("\n")

    def test_file_is_pretty_printed(self, repo: JsonFileBaselineRepository, repo_path: str):
        repo.save_baseline("test", 100)
        content = Path(repo_path).read_text()
        # Pretty printed JSON has multiple lines
        assert content.count("\n") > 2

    def test_corrupted_json(self, repo_path: str):
        Path(repo_path).parent.mkdir(parents=True, exist_ok=True)
        Path(repo_path).write_text("not json {{{")
        repo = JsonFileBaselineRepository(file_path=repo_path)
        assert repo.get_baseline("anything") is None

    def test_corrupted_json_get_all(self, repo_path: str):
        Path(repo_path).parent.mkdir(parents=True, exist_ok=True)
        Path(repo_path).write_text("not json {{{")
        repo = JsonFileBaselineRepository(file_path=repo_path)
        assert repo.get_all() == {}

    def test_empty_json_file(self, repo_path: str):
        Path(repo_path).parent.mkdir(parents=True, exist_ok=True)
        Path(repo_path).write_text("")
        repo = JsonFileBaselineRepository(file_path=repo_path)
        assert repo.get_baseline("anything") is None

    def test_save_to_corrupted_file(self, repo_path: str):
        Path(repo_path).parent.mkdir(parents=True, exist_ok=True)
        Path(repo_path).write_text("garbage")
        repo = JsonFileBaselineRepository(file_path=repo_path)
        # Save should overwrite
        repo.save_baseline("fresh", 42)
        assert repo.get_baseline("fresh") == 42

    def test_get_nonexistent_profile(self, repo: JsonFileBaselineRepository):
        repo.save_baseline("a", 100)
        assert repo.get_baseline("b") is None

    def test_default_path(self):
        repo = JsonFileBaselineRepository()
        assert repo._path == Path("tests/baselines/token_baselines.json")

    def test_custom_path(self, tmp_path: Path):
        custom = str(tmp_path / "custom" / "baselines.json")
        repo = JsonFileBaselineRepository(file_path=custom)
        repo.save_baseline("test", 999)
        assert repo.get_baseline("test") == 999

    def test_round_trip_preserves_all_profiles(self, repo: JsonFileBaselineRepository):
        for i in range(20):
            repo.save_baseline(f"profile_{i}", i * 100)
        all_baselines = repo.get_all()
        assert len(all_baselines) == 20
        for i in range(20):
            assert all_baselines[f"profile_{i}"] == i * 100

    def test_save_zero_tokens(self, repo: JsonFileBaselineRepository):
        repo.save_baseline("zero", 0)
        assert repo.get_baseline("zero") == 0

    def test_save_large_tokens(self, repo: JsonFileBaselineRepository):
        repo.save_baseline("large", 999_999)
        assert repo.get_baseline("large") == 999_999

    def test_meta_updated_on_save(self, repo: JsonFileBaselineRepository, repo_path: str):
        repo.save_baseline("a", 100)
        data1 = json.loads(Path(repo_path).read_text())
        ts1 = data1["meta"]["updated"]

        repo.save_baseline("b", 200)
        data2 = json.loads(Path(repo_path).read_text())
        ts2 = data2["meta"]["updated"]

        assert ts2 >= ts1

    def test_json_valid_after_multiple_saves(self, repo: JsonFileBaselineRepository, repo_path: str):
        repo.save_baseline("a", 100)
        repo.save_baseline("b", 200)
        repo.save_baseline("a", 150)
        # File should still be valid JSON
        data = json.loads(Path(repo_path).read_text())
        assert data["baselines"]["a"] == 150
        assert data["baselines"]["b"] == 200

    def test_no_baselines_key_in_file(self, repo_path: str):
        Path(repo_path).parent.mkdir(parents=True, exist_ok=True)
        Path(repo_path).write_text('{"meta": {"version": 1}}')
        repo = JsonFileBaselineRepository(file_path=repo_path)
        assert repo.get_baseline("x") is None
        assert repo.get_all() == {}
