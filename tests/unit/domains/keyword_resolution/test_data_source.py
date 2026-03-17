"""Tests for DataDriver integration (ADR-019 Phase 4)."""

import json
import os
import tempfile

import pytest

from robotmcp.domains.keyword_resolution.value_objects import (
    DataFormat,
    DataRow,
    DataSource,
)


class TestDataRowValueObject:
    def test_fields(self):
        row = DataRow(
            test_name="Test 1",
            arguments={"${a}": "1", "${b}": "2"},
            tags=("smoke",),
            documentation="A test",
        )
        assert row.test_name == "Test 1"
        assert row.arguments == {"${a}": "1", "${b}": "2"}
        assert row.tags == ("smoke",)
        assert row.documentation == "A test"

    def test_defaults(self):
        row = DataRow(test_name="Test", arguments={})
        assert row.tags == ()
        assert row.documentation == ""

    def test_frozen(self):
        row = DataRow(test_name="Test", arguments={})
        with pytest.raises(AttributeError):
            row.test_name = "other"  # type: ignore[misc]


class TestDataSourceValueObject:
    def test_count(self):
        ds = DataSource(
            file_path="/tmp/test.csv",
            format=DataFormat.CSV,
            rows=(
                DataRow(test_name="T1", arguments={}),
                DataRow(test_name="T2", arguments={}),
            ),
            column_names=("a", "b"),
        )
        assert ds.count == 2

    def test_empty(self):
        ds = DataSource(
            file_path="/tmp/test.csv",
            format=DataFormat.CSV,
            rows=(),
            column_names=(),
        )
        assert ds.count == 0


class TestDataFormatEnum:
    def test_values(self):
        assert DataFormat.CSV.value == "csv"
        assert DataFormat.JSON.value == "json"
        assert DataFormat.XLSX.value == "xlsx"
        assert DataFormat.XLS.value == "xls"


def _force_builtin(monkeypatch):
    """Monkeypatch helper to force built-in fallback by making DataDriver import fail."""
    import robotmcp.domains.keyword_resolution.services as svc

    monkeypatch.setattr(
        svc.DataSourceLoaderService,
        "_load_with_datadriver",
        staticmethod(
            lambda *a, **kw: (_ for _ in ()).throw(ImportError("forced"))
        ),
    )


class TestDataSourceLoaderBuiltin:
    """Tests using built-in CSV/JSON parsing (no DataDriver required)."""

    def test_load_csv(self, monkeypatch):
        from robotmcp.domains.keyword_resolution.services import (
            DataSourceLoaderService,
        )

        _force_builtin(monkeypatch)

        csv_content = "*** Test Cases ***;arg1;arg2\nTest Add;1;2\nTest Sub;4;5\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv_content)
            csv_path = f.name
        try:
            ds = DataSourceLoaderService.load(csv_path, delimiter=";")
            assert ds.format == DataFormat.CSV
            assert ds.count == 2
            assert ds.rows[0].test_name == "Test Add"
            assert "arg1" in ds.rows[0].arguments
            assert ds.rows[0].arguments["arg1"] == "1"
        finally:
            os.unlink(csv_path)

    def test_load_json(self, monkeypatch):
        from robotmcp.domains.keyword_resolution.services import (
            DataSourceLoaderService,
        )

        _force_builtin(monkeypatch)

        data = [
            {
                "test_case_name": "Login Admin",
                "arguments": {"${user}": "admin"},
                "tags": ["smoke"],
            },
            {
                "test_case_name": "Login Guest",
                "arguments": {"${user}": "guest"},
            },
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            json_path = f.name
        try:
            ds = DataSourceLoaderService.load(json_path)
            assert ds.format == DataFormat.JSON
            assert ds.count == 2
            assert ds.rows[0].test_name == "Login Admin"
            assert ds.rows[0].arguments["${user}"] == "admin"
            assert ds.rows[0].tags == ("smoke",)
        finally:
            os.unlink(json_path)

    def test_file_not_found(self):
        from robotmcp.domains.keyword_resolution.services import (
            DataSourceLoaderService,
        )

        with pytest.raises(FileNotFoundError):
            DataSourceLoaderService.load("/nonexistent/file.csv")

    def test_load_empty_csv(self, monkeypatch):
        from robotmcp.domains.keyword_resolution.services import (
            DataSourceLoaderService,
        )

        _force_builtin(monkeypatch)

        csv_content = "*** Test Cases ***;arg1\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv_content)
            csv_path = f.name
        try:
            ds = DataSourceLoaderService.load(csv_path, delimiter=";")
            assert ds.count == 0
        finally:
            os.unlink(csv_path)

    def test_csv_column_names(self, monkeypatch):
        from robotmcp.domains.keyword_resolution.services import (
            DataSourceLoaderService,
        )

        _force_builtin(monkeypatch)

        csv_content = "test_name;username;password\nLogin;admin;secret\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv_content)
            csv_path = f.name
        try:
            ds = DataSourceLoaderService.load(csv_path, delimiter=";")
            assert ds.column_names == ("username", "password")
            assert ds.count == 1
            assert ds.rows[0].arguments["username"] == "admin"
        finally:
            os.unlink(csv_path)

    def test_json_missing_tags_defaults_empty(self, monkeypatch):
        from robotmcp.domains.keyword_resolution.services import (
            DataSourceLoaderService,
        )

        _force_builtin(monkeypatch)

        data = [{"test_case_name": "T1", "arguments": {"${x}": "1"}}]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            json_path = f.name
        try:
            ds = DataSourceLoaderService.load(json_path)
            assert ds.rows[0].tags == ()
            assert ds.rows[0].documentation == ""
        finally:
            os.unlink(json_path)

    def test_csv_skip_blank_rows(self, monkeypatch):
        from robotmcp.domains.keyword_resolution.services import (
            DataSourceLoaderService,
        )

        _force_builtin(monkeypatch)

        csv_content = "name;val\nRow1;a\n ;b\nRow2;c\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv_content)
            csv_path = f.name
        try:
            ds = DataSourceLoaderService.load(csv_path, delimiter=";")
            assert ds.count == 2
            names = [r.test_name for r in ds.rows]
            assert "Row1" in names
            assert "Row2" in names
        finally:
            os.unlink(csv_path)
