"""Keyword Resolution Domain Services."""
from __future__ import annotations

import csv as csv_mod
import json as json_mod
import logging
import os
from typing import List, Optional, Tuple

from robotmcp.models.library_models import KeywordInfo
from .value_objects import (
    BddPrefix, DataFormat, DataRow, DataSource, EmbeddedMatch, EmbeddedPattern,
)

logger = logging.getLogger(__name__)


class BddPrefixService:
    """Stateless service for BDD prefix operations."""

    @staticmethod
    def strip_prefix(keyword_name: str) -> BddPrefix:
        return BddPrefix.from_keyword(keyword_name)

    @staticmethod
    def is_bdd_prefixed(keyword_name: str) -> bool:
        return BddPrefix.from_keyword(keyword_name).has_prefix


class EmbeddedMatcherService:
    """Service for matching concrete keyword names against embedded patterns.

    Uses RF's EmbeddedArguments API for parsing and matching.
    """

    @staticmethod
    def is_embedded_keyword(keyword_name: str) -> bool:
        return "${" in keyword_name

    @staticmethod
    def create_pattern(keyword_name: str) -> Optional[EmbeddedPattern]:
        if "${" not in keyword_name:
            return None
        try:
            from robot.running.arguments.embedded import EmbeddedArguments
            ea = EmbeddedArguments.from_name(keyword_name)
            if ea is None:
                return None
            return EmbeddedPattern(
                template_name=keyword_name,
                arg_names=tuple(ea.args),
                regex_pattern=ea.name.pattern if hasattr(ea.name, 'pattern') else str(ea.name),
            )
        except Exception:
            return None

    @staticmethod
    def match(
        concrete_name: str,
        embedded_keywords: List[Tuple[EmbeddedPattern, KeywordInfo]],
    ) -> Optional[Tuple[EmbeddedMatch, KeywordInfo]]:
        """Match a concrete keyword name against all embedded patterns."""
        from robot.running.arguments.embedded import EmbeddedArguments

        matches = []
        for pattern, kw_info in embedded_keywords:
            try:
                ea = EmbeddedArguments.from_name(pattern.template_name)
                if ea and ea.matches(concrete_name):
                    args = ea.parse_args(concrete_name)
                    match = EmbeddedMatch(
                        template_name=pattern.template_name,
                        concrete_name=concrete_name,
                        extracted_args=tuple(args),
                        library=kw_info.library,
                    )
                    matches.append((match, kw_info))
            except Exception:
                continue

        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        # Multiple matches: prefer more specific (more literal chars = fewer wildcards)
        return min(matches, key=lambda m: len(m[0].template_name))


class DataSourceLoaderService:
    """Loads test data from external files using DataDriver readers or built-in fallbacks."""

    @staticmethod
    def load(
        file_path: str,
        encoding: str = "utf-8",
        dialect: str = "Excel-EU",
        delimiter: str = ";",
        sheet_name: str = "0",
    ) -> DataSource:
        """Load data from a file. Uses DataDriver if available, else built-in CSV/JSON."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        try:
            return DataSourceLoaderService._load_with_datadriver(
                file_path, ext, encoding, dialect, delimiter, sheet_name
            )
        except ImportError:
            return DataSourceLoaderService._load_builtin(
                file_path, ext, encoding, delimiter
            )

    @staticmethod
    def _load_with_datadriver(
        file_path: str,
        ext: str,
        encoding: str,
        dialect: str,
        delimiter: str,
        sheet_name: str,
    ) -> DataSource:
        from DataDriver.ReaderConfig import ReaderConfig  # type: ignore[import-untyped]

        if ext == ".json":
            from DataDriver.json_reader import json_reader as reader_cls  # type: ignore[import-untyped]
        elif ext in (".xlsx", ".xls"):
            from DataDriver.xlsx_reader import xlsx_reader as reader_cls  # type: ignore[import-untyped]
        else:
            from DataDriver.csv_reader import csv_reader as reader_cls  # type: ignore[import-untyped]

        config = ReaderConfig(
            file=file_path,
            encoding=encoding,
            dialect=dialect,
            delimiter=delimiter,
            sheet_name=sheet_name,
        )
        reader = reader_cls(config)
        data = reader.get_data_from_source()

        rows = tuple(
            DataRow(
                test_name=tc.test_case_name or f"Test {i + 1}",
                arguments=dict(tc.arguments) if tc.arguments else {},
                tags=tuple(tc.tags) if tc.tags else (),
                documentation=tc.documentation or "",
            )
            for i, tc in enumerate(data)
        )
        col_names = (
            tuple(data[0].arguments.keys()) if data and data[0].arguments else ()
        )
        fmt_map = {
            ".json": DataFormat.JSON,
            ".xlsx": DataFormat.XLSX,
            ".xls": DataFormat.XLS,
        }
        fmt = fmt_map.get(ext, DataFormat.CSV)
        return DataSource(
            file_path=file_path, format=fmt, rows=rows, column_names=col_names
        )

    @staticmethod
    def _load_builtin(
        file_path: str, ext: str, encoding: str, delimiter: str
    ) -> DataSource:
        """Fallback: parse CSV/JSON without DataDriver."""
        if ext == ".json":
            with open(file_path, "r", encoding=encoding) as f:
                raw = json_mod.load(f)
            rows = tuple(
                DataRow(
                    test_name=item.get("test_case_name", f"Test {i + 1}"),
                    arguments=item.get("arguments", {}),
                    tags=tuple(item.get("tags", [])),
                    documentation=item.get("documentation", ""),
                )
                for i, item in enumerate(raw)
            )
            col_names = (
                tuple(raw[0]["arguments"].keys())
                if raw and "arguments" in raw[0]
                else ()
            )
            return DataSource(
                file_path=file_path,
                format=DataFormat.JSON,
                rows=rows,
                column_names=col_names,
            )

        # CSV fallback
        with open(file_path, "r", encoding=encoding) as f:
            reader = csv_mod.reader(f, delimiter=delimiter)
            header = next(reader)
            rows_list: list[DataRow] = []
            for i, row in enumerate(reader):
                if not row or not row[0].strip():
                    continue
                args: dict[str, str] = {}
                for j, col in enumerate(header[1:], 1):
                    if j < len(row):
                        args[col.strip()] = row[j].strip() if row[j] else ""
                rows_list.append(
                    DataRow(
                        test_name=row[0].strip() if row else f"Test {i + 1}",
                        arguments=args,
                    )
                )
        col_names = tuple(h.strip() for h in header[1:]) if len(header) > 1 else ()
        return DataSource(
            file_path=file_path,
            format=DataFormat.CSV,
            rows=tuple(rows_list),
            column_names=col_names,
        )
