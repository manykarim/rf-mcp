"""Token Accounting Domain Repository (ADR-017)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Protocol

logger = logging.getLogger(__name__)


class TokenBaselineRepository(Protocol):
    """Protocol for baseline persistence."""

    def get_baseline(self, profile_name: str) -> Optional[int]:
        ...

    def save_baseline(self, profile_name: str, tokens: int) -> None:
        ...

    def get_all(self) -> Dict[str, int]:
        ...


class JsonFileBaselineRepository:
    """JSON file-backed baseline repository."""

    def __init__(
        self, file_path: str = "tests/baselines/token_baselines.json"
    ):
        self._path = Path(file_path)

    def get_baseline(self, profile_name: str) -> Optional[int]:
        data = self._load()
        return data.get("baselines", {}).get(profile_name)

    def save_baseline(self, profile_name: str, tokens: int) -> None:
        data = self._load()
        if "baselines" not in data:
            data["baselines"] = {}
        data["baselines"][profile_name] = tokens
        data["meta"] = {
            "version": 1,
            "updated": datetime.now().isoformat(),
        }
        self._save(data)

    def get_all(self) -> Dict[str, int]:
        return self._load().get("baselines", {})

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning(f"Failed to load baselines from {self._path}")
            return {}

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2) + "\n")
