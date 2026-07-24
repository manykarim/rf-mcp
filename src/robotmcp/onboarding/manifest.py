"""Hash-tracked install manifest.

Records every change the installer makes so ``uninstall`` can revert exactly what
rf-mcp wrote and nothing else: each entry stores the agent, scope, config path,
whether rf-mcp created the whole file, and a hash of the server entry it inserted.
On uninstall, an entry is reverted only if the current on-disk value still hashes
to the recorded value - user-edited entries are left intact and reported.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1


def default_manifest_path() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "robotmcp" / "install-manifest.json"


def value_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


class Manifest:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or default_manifest_path()
        self.data: Dict[str, Any] = {"version": SCHEMA_VERSION, "entries": []}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
            self.data.setdefault("version", SCHEMA_VERSION)
            self.data.setdefault("entries", [])

    @property
    def entries(self) -> List[Dict[str, Any]]:
        return self.data["entries"]

    def find(self, agent: str, scope: str, what: str, path: str) -> Optional[Dict[str, Any]]:
        for e in self.entries:
            if (e["agent"], e["scope"], e["what"], e["path"]) == (agent, scope, what, path):
                return e
        return None

    def record(self, *, agent: str, scope: str, what: str, path: str,
               value: Any, created_file: bool) -> None:
        e = self.find(agent, scope, what, path)
        payload = {
            "agent": agent, "scope": scope, "what": what, "path": path,
            "value_hash": value_hash(value), "created_file": created_file,
        }
        if e:
            e.update(payload)
        else:
            self.entries.append(payload)

    def drop(self, entry: Dict[str, Any]) -> None:
        self.entries[:] = [
            e for e in self.entries
            if (e["agent"], e["scope"], e["what"], e["path"])
            != (entry["agent"], entry["scope"], entry["what"], entry["path"])
        ]

    def entries_for(self, agent_ids: List[str], scope: Optional[str], whats: List[str]) -> List[Dict[str, Any]]:
        return [
            e for e in self.entries
            if e["agent"] in agent_ids
            and (scope is None or e["scope"] == scope)
            and e["what"] in whats
        ]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
