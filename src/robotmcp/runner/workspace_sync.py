"""Inline workspace file synchronization for Mode 2 remote test runners."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping


def sync_inline_files(
    workspace_root: Path,
    files: Mapping[str, str] | None,
) -> list[str]:
    """Write inline file payloads into `workspace_root` with path-traversal protection.

    Returns the list of relative POSIX file paths written.
    """
    if not files:
        return []

    resolved_root = workspace_root.expanduser().resolve()
    written: list[str] = []

    for rel_path_str, content in files.items():
        if not rel_path_str or not rel_path_str.strip():
            raise ValueError("File path in `files` cannot be empty.")
        candidate_rel = Path(rel_path_str)
        if candidate_rel.is_absolute():
            raise ValueError(
                f"Absolute paths are not allowed in inline workspace sync: {rel_path_str}"
            )
        target = (resolved_root / candidate_rel).resolve()
        try:
            rel_to_root = target.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"Path traversal outside workspace root is forbidden: {rel_path_str}"
            ) from exc

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(rel_to_root.as_posix())

    return written


# Directories a glob never descends into: environments, caches, VCS and run results.
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".robotcode_cache", ".pytest_cache", "results"}
# Result artifacts are read through rf_results, never as text.
_RESULT_FILES = ("output.xml", "log.html", "report.html")
_MAX_MATCHES = 200


def _is_result_file(path: Path) -> bool:
    name = path.name.lower()
    return name in _RESULT_FILES or (name.startswith("output") and name.endswith((".xml", ".json")) and "results" in path.parts)


def _expand(resolved_root: Path, pattern: str) -> list[Path]:
    if not pattern or not pattern.strip():
        raise ValueError("A path in `paths` cannot be empty.")
    if Path(pattern).is_absolute():
        raise ValueError(f"Absolute paths are not allowed; give paths relative to the project root: {pattern}")
    if any(ch in pattern for ch in "*?["):
        matches = []
        for p in sorted(resolved_root.glob(pattern)):
            rel_parts = p.relative_to(resolved_root).parts
            if p.is_file() and not any(part in _SKIP_DIRS for part in rel_parts[:-1]):
                matches.append(p)
        return matches[:_MAX_MATCHES]
    return [resolved_root / pattern]


def read_project_files(workspace_root: Path, patterns: list[str], max_chars: int) -> dict:
    """Read several text files under `workspace_root` (paths or globs) into one payload.

    Files are returned in request order, each once, keyed by root-relative POSIX path, until `max_chars`
    is used up; files that do not fit are listed in `omitted` with their size so they can be asked for
    in another call. A first file larger than the whole budget is cut and listed in `truncated`.
    """
    resolved_root = workspace_root.expanduser().resolve()
    files: dict[str, str] = {}
    omitted: list[dict] = []
    truncated: list[str] = []
    skipped: list[dict] = []
    unmatched: list[str] = []
    remaining = max_chars
    for pattern in patterns:
        matches = _expand(resolved_root, pattern)
        if not matches:
            unmatched.append(pattern)
        for candidate in matches:
            target = candidate.resolve()
            try:
                rel = target.relative_to(resolved_root).as_posix()
            except ValueError as exc:
                raise ValueError(f"Path outside the project root is forbidden: {pattern}") from exc
            if rel in files or any(o["path"] == rel for o in omitted):
                continue
            if not target.is_file():
                unmatched.append(pattern)
                continue
            if _is_result_file(target):
                skipped.append({"path": rel, "reason": "run result: query it with rf_results (output_path=...)"})
                continue
            raw = target.read_bytes()
            if b"\x00" in raw[:4096]:
                skipped.append({"path": rel, "reason": "binary file"})
                continue
            text = raw.decode("utf-8", errors="replace")
            if len(text) <= remaining:
                files[rel] = text
                remaining -= len(text)
            elif not files and not omitted:
                files[rel] = text[:remaining]
                truncated.append(rel)
                remaining = 0
            else:
                omitted.append({"path": rel, "chars": len(text)})
    result: dict = {"files": files}
    for key, value in (("truncated", truncated), ("omitted", omitted), ("skipped", skipped), ("unmatched", unmatched)):
        if value:
            result[key] = value
    if omitted or truncated:
        result["note"] = f"Output limit {max_chars} characters reached; ask for the omitted files in another call."
    return result
