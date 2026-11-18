"""The official Robot Framework skills: bundled in the `robotframework-plugins` package, or a snapshot of a ref.

rf-mcp reads them only through the package's public `robotframework_plugins.bundle` API
(MarketSquare/robotframework-plugins#5): `CATALOG_FORMAT`, `catalog()` and `skills_dirs()` for the bundled content,
and `fetch()` / `Snapshot` for a branch, tag or commit of the repository (official-skills-source, design D1-D3).
Which commit was fetched for a ref is recorded in rf-mcp's own state file, so startup never needs the network.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = frozenset({1})
BUNDLED = "bundled"
_SHA = re.compile(r"[0-9a-f]{40}")


def cache_base() -> Path:
    """The platform cache folder that robotframework-plugins also keeps its snapshots in."""
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")


def state_path() -> Path:
    """rf-mcp's record of the snapshot fetched for each ref (design D2)."""
    return cache_base() / "rf-mcp" / "official-skills.json"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")  # noqa: UP017 (Python 3.10)


def read_state() -> dict[str, dict[str, Any]]:
    """`{ref: {commit, path, fetched_at, checked_at, catalog_format, package_version}}`; empty when missing or invalid."""
    try:
        doc = json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    refs = doc.get("refs") if isinstance(doc, dict) else None
    return {str(k): dict(v) for k, v in refs.items() if isinstance(v, dict)} if isinstance(refs, dict) else {}


def _write_state(refs: dict[str, dict[str, Any]]) -> bool:
    """Replace the state file in one step; another process holding it open (Windows) gets a few retries."""
    target = state_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".official-skills-", suffix=".json", dir=target.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump({"refs": refs}, out, indent=2)
            out.write("\n")
    except OSError as exc:
        logger.info("could not write %s: %s", target, exc)
        return False
    for attempt in range(5):
        try:
            os.replace(tmp, target)
            return True
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
        except OSError as exc:
            logger.info("could not write %s: %s", target, exc)
            break
    try:
        os.unlink(tmp)
    except OSError:
        pass
    return False


def record_check(ref: str) -> None:
    """Note that `ref` is being checked now, before fetching, so other processes skip it (design D4)."""
    refs = read_state()
    refs.setdefault(ref, {})["checked_at"] = _now()
    _write_state(refs)


def record_fetch(ref: str, snapshot: Any, *, catalog_format: Any, package_version: str | None) -> dict[str, Any]:
    """Record `snapshot` (a `bundle.Snapshot`) as the current one of `ref`, and return the entry."""
    now = _now()
    entry = {
        "commit": str(snapshot.commit),
        "path": str(Path(snapshot.path).resolve()),
        "fetched_at": now,
        "checked_at": now,
        "catalog_format": catalog_format,
        "package_version": package_version,
    }
    refs = read_state()
    refs[ref] = entry
    _write_state(refs)
    return entry


@dataclass(frozen=True)
class OfficialSkills:
    """What rf-mcp serves as the official skills: the bundled content or a snapshot; empty when unusable."""

    version: str | None = None
    catalog_format: Any = None
    skills_dirs: dict[str, Path] = field(default_factory=dict)
    catalog: list[dict[str, Any]] = field(default_factory=list)
    warning: str | None = None
    channel: str = BUNDLED
    # Set when a snapshot of `channel` is served (design D3).
    commit: str | None = None
    fetched_at: str | None = None
    path: Path | None = None
    # Why a ref channel serves the bundled content instead.
    fallback_reason: str | None = None

    @property
    def installed(self) -> bool:
        return self.version is not None or self.catalog_format is not None

    @property
    def from_snapshot(self) -> bool:
        return self.commit is not None

    def skill_count(self) -> int:
        return sum(1 for d in self.skills_dirs.values() for s in d.iterdir() if (s / "SKILL.md").is_file())


def is_commit(ref: str) -> bool:
    """A full commit SHA: its snapshot never changes, so it is never refreshed (design D6)."""
    return bool(_SHA.fullmatch(ref.strip().lower()))


def _bundle() -> Any:
    try:
        from robotframework_plugins import bundle  # type: ignore[import-not-found]
    except ImportError:
        return None
    return bundle


def _load_bundled(bundle: Any) -> OfficialSkills:
    """The package's own content; a missing package or an unsupported catalogue format gives no official skills."""
    if bundle is None:
        return OfficialSkills()
    version = getattr(sys.modules.get("robotframework_plugins"), "__version__", None)
    fmt = getattr(bundle, "CATALOG_FORMAT", None)
    if fmt not in SUPPORTED_FORMATS:
        warning = (
            f"robotframework-plugins {version or '?'} has catalogue format {fmt!r}; rf-mcp supports "
            f"{', '.join(str(f) for f in sorted(SUPPORTED_FORMATS))}. Its skills are not served."
        )
        logger.warning(warning)
        return OfficialSkills(version, fmt, warning=warning)
    try:
        dirs = {str(name): Path(path).resolve() for name, path in bundle.skills_dirs().items()}
        catalog = [dict(entry) for entry in bundle.catalog()]
    except Exception as exc:  # a broken install must not take the server down
        warning = f"robotframework-plugins {version or '?'} could not be read: {exc}. Its skills are not served."
        logger.warning(warning)
        return OfficialSkills(version, fmt, warning=warning)
    return OfficialSkills(version, fmt, {n: p for n, p in dirs.items() if p.is_dir()}, catalog)


def _snapshot_problem(channel: str, entry: dict[str, Any] | None, bundled: OfficialSkills) -> str | None:
    if not entry or not entry.get("commit") or not entry.get("path"):
        return f"no snapshot of {channel!r} has been fetched yet"
    commit, path = str(entry["commit"]), Path(str(entry["path"]))
    if not (path / "catalog.json").is_file():
        return f"the snapshot of {channel!r} ({commit[:12]}) is no longer in {path}"
    fmt = entry.get("catalog_format")
    if fmt not in SUPPORTED_FORMATS or fmt != bundled.catalog_format:
        return (f"the snapshot of {channel!r} ({commit[:12]}) has catalogue format {fmt!r}, but the installed "
                f"robotframework-plugins {bundled.version or '?'} has {bundled.catalog_format!r}")
    return None


@lru_cache(maxsize=8)
def load_official_skills(channel: str = BUNDLED) -> OfficialSkills:
    """The official skills of `channel`, read once per process and channel, never over the network (design D3).

    `bundled` reads the installed package. A ref reads the snapshot recorded for it in the state file when that is
    complete and in the package's catalogue format, else falls back to the bundled content with a warning.
    """
    channel = channel.strip() or BUNDLED
    bundle = _bundle()
    bundled = _load_bundled(bundle)
    if channel == BUNDLED or bundle is None or (bundled.warning and not bundled.skills_dirs):
        return replace(bundled, channel=channel)
    entry = read_state().get(channel)
    problem = _snapshot_problem(channel, entry, bundled)
    if problem is None and entry is not None:
        try:
            snap = bundle.Snapshot(ref=channel, commit=str(entry["commit"]), path=Path(str(entry["path"])))
            dirs = {str(n): Path(p).resolve() for n, p in snap.skills_dirs().items()}
            catalog = [dict(e) for e in snap.catalog()]
        except Exception as exc:  # a damaged snapshot must not take the server down
            problem = f"the snapshot of {channel!r} could not be read: {exc}"
        else:
            return OfficialSkills(
                bundled.version, bundled.catalog_format, {n: p for n, p in dirs.items() if p.is_dir()}, catalog,
                channel=channel, commit=str(entry["commit"]), fetched_at=entry.get("fetched_at"),
                path=Path(str(entry["path"])),
            )
    warning = (
        f"Official skills channel {channel!r}: {problem}. Serving the bundled robotframework-plugins "
        f"{bundled.version or '?'} instead; `rf-mcp update-skills --ref {channel}` fetches it."
    )
    logger.warning(warning)
    return replace(bundled, channel=channel, fallback_reason=problem, warning=warning)


def fetch_and_record(ref: str, *, timeout: float = 30.0) -> tuple[Any, dict[str, Any]]:
    """Fetch `ref` with the package's `bundle.fetch()` and record it as the ref's snapshot (design D4, D7).

    Raises the package's `FetchError` (or `RuntimeError` without the package); the state is then unchanged.
    """
    bundle = _bundle()
    if bundle is None:
        raise RuntimeError("robotframework-plugins is not installed")
    snap = bundle.fetch(ref, timeout=timeout)
    version = getattr(sys.modules.get("robotframework_plugins"), "__version__", None)
    # The snapshot's own format (upstream #23); robotframework-plugins 2026.11.0.dev1 doesn't record it.
    fmt = getattr(snap, "catalog_format", None)
    if fmt is None:
        fmt = getattr(bundle, "CATALOG_FORMAT", None)
    entry = record_fetch(ref, snap, catalog_format=fmt, package_version=version)
    return snap, entry


def refresh_due(channel: str, hours: float, *, now: _dt.datetime | None = None) -> bool:
    """Whether a background fetch of `channel` is due: a branch or tag last checked more than `hours` ago."""
    if channel == BUNDLED or is_commit(channel) or hours <= 0:
        return False
    checked = read_state().get(channel, {}).get("checked_at")
    if not checked:
        return True
    try:
        last = _dt.datetime.fromisoformat(str(checked))
    except ValueError:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=_dt.timezone.utc)  # noqa: UP017 (Python 3.10)
    current = now or _dt.datetime.now(_dt.timezone.utc)  # noqa: UP017 (Python 3.10)
    return current - last >= _dt.timedelta(hours=hours)


def refresh(channel: str, hours: float) -> str:
    """One background refresh of `channel`: `skipped`, `fetched` or `failed`. Never raises (design D4)."""
    if _bundle() is None or not refresh_due(channel, hours):
        return "skipped"
    record_check(channel)  # before fetching, so rf-mcp processes started at the same time skip it
    try:
        snap, _ = fetch_and_record(channel)
    except Exception as exc:  # offline, rate-limited, unknown ref: keep serving what we have
        logger.warning("Could not refresh the official skills channel %r: %s", channel, exc)
        return "failed"
    logger.info("Fetched the official skills channel %r at %s; served from the next start", channel, snap.commit)
    return "fetched"


def start_background_refresh(channel: str, hours: float) -> threading.Thread | None:
    """Run `refresh()` in a daemon thread when it is due, without waiting for it; None when nothing is due."""
    if channel == BUNDLED or is_commit(channel) or hours <= 0:
        return None
    thread = threading.Thread(target=refresh, args=(channel, hours), name="rf-mcp-official-skills", daemon=True)
    thread.start()
    return thread
