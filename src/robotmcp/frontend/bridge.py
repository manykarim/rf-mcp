"""Bridge exposing RobotMCP data to the Django frontend."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from robotmcp.core.event_bus import FrontendEvent, event_bus
from robotmcp.models.execution_models import ExecutionStep
from robotmcp.models.session_models import ExecutionSession
from robotmcp.server import execution_engine, state_manager, test_builder


class McpFrontendBridge:
    """Adapter layer providing safe access to RobotMCP internals for the frontend."""

    def __init__(self):
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._history_sessions: Dict[str, Dict[str, Any]] = {}
        self._history_steps: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._history_order: Dict[str, List[str]] = defaultdict(list)
        self._history_lock = asyncio.Lock()
        self._history_snapshot_size = 0
        self._history_snapshot_marker: Optional[datetime] = None

    # Session helpers -----------------------------------------------------------------
    def _get_session(self, session_id: str) -> ExecutionSession:
        session = execution_engine.session_manager.get_session(session_id)
        if not session:
            raise KeyError(f"Session '{session_id}' not found")
        return session

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    def _deduplicate(self, items: List[str]) -> List[str]:
        seen: List[str] = []
        for item in items:
            if not item:
                continue
            if item not in seen:
                seen.append(item)
        return seen

    def _infer_metadata_from_history(self, session_id: str) -> Dict[str, Any]:
        mapping = {
            "Browser": ["open browser", "close browser", "go to", "new browser", "new page"],
            "SeleniumLibrary": ["click element", "input text", "press keys", "wait until element"],
            "RequestsLibrary": ["create session", "get on session", "post on session", "get request", "post request"],
            "Collections": ["create list", "append to list", "set to dictionary", "get from dictionary"],
            "BuiltIn": ["log", "evaluate", "set suite variable", "set test variable"],
            "AppiumLibrary": ["open application", "close application", "tap"],
        }

        libraries: set[str] = set()
        inferred_url: Optional[str] = None
        inferred_browser: Optional[str] = None

        steps = self._history_steps.get(session_id, {})
        if not steps:
            return {"libraries": [], "current_url": None, "browser_type": None}

        for step in steps.values():
            keyword = str(step.get("keyword", "")).lower()
            arguments = step.get("arguments") or []

            for lib, hints in mapping.items():
                for hint in hints:
                    if keyword == hint or hint in keyword:
                        libraries.add(lib)
                        break

            if inferred_browser is None:
                if keyword == "open browser" and len(arguments) >= 2:
                    inferred_browser = arguments[1]
                elif keyword == "new browser" and arguments:
                    inferred_browser = arguments[0]

            if inferred_url is None:
                candidate_args: List[str] = []
                if keyword in {"open browser", "go to", "go to url"}:
                    candidate_args.extend(arguments)
                if keyword == "new page" and arguments:
                    candidate_args.append(arguments[0])
                for arg in candidate_args:
                    if isinstance(arg, str) and arg.strip().lower().startswith(("http://", "https://")):
                        inferred_url = arg.strip()
                        break

        return {
            "libraries": list(libraries),
            "current_url": inferred_url,
            "browser_type": inferred_browser,
        }

    def _build_summary(self, session_id: str, session: ExecutionSession | None = None) -> Dict[str, Any]:
        libraries: List[str] = []
        if session:
            libraries.extend(session.imported_libraries)
            libraries.extend(getattr(session, "loaded_libraries", []))
            libraries.extend(getattr(session, "search_order", []))

        history = self._history_sessions.get(session_id)
        if history:
            libraries.extend(history.get("imported_libraries", []))
            libraries.extend(history.get("loaded_libraries", []))

        inferred = self._infer_metadata_from_history(session_id)
        libraries.extend(inferred.get("libraries", []))
        libraries = self._deduplicate(libraries)

        summary: Dict[str, Any] = {
            "libraries": libraries,
        }

        if session:
            summary["active_library"] = session.get_active_library()
            summary["browser_type"] = session.browser_state.browser_type
            summary["current_url"] = session.browser_state.current_url
        elif history:
            summary["active_library"] = history.get("active_library")
            summary["browser_type"] = history.get("browser_type")
            summary["current_url"] = history.get("current_url")

        if inferred.get("browser_type") and not summary.get("browser_type"):
            summary["browser_type"] = inferred["browser_type"]
        if inferred.get("current_url") and not summary.get("current_url"):
            summary["current_url"] = inferred["current_url"]

        return summary

    async def _merge_application_state(self, detail: Dict[str, Any]) -> Dict[str, Any]:
        session_id = detail.get("session_id")
        summary = detail.get("summary", {}).copy()

        session = execution_engine.session_manager.get_session(session_id)
        base_summary = self._build_summary(session_id, session)
        base_summary.update({k: v for k, v in summary.items() if v})
        summary = base_summary

        try:
            state = await state_manager.get_state(
                state_type="dom",
                session_id=session_id,
                execution_engine=execution_engine,
            )
        except Exception:
            state = None

        if state and state.get("success", True):
            dom_state = state.get("dom") or {}
            browser_state = dom_state.get("browser_state") or {}
            url = dom_state.get("url") or browser_state.get("current_url")
            browser_type = browser_state.get("browser_type") or dom_state.get("browser")
            if url and not summary.get("current_url"):
                summary["current_url"] = url
            if browser_type and not summary.get("browser_type"):
                summary["browser_type"] = browser_type

            # If the state manager tracks active library in variables
            state_vars = state.get("variables") or {}
            if not summary.get("active_library"):
                candidate = state_vars.get("ACTIVE_LIBRARY") or state_vars.get("active_library")
                if candidate:
                    summary["active_library"] = candidate

        if not summary.get("active_library"):
            for candidate in summary.get("libraries", []):
                if candidate.lower().endswith("library") or candidate in {"Browser", "BuiltIn", "Collections"}:
                    summary["active_library"] = candidate
                    break

        detail["summary"] = summary

        if summary.get("current_url"):
            detail["current_url"] = summary["current_url"]
            detail.setdefault("browser_state", {})["current_url"] = summary["current_url"]
        if summary.get("browser_type"):
            detail["browser_type"] = summary["browser_type"]
            detail.setdefault("browser_state", {})["browser_type"] = summary["browser_type"]
        if summary.get("active_library"):
            detail["active_library"] = summary["active_library"]
            detail.setdefault("browser_state", {})["active_library"] = summary["active_library"]
        if summary.get("libraries"):
            combined_libs = detail.get("imported_libraries", []) + summary["libraries"]
            detail["imported_libraries"] = self._deduplicate(combined_libs)

        return detail

    # Serialization -------------------------------------------------------------------
    @staticmethod
    def _serialize_session(session: ExecutionSession) -> Dict[str, Any]:
        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "duration": session.duration,
            "step_count": session.step_count,
            "imported_libraries": list(session.imported_libraries),
            "loaded_libraries": list(getattr(session, "loaded_libraries", [])),
            "search_order": list(getattr(session, "search_order", [])),
            "platform_type": session.platform_type.value,
            "is_mobile": session.is_mobile_session(),
            "browser_state": {
                "current_url": session.browser_state.current_url,
                "browser_type": session.browser_state.browser_type,
                "active_library": session.get_active_library(),
            },
            "current_url": session.browser_state.current_url,
            "browser_type": session.browser_state.browser_type,
            "active_library": session.get_active_library(),
            "summary": {},
            "status": "active",
            "is_active": True,
        }

    @staticmethod
    def _serialize_step(step: ExecutionStep) -> Dict[str, Any]:
        def _to_iso(dt: Optional[datetime]) -> Optional[str]:
            return dt.isoformat() if dt else None

        def _safe_value(value: Any) -> Any:
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, (list, tuple)):
                return [_safe_value(item) for item in value]
            if isinstance(value, dict):
                return {str(k): _safe_value(v) for k, v in value.items()}
            return str(value)

        return {
            "step_id": step.step_id,
            "keyword": step.keyword,
            "arguments": list(step.arguments),
            "status": step.status,
            "start_time": _to_iso(step.start_time),
            "end_time": _to_iso(step.end_time),
            "duration": step.execution_time,
            "error": step.error,
            "assigned_variables": list(step.assigned_variables),
            "assignment_type": step.assignment_type,
            "result": step.result,
            "variables": {str(k): _safe_value(v) for k, v in step.variables.items()},
        }

    # Public API ----------------------------------------------------------------------
    async def list_sessions(self) -> List[Dict[str, Any]]:
        await self._ensure_history()

        info = execution_engine.session_manager.get_all_sessions_info()
        sessions: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for session_id, payload in info.items():
            if not payload:
                continue
            combined = dict(payload)
            combined.setdefault("status", "active")
            combined["is_active"] = True
            history = self._history_sessions.get(session_id)
            if history:
                combined.setdefault("created_at", history["created_at"].isoformat())
                combined.setdefault("last_activity", history["last_activity"].isoformat())
                combined.setdefault("duration", history["duration"])
                combined.setdefault("step_count", history["step_count"])
            sessions.append(combined)
            seen.add(session_id)

        for session_id, history in self._history_sessions.items():
            if session_id in seen:
                continue
            sessions.append(self._serialize_history_session(history))

        sessions.sort(key=lambda s: s.get("last_activity", ""), reverse=True)
        return sessions

    async def get_session_details(self, session_id: str) -> Dict[str, Any]:
        await self._ensure_history()
        try:
            session = self._get_session(session_id)
            detail = self._serialize_session(session)
            return await self._merge_application_state(detail)
        except KeyError:
            history = self._history_sessions.get(session_id)
            if history:
                detail = self._serialize_history_session(history)
                return await self._merge_application_state(detail)
            raise

    async def get_session_steps(self, session_id: str) -> List[Dict[str, Any]]:
        await self._ensure_history()
        try:
            session = self._get_session(session_id)
            return [self._serialize_step(step) for step in session.steps]
        except KeyError:
            cached = self._get_cached_steps(session_id)
            if cached:
                return cached
            raise

    # Reconstruction helpers ---------------------------------------------------------
    async def _ensure_history(self) -> None:
        events = await event_bus.recent_events(limit=0)
        if not events:
            async with self._history_lock:
                self._history_sessions.clear()
                self._history_steps.clear()
                self._history_order.clear()
                self._history_snapshot_size = 0
                self._history_snapshot_marker = None
            return

        latest_marker = events[-1].timestamp
        if (
            self._history_snapshot_size == len(events)
            and self._history_snapshot_marker == latest_marker
        ):
            return

        async with self._history_lock:
            if (
                self._history_snapshot_size == len(events)
                and self._history_snapshot_marker == latest_marker
            ):
                return

            self._history_sessions = {}
            self._history_steps = defaultdict(dict)
            self._history_order = defaultdict(list)

            for event in sorted(events, key=lambda e: e.timestamp):
                self._ingest_event(event)

            self._history_snapshot_size = len(events)
            self._history_snapshot_marker = latest_marker

    def _ingest_event(self, event: FrontendEvent) -> None:
        session_id = event.session_id
        if not session_id:
            return

        timestamp = event.timestamp or datetime.now(timezone.utc)
        record = self._ensure_history_session(session_id, timestamp)

        if event.event_type == "session_created":
            record["is_active"] = True
            return

        if event.event_type == "session_removed":
            record["is_active"] = False
            return

        if event.step_id is None:
            return

        steps = self._history_steps[session_id]
        order = self._history_order[session_id]
        step = steps.get(event.step_id)

        if event.event_type == "step_started":
            if not step:
                step = {
                    "step_id": event.step_id,
                    "keyword": event.payload.get("keyword"),
                    "arguments": list(event.payload.get("arguments", [])),
                    "status": "running",
                    "start_time": timestamp,
                    "end_time": None,
                    "duration": 0.0,
                    "error": None,
                    "assigned_variables": [],
                    "assignment_type": None,
                    "result": None,
                    "variables": {},
                }
                steps[event.step_id] = step
                order.append(event.step_id)
            else:
                step.setdefault("keyword", event.payload.get("keyword"))
                step.setdefault(
                    "arguments", list(event.payload.get("arguments", []))
                )
                step["status"] = "running"
                step["start_time"] = step.get("start_time") or timestamp
            return

        if not step:
            step = {
                "step_id": event.step_id,
                "keyword": event.payload.get("keyword"),
                "arguments": list(event.payload.get("arguments", [])),
                "status": "running",
                "start_time": None,
                "end_time": None,
                "duration": 0.0,
                "error": None,
                "assigned_variables": [],
                "assignment_type": None,
                "result": None,
                "variables": {},
            }
            steps[event.step_id] = step
            order.append(event.step_id)

        if event.event_type == "step_completed":
            step["status"] = "pass"
            step["end_time"] = timestamp
            if step.get("start_time") and step.get("end_time"):
                start = step["start_time"]
                end = step["end_time"]
                if isinstance(start, datetime) and isinstance(end, datetime):
                    step["duration"] = max((end - start).total_seconds(), 0.0)
            if "result" in event.payload:
                step["result"] = event.payload.get("result")
            assigned_vars = event.payload.get("assigned_variables")
            if assigned_vars:
                step["assigned_variables"] = list(assigned_vars)
                step["assignment_type"] = event.payload.get("assignment_type")
            assigned_values = event.payload.get("assigned_values")
            if assigned_values:
                step["variables"] = dict(assigned_values)
        elif event.event_type == "step_failed":
            step["status"] = "fail"
            step["end_time"] = timestamp
            step["error"] = event.payload.get("error")

        record["step_count"] = len(order)

    def _ensure_history_session(
        self, session_id: str, timestamp: datetime
    ) -> Dict[str, Any]:
        record = self._history_sessions.get(session_id)
        if record is None:
            record = {
                "session_id": session_id,
                "created_at": timestamp,
                "last_activity": timestamp,
                "duration": 0.0,
                "step_count": 0,
                "imported_libraries": [],
                "loaded_libraries": [],
                "platform_type": "web",
                "is_mobile": False,
                "browser_state": {
                    "current_url": None,
                    "browser_type": None,
                    "active_library": None,
                },
                "is_active": False,
            }
            self._history_sessions[session_id] = record
        else:
            if timestamp < record["created_at"]:
                record["created_at"] = timestamp

        if timestamp > record["last_activity"]:
            record["last_activity"] = timestamp
            record["duration"] = max(
                (record["last_activity"] - record["created_at"]).total_seconds(),
                0.0,
            )
        return record

    def _serialize_history_session(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "session_id": record["session_id"],
            "created_at": record["created_at"].isoformat(),
            "last_activity": record["last_activity"].isoformat(),
            "duration": record["duration"],
            "step_count": record["step_count"],
            "imported_libraries": list(record.get("imported_libraries", [])),
            "loaded_libraries": list(record.get("loaded_libraries", [])),
            "platform_type": record.get("platform_type", "web"),
            "is_mobile": record.get("is_mobile", False),
            "browser_state": dict(record.get("browser_state", {})),
            "current_url": record.get("browser_state", {}).get("current_url"),
            "browser_type": record.get("browser_state", {}).get("browser_type"),
            "active_library": record.get("browser_state", {}).get("active_library"),
            "summary": {},
            "status": "active" if record.get("is_active") else "archived",
            "is_active": bool(record.get("is_active")),
        }

    def _serialize_history_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        def _iso(value: Optional[datetime]) -> Optional[str]:
            return value.isoformat() if isinstance(value, datetime) else None

        return {
            "step_id": step.get("step_id"),
            "keyword": step.get("keyword"),
            "arguments": list(step.get("arguments", [])),
            "status": step.get("status", "running"),
            "start_time": _iso(step.get("start_time")),
            "end_time": _iso(step.get("end_time")),
            "duration": step.get("duration", 0.0),
            "error": step.get("error"),
            "assigned_variables": list(step.get("assigned_variables", [])),
            "assignment_type": step.get("assignment_type"),
            "result": step.get("result"),
            "variables": dict(step.get("variables", {})),
        }

    def _get_cached_steps(self, session_id: str) -> List[Dict[str, Any]]:
        steps = self._history_steps.get(session_id)
        if not steps:
            return []
        order = self._history_order.get(session_id) or list(steps.keys())
        return [
            self._serialize_history_step(steps[step_id])
            for step_id in order
            if step_id in steps
        ]

    async def get_session_variables(self, session_id: str) -> Dict[str, Any]:
        try:
            session = self._get_session(session_id)
        except KeyError:
            return {}
        result: Dict[str, Any] = {}
        for key, value in session.variables.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[key] = value
            else:
                result[key] = f"<{type(value).__name__}>"
        return result

    async def get_application_state(
        self, session_id: str, state_type: str = "all"
    ) -> Dict[str, Any]:
        return await state_manager.get_state(
            state_type=state_type,
            session_id=session_id,
            execution_engine=execution_engine,
        )

    async def get_suite_preview(
        self, session_id: str, overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        lock = self._get_lock(session_id)
        async with lock:
            await self._ensure_history()
            session = execution_engine.session_manager.get_session(session_id)
            temporary_session = False

            if session is None:
                cached_steps = self._get_cached_steps(session_id)
                if not cached_steps:
                    return {
                        "success": False,
                        "error": f"Session '{session_id}' not found",
                        "suite": None,
                    }
                session = ExecutionSession(session_id=session_id)
                session.libraries_loaded = True
                history = self._history_sessions.get(session_id)
                if history:
                    session.created_at = history["created_at"]
                    session.last_activity = history["last_activity"]
                session.steps = [
                    self._hydrate_execution_step(step) for step in cached_steps
                ]
                session._session_manager = execution_engine.session_manager  # type: ignore[attr-defined]
                execution_engine.session_manager.sessions[session_id] = session
                temporary_session = True

            original_steps = list(session.steps)
            applied: Dict[str, Any] = {}
            if overrides:
                step_map = {step.step_id: step for step in original_steps}
                order = overrides.get("order")
                excluded = set(overrides.get("excluded", []))
                edits_input = overrides.get("edits", []) or []
                edits_map = {
                    edit.get("step_id"): edit
                    for edit in edits_input
                    if edit.get("step_id") in step_map
                }

                order_sequence = order or [step.step_id for step in original_steps]
                updated_steps = []
                for sid in order_sequence:
                    base = step_map.get(sid)
                    if base is None or sid in excluded:
                        continue
                    step_copy = self._clone_execution_step(base)
                    edit_data = edits_map.get(sid)
                    if edit_data:
                        if edit_data.get("keyword"):
                            step_copy.keyword = edit_data["keyword"]
                        if "arguments" in edit_data and edit_data["arguments"] is not None:
                            step_copy.arguments = list(edit_data["arguments"])
                        if "assigned_variables" in edit_data and edit_data["assigned_variables"] is not None:
                            assigned_override = edit_data["assigned_variables"]
                            if isinstance(assigned_override, list):
                                step_copy.assigned_variables = list(assigned_override)
                            elif isinstance(assigned_override, dict):
                                step_copy.assigned_variables = list(assigned_override.keys())
                    updated_steps.append(step_copy)

                session.steps = updated_steps
                applied = {
                    "order": order_sequence,
                    "excluded": list(excluded),
                }
                if edits_map:
                    applied["edited"] = list(edits_map.keys())

            try:
                preview = await test_builder.build_suite(
                    session_id=session_id,
                    remove_library_prefixes=False,
                )
            finally:
                session.steps = original_steps
                if temporary_session:
                    session.cleanup()
                    execution_engine.session_manager.sessions.pop(session_id, None)

        if overrides:
            preview.setdefault("metadata", {})
            preview["metadata"]["applied_overrides"] = applied
        return preview

    @staticmethod
    def _clone_execution_step(step: ExecutionStep) -> ExecutionStep:
        cloned = ExecutionStep(
            step_id=step.step_id,
            keyword=step.keyword,
            arguments=list(step.arguments),
        )
        cloned.status = step.status
        cloned.start_time = step.start_time
        cloned.end_time = step.end_time
        cloned.error = step.error
        cloned.result = step.result
        cloned.assigned_variables = list(step.assigned_variables)
        cloned.assignment_type = step.assignment_type
        cloned.variables = dict(step.variables)
        return cloned

    @staticmethod
    def _hydrate_execution_step(step: Dict[str, Any]) -> ExecutionStep:
        execution_step = ExecutionStep(
            step_id=step.get("step_id", ""),
            keyword=step.get("keyword", ""),
            arguments=list(step.get("arguments", [])),
        )
        execution_step.status = step.get("status", "pass")
        start_time = step.get("start_time")
        end_time = step.get("end_time")
        if start_time:
            execution_step.start_time = datetime.fromisoformat(start_time)
        if end_time:
            execution_step.end_time = datetime.fromisoformat(end_time)
        execution_step.error = step.get("error")
        execution_step.assigned_variables = list(step.get("assigned_variables", []))
        execution_step.assignment_type = step.get("assignment_type")
        execution_step.variables = dict(step.get("variables", {}))
        return execution_step

    async def execute_keyword(
        self,
        session_id: str,
        keyword: str,
        arguments: Optional[List[str]] = None,
        use_context: bool = True,
    ) -> Dict[str, Any]:
        if arguments is None:
            arguments = []
        lock = self._get_lock(session_id)
        async with lock:
            result = await execution_engine.execute_step(
                keyword,
                arguments=arguments,
                session_id=session_id,
                detail_level="standard",
                use_context=use_context,
            )
        if result.get("success"):
            event_bus.publish_sync(
                FrontendEvent(
                    event_type="execute_step",
                    session_id=session_id,
                    payload={"keyword": keyword, "arguments": arguments},
                )
            )
        return result

    async def stream_events(self):
        async for event in event_bus.subscribe():
            yield event


# Singleton bridge used across views
bridge = McpFrontendBridge()
