"""HTTP API views exposed to the frontend."""

from __future__ import annotations

import json
from typing import Any, Dict

from asgiref.sync import async_to_sync
import asyncio
import logging

from django.http import (
    Http404,
    HttpRequest,
    HttpResponse,
    JsonResponse,
    StreamingHttpResponse,
)
from django.views.decorators.http import require_http_methods

from .bridge import bridge
from robotmcp.core.event_bus import event_bus, FrontendEvent


def _json_response(data: Dict[str, Any], status: int = 200) -> JsonResponse:
    return JsonResponse(data, status=status, safe=False)


@require_http_methods(["GET"])
def sessions_list(request: HttpRequest) -> JsonResponse:
    sessions = async_to_sync(bridge.list_sessions)()
    return _json_response({"sessions": sessions})


@require_http_methods(["GET"])
def session_detail(request: HttpRequest, session_id: str) -> JsonResponse:
    try:
        data = async_to_sync(bridge.get_session_details)(session_id)
    except KeyError as exc:
        return _json_response({"error": str(exc)}, status=404)
    return _json_response(data)


@require_http_methods(["GET"])
def session_steps(request: HttpRequest, session_id: str) -> JsonResponse:
    try:
        steps = async_to_sync(bridge.get_session_steps)(session_id)
    except KeyError as exc:
        return _json_response({"steps": [], "error": str(exc)}, status=200)
    return _json_response({"steps": steps})


@require_http_methods(["GET"])
def session_variables(request: HttpRequest, session_id: str) -> JsonResponse:
    try:
        variables = async_to_sync(bridge.get_session_variables)(session_id)
    except KeyError as exc:
        return _json_response({"variables": {}, "error": str(exc)}, status=200)
    return _json_response({"variables": variables})


@require_http_methods(["GET"])
def session_state(request: HttpRequest, session_id: str) -> JsonResponse:
    state_type = request.GET.get("type", "all")
    state = async_to_sync(bridge.get_application_state)(session_id, state_type)
    if not state.get("success", True):
        return _json_response(state, status=500)
    return _json_response(state)


@require_http_methods(["GET", "POST"])
def suite_preview(request: HttpRequest, session_id: str) -> JsonResponse:
    overrides = None
    if request.method == "POST":
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return _json_response(
                {"success": False, "error": "Invalid JSON body"}, status=400
            )
        if not isinstance(payload, dict):
            return _json_response(
                {"success": False, "error": "Payload must be a JSON object"}, status=400
            )
        overrides = {
            "order": payload.get("order"),
            "excluded": payload.get("excluded", []),
            "edits": payload.get("edits") or [],
        }

    preview = async_to_sync(bridge.get_suite_preview)(session_id, overrides=overrides)
    if not preview.get("success", False):
        status_code = 200 if preview.get("error") else 500
        return _json_response(preview, status=status_code)
    return _json_response(preview)


@require_http_methods(["POST"])
def execute_keyword(request: HttpRequest, session_id: str) -> HttpResponse:
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return _json_response({"success": False, "error": "Invalid JSON body"}, status=400)

    keyword = payload.get("keyword")
    arguments = payload.get("arguments") or []
    use_context = payload.get("use_context", True)

    if not keyword or not isinstance(keyword, str):
        return _json_response(
            {"success": False, "error": "Keyword must be provided as a string"}, status=400
        )

    try:
        result = async_to_sync(bridge.execute_keyword)(
            session_id, keyword, arguments=arguments, use_context=use_context
        )
    except KeyError as exc:
        raise Http404(str(exc)) from exc

    status_code = 200 if result.get("success", False) else 400
    return _json_response(result, status=status_code)


@require_http_methods(["GET"])
def events_stream(request: HttpRequest) -> StreamingHttpResponse:
    import queue
    import threading

    event_queue: queue.Queue[str | None] = queue.Queue()
    stop_flag = threading.Event()

    async def consume_events() -> None:
        try:
            # Send connection-open event
            import logging

            logging.getLogger(__name__).debug("events_stream: connection opened")
            event_queue.put_nowait("event: ping\ndata: {}\n\n")
            async for event in event_bus.subscribe():
                if stop_flag.is_set():
                    break
                payload = {
                    "event_type": event.event_type,
                    "session_id": event.session_id,
                    "step_id": event.step_id,
                    "payload": event.payload,
                    "timestamp": event.timestamp.isoformat(),
                }
                event_queue.put_nowait(f"data: {json.dumps(payload)}\n\n")
        finally:
            event_queue.put_nowait(None)

    def worker() -> None:
        try:
            asyncio.run(consume_events())
        except Exception:
            logging.getLogger(__name__).exception("events_stream worker crashed")
            event_queue.put_nowait(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        try:
            while True:
                chunk = event_queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            stop_flag.set()

    response = StreamingHttpResponse(stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@require_http_methods(["GET"])
def recent_events(request: HttpRequest) -> JsonResponse:
    limit = request.GET.get("limit")
    try:
        limit_int = int(limit) if limit is not None else 50
    except ValueError:
        limit_int = 50
    try:
        events = asyncio.run(event_bus.recent_events(limit=limit_int))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            events = loop.run_until_complete(event_bus.recent_events(limit=limit_int))
        finally:
            loop.close()
    payload = [
        {
            "event_type": event.event_type,
            "session_id": event.session_id,
            "step_id": event.step_id,
            "payload": event.payload,
            "timestamp": event.timestamp.isoformat(),
        }
        for event in events
    ]
    return _json_response({"events": payload})
