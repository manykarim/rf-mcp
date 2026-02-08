"""Development server for the RobotMCP Django frontend."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime
from typing import Iterable, Sequence

from robotmcp.frontend.config import FrontendConfig, build_frontend_config
from robotmcp.frontend.django_app import get_django_application
from robotmcp.models.execution_models import ExecutionStep
from robotmcp.server import execution_engine
from robotmcp.core.event_bus import FrontendEvent, event_bus

logger = logging.getLogger(__name__)


def _create_sample_session(session_id: str = "frontend-demo") -> None:
    """Populate the execution engine with a sample session for UI exploration."""

    session_manager = execution_engine.session_manager
    if session_manager.get_session(session_id):
        return

    session = session_manager.create_session(session_id)
    event_bus.publish_sync(
        FrontendEvent(
            event_type="session_created",
            session_id=session_id,
            payload={"session_id": session_id},
        )
    )
    session.imported_libraries.extend(["BuiltIn", "Collections", "Browser"])
    session.loaded_libraries.update(["BuiltIn", "Collections", "Browser"])
    session.search_order = ["Browser", "BuiltIn", "Collections", "String"]
    session.variables.update({"CITY": "Helsinki", "ENV": "demo"})
    session.browser_state.browser_type = "chromium"
    session.browser_state.current_url = "https://robotframework.org/"
    session.browser_state.active_library = "browser"

    steps: Sequence[tuple[str, Iterable[str]]] = [
        ("Open Browser", ["https://robotframework.org", "chromium"]),
        ("Go To", ["https://robotframework.org/robotframework/"]),
        ("Create List", ["test1", "test2"]),
        ("Set Suite Variable", ["CITY", "Helsinki"]),
        ("Log", ["Frontend demo ready"]),
    ]
    for idx, (keyword, arguments) in enumerate(steps, start=1):
        step_id = f"{session_id}-step-{idx}"
        step = ExecutionStep(
            step_id=step_id,
            keyword=keyword,
            arguments=list(arguments),
        )
        step.mark_running()
        event_bus.publish_sync(
            FrontendEvent(
                event_type="step_started",
                session_id=session_id,
                step_id=step_id,
                payload={"keyword": keyword, "arguments": list(arguments)},
            )
        )
        step.mark_success(result="OK")
        session.steps.append(step)
        event_bus.publish_sync(
            FrontendEvent(
                event_type="step_completed",
                session_id=session_id,
                step_id=step_id,
                payload={
                    "status": "pass",
                    "keyword": keyword,
                    "arguments": list(arguments),
                },
            )
        )

    session.last_activity = datetime.now()
    logger.info("Created sample session '%s' with %d steps", session_id, len(steps))


async def _serve(config: FrontendConfig) -> None:
    """Run uvicorn with the configured Django ASGI application."""

    import uvicorn

    application = get_django_application(config)
    uvicorn_config = uvicorn.Config(
        application,
        host=config.host,
        port=config.port,
        log_level="info" if config.debug else "warning",
        access_log=config.debug,
        lifespan="auto",
    )
    server = uvicorn.Server(config=uvicorn_config)
    await server.serve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RobotMCP frontend for development and manual testing."
    )
    parser.add_argument("--host", dest="host", help="Host interface (default 127.0.0.1)")
    parser.add_argument("--port", dest="port", type=int, help="Port (default 8051)")
    parser.add_argument(
        "--base-path",
        dest="base_path",
        help="Base path prefix for serving the frontend (default '/')",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Enable Django debug mode for the frontend.",
    )
    parser.add_argument(
        "--no-sample-data",
        dest="sample_data",
        action="store_false",
        default=True,
        help="Skip creating demo session data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    config = build_frontend_config(
        enabled=True,
        host=args.host,
        port=args.port,
        base_path=args.base_path,
        debug=args.debug,
    )

    if args.sample_data:
        _create_sample_session()

    logger.info("Starting RobotMCP frontend devserver at %s", config.url)

    try:
        asyncio.run(_serve(config))
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown
        logger.info("Frontend devserver stopped by user")


if __name__ == "__main__":
    main()
