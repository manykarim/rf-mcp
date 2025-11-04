"""Background controller for serving the Django frontend with uvicorn."""

from __future__ import annotations

import asyncio
import logging
import threading

from .config import FrontendConfig
from .django_app import get_django_application

logger = logging.getLogger(__name__)


class FrontendServerController:
    """Manage lifecycle of the uvicorn server that hosts the Django frontend."""

    def __init__(self, config: FrontendConfig):
        self.config = config
        self._server = None
        self._thread: threading.Thread | None = None

    async def start(self) -> None:
        """Start the uvicorn server in the background."""

        if self._thread and self._thread.is_alive():
            return

        application = get_django_application(self.config)

        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "uvicorn is required to run the RobotMCP frontend. "
                "Install with 'pip install rf-mcp[frontend]'."
            ) from exc

        config = uvicorn.Config(
            application,
            host=self.config.host,
            port=self.config.port,
            log_level="info" if self.config.debug else "warning",
            access_log=self.config.debug,
            lifespan="auto",
        )
        self._server = uvicorn.Server(config=config)

        def _run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._server.serve())
            except Exception:
                logger.exception("Frontend server crashed unexpectedly")
            finally:
                loop.close()

        self._thread = threading.Thread(
            target=_run_server, name="robotmcp-frontend-server", daemon=True
        )
        self._thread.start()

        waited = 0.0
        while not self._server.started:
            if self._thread and not self._thread.is_alive():
                raise RuntimeError("Frontend server terminated before startup")
            await asyncio.sleep(0.05)
            waited += 0.05
            if waited > 10:
                raise TimeoutError("Frontend server did not start within 10 seconds")

        logger.info(
            "RobotMCP frontend running at http://%s:%s%s",
            self.config.host,
            self.config.port,
            self.config.base_path,
        )

    async def stop(self) -> None:
        """Stop the uvicorn server if it is running."""

        if not self._server:
            return

        self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        self._thread = None
        self._server = None

    @property
    def url(self) -> str:
        """Return the accessible URL for the frontend."""

        return f"http://{self.config.host}:{self.config.port}{self.config.base_path}"
