from __future__ import annotations

import json
import http.client
from typing import Any, Dict, List, Optional


class ExternalRFClient:
    """Minimal client for the McpAttach bridge.

    This adaptor allows RobotMCP code to call into a running RF process
    that has imported the `McpAttach` library and started `MCP Serve`.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7317, token: str = "change-me") -> None:
        self.host = host
        self.port = int(port)
        self.token = token

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "X-MCP-Token": self.token,
        }
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
            conn.request("POST", path, body, headers)
            resp = conn.getresponse()
            data = resp.read()
            try:
                return json.loads(data.decode("utf-8"))
            except Exception:
                return {"success": False, "error": f"invalid response: {data!r}"}
            finally:
                conn.close()
        except Exception as e:
            return {"success": False, "error": f"connection error: {e}"}

    def diagnostics(self) -> Dict[str, Any]:
        return self._post("/diagnostics", {})

    def stop(self) -> Dict[str, Any]:
        return self._post("/stop", {})

    def run_keyword(
        self, name: str, args: Optional[List[str]] = None, assign_to: Optional[str | List[str]] = None,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"name": name, "args": list(args or [])}
        if assign_to is not None:
            payload["assign_to"] = assign_to
        if timeout_ms is not None and timeout_ms > 0:
            payload["timeout_ms"] = timeout_ms
        return self._post("/run_keyword", payload)

    def import_library(self, name_or_path: str, args: Optional[List[str]] = None, alias: Optional[str] = None) -> Dict[str, Any]:
        return self._post(
            "/import_library",
            {"name_or_path": name_or_path, "args": list(args or []), "alias": alias},
        )

    def import_resource(self, path: str) -> Dict[str, Any]:
        return self._post("/import_resource", {"path": path})

    def import_variables(self, variable_file_path: str, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """Import variables from a file via the attach bridge.
        
        Args:
            variable_file_path: Path to variable file (.py, .yaml, .json)
            args: Optional arguments for Python variable files
            
        Returns:
            Dict with success status and variable loading results
        """
        return self._post("/import_variables", {
            "variable_file_path": variable_file_path, 
            "args": list(args or [])
        })

    def list_keywords(self) -> Dict[str, Any]:
        return self._post("/list_keywords", {})

    def get_keyword_doc(self, name: str) -> Dict[str, Any]:
        return self._post("/get_keyword_doc", {"name": name})

    def get_variables(self, names: Optional[List[str]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if names is not None:
            payload["names"] = names
        return self._post("/get_variables", payload)

    def set_variable(self, name: str, value: Any, scope: str = "test") -> Dict[str, Any]:
        return self._post("/set_variable", {"name": name, "value": value, "scope": scope})

    def get_page_source(self) -> Dict[str, Any]:
        """Get page source directly from bridge.

        Automatically detects Browser Library vs SeleniumLibrary and uses
        the appropriate keyword.

        Returns:
            Dict with success status and page source content, including
            which library was used (Browser, SeleniumLibrary, or AppiumLibrary)
        """
        return self._post("/get_page_source", {})

    def get_aria_snapshot(
        self,
        selector: str = "css=html",
        format_type: str = "yaml"
    ) -> Dict[str, Any]:
        """Get ARIA accessibility tree snapshot from bridge.

        Only available with Browser Library (Playwright).

        Args:
            selector: CSS selector for the element to snapshot (default: "css=html")
            format_type: Output format - "yaml" or "json" (default: "yaml")

        Returns:
            Dict with success status and ARIA snapshot content
        """
        return self._post("/get_aria_snapshot", {
            "selector": selector,
            "format": format_type,
        })

    def get_session_info(self) -> Dict[str, Any]:
        """Get RF execution context information from bridge.

        Returns:
            Dict with success status and session info including:
            - context_active: Whether RF context is active
            - variable_count: Number of variables defined
            - suite_name: Current test suite name
            - test_name: Current test case name
            - libraries: List of loaded library names
        """
        return self._post("/get_session_info", {})
