"""HTTP bridge that lets external clients drive Robot Framework sessions.

The attach bridge exposes a tiny REST interface that forwards commands
into the currently executing Robot Framework context.  It is primarily
used by the MCP server when running in *attach* mode, but it can also
be imported and controlled directly from Robot Framework test cases.

Examples:
| *** Settings ***
| Library    robotmcp.attach.McpAttach    token=${MCP_TOKEN}
|
| *** Test Cases ***
| Serve External Commands
| MCP Serve    mode=step
| FOR    ${index}    IN RANGE    0    10
| ...    MCP Process Once
| ...    Sleep    0.1 seconds
| END
| MCP Stop

The keywords provided by this module follow the same conventions as
other Robot Framework libraries: keyword names use spaces and the
docstrings below document arguments and return values.
"""

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional, Tuple

from robot.libraries.BuiltIn import BuiltIn
from robot.running.context import EXECUTION_CONTEXTS


class _Command:
    def __init__(
        self, verb: str, payload: Dict[str, Any], replyq: "queue.Queue"
    ) -> None:
        self.verb = verb
        self.payload = payload
        self.replyq = replyq


class _Server(threading.Thread):
    def __init__(self, host: str, port: int, token: str, cmdq: "queue.Queue") -> None:
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.token = token
        self.cmdq = cmdq
        self.httpd: Optional[HTTPServer] = None

    def run(self) -> None:
        token = self.token
        cmdq = self.cmdq

        class Handler(BaseHTTPRequestHandler):
            def _auth(self) -> bool:
                return self.headers.get("X-MCP-Token") == token

            def _bad(self, code: int, msg: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                body = json.dumps({"success": False, "error": msg}).encode("utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> Dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except Exception:
                    length = 0
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    return json.loads(raw.decode("utf-8") or "{}")
                except Exception:
                    return {}

            def do_POST(self) -> None:  # noqa: N802 (RF env; stdlib signature)
                if not self._auth():
                    self._bad(401, "unauthorized")
                    return
                verb = (self.path or "/").strip("/")
                payload = self._read_json()
                replyq: "queue.Queue" = queue.Queue()
                cmdq.put(_Command(verb, payload, replyq))
                try:
                    # Use timeout from payload if provided, else default 120s
                    cmd_timeout = payload.get("timeout_ms")
                    if cmd_timeout and isinstance(cmd_timeout, (int, float)) and cmd_timeout > 0:
                        effective_timeout = cmd_timeout / 1000.0
                    else:
                        effective_timeout = 120.0
                    resp = replyq.get(timeout=effective_timeout)
                except queue.Empty:
                    resp = {"success": False, "error": "timeout"}
                try:
                    body = json.dumps(resp).encode("utf-8")
                except (TypeError, ValueError) as e:
                    body = json.dumps({
                        "success": False,
                        "error": f"Response serialization failed: {type(e).__name__}: {str(e)[:200]}"
                    }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(
                self, format: str, *args: Any
            ) -> None:  # silence console spam
                return

        self.httpd = HTTPServer((self.host, self.port), Handler)
        self.httpd.serve_forever()


class McpAttach:
    """Robot Framework library that exposes an HTTP bridge for MCP attach mode.

    The library spins up a small HTTP server that accepts authenticated
    JSON requests and forwards them to the currently active Robot Framework
    execution context.  External tools can use this bridge to execute
    keywords, import resources, or inspect variables during a live run.

    Examples:
    | *** Settings ***
    | Library    robotmcp.attach.McpAttach    host=0.0.0.0    port=7317    token=${TOKEN}
    |
    | *** Test Cases ***
    | Serve In Background
    | Start Process    python    -m    my_agent    --attach    stdout=NONE
    | MCP Serve    mode=blocking

    The library exposes the following keywords:

    - ``MCP Serve`` / ``MCP Start`` – start the HTTP server and process
      incoming commands.
    - ``MCP Process Once`` – process at most one pending command and
      return immediately (useful for custom loops).
    - ``MCP Stop`` – request the processing loop to stop.
    """

    from importlib import metadata

    try:
        ROBOT_LIBRARY_VERSION = metadata.version("rf-mcp")
    except metadata.PackageNotFoundError:
        ROBOT_LIBRARY_VERSION = "0.1"
        pass

    def __init__(
        self, host: str = "127.0.0.1", port: int = 7317, token: str = "change-me"
    ) -> None:
        self.host = host
        self.port = int(port)
        self.token = token
        self._cmdq: "queue.Queue" = queue.Queue()
        self._srv: Optional[_Server] = None
        self._stop_flag = False

    # --- Public Library Keywords ---
    def MCP_Serve(
        self,
        port: Optional[int] = None,
        token: Optional[str] = None,
        mode: str = "blocking",
        poll_ms: int = 100,
    ) -> None:
        """Start the bridge server and process incoming commands.

        Args:
            port: Optional port override.  Defaults to the library level port.
            token: Optional token override.  Defaults to the library level token.
            mode: ``"blocking"`` keeps looping until ``MCP Stop`` or the ``/stop``
                HTTP command is received. ``"step"`` processes exactly one command
                and returns immediately.
            poll_ms: Polling interval in milliseconds while in blocking mode.

        Examples:
        | MCP Serve    mode=blocking
        | MCP Serve    port=7320    token=${TOKEN}    mode=step
        """
        if self._srv is None:
            self._srv = _Server(
                self.host, int(port or self.port), str(token or self.token), self._cmdq
            )
            self._srv.start()
            try:
                BuiltIn().log_to_console(
                    f"[MCP] Bridge server started on http://{self.host}:{int(port or self.port)}"
                )
            except Exception:
                pass

        if str(mode).lower() == "step":
            try:
                BuiltIn().log_to_console(
                    "[MCP] Processing one external command (mode=step)."
                )
                BuiltIn().log_to_console(
                    "[MCP] Tip: Use 'attach_stop_bridge' from RobotMCP or POST /stop to end blocking serve."
                )
            except Exception:
                pass
            self.MCP_Process_Once()
            return

        self._stop_flag = False
        try:
            BuiltIn().log_to_console(
                "[MCP] Bridge active (mode=blocking). Waiting for external commands..."
            )
            BuiltIn().log_to_console(
                "[MCP] To stop: call RobotMCP tool 'attach_stop_bridge' or POST /stop to the bridge."
            )
        except Exception:
            pass
        while not self._stop_flag:
            self.MCP_Process_Once()
            time.sleep(max(0.0, int(poll_ms) / 1000.0))

    # Backwards-compatible alias with improved naming
    def MCP_Start(
        self,
        port: Optional[int] = None,
        token: Optional[str] = None,
        mode: str = "blocking",
        poll_ms: int = 100,
    ) -> None:
        """Start the MCP attach bridge (alias for :keyword:`MCP Serve`).

        This keyword exists for backwards compatibility with older
        suites that used ``MCP Start`` instead of ``MCP Serve``.

        Examples:
        | MCP Start    mode=blocking
        """
        try:
            BuiltIn().log_to_console("[MCP] MCP Start invoked (alias of MCP Serve).")
        except Exception:
            pass
        return self.MCP_Serve(port=port, token=token, mode=mode, poll_ms=poll_ms)

    def MCP_Process_Once(self) -> None:
        """Process at most one pending command and return immediately.

        Useful when the test suite wants to interleave custom logic with
        bridge commands instead of yielding completely to ``MCP Serve``.

        Example:
        | FOR    ${index}    IN RANGE    0    5
        | ...    MCP Process Once
        | ...    Sleep    0.2 seconds
        | END
        """
        try:
            cmd: _Command = self._cmdq.get_nowait()
        except queue.Empty:
            return
        try:
            resp = self._execute_command(cmd.verb, cmd.payload)
        except Exception as e:  # pragma: no cover - defensive
            resp = {"success": False, "error": str(e)}
        cmd.replyq.put(resp)

    def MCP_Stop(self) -> None:
        """Request the serve loop to stop.

        Sets an internal flag that causes the next iteration of
        ``MCP Serve`` to exit gracefully.  External clients can also
        trigger the same behaviour by POSTing ``/stop`` to the bridge.

        Example:
        | MCP Stop
        """
        self._stop_flag = True

    # --- Internal command execution on RF thread ---
    def _execute_command(self, verb: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if verb == "stop":
            self._stop_flag = True
            try:
                BuiltIn().log_to_console("[MCP] Bridge stopping; returning to test.")
            except Exception:
                pass
            return {"success": True}
        if verb == "diagnostics":
            return self._diagnostics()
        if verb == "run_keyword":
            name, args, assign_to = self._parse_run_keyword_payload(payload)
            return self._run_keyword(name, args, assign_to)
        if verb == "import_library":
            return self._import_library(payload)
        if verb == "import_resource":
            return self._import_resource(payload)
        if verb == "list_keywords":
            return self._list_keywords()
        if verb == "get_keyword_doc":
            return self._get_keyword_doc(payload)
        if verb == "get_variables":
            return self._get_variables(payload)
        if verb == "set_variable":
            return self._set_variable(payload)
        if verb == "import_variables":
            return self._import_variables(payload)
        if verb == "get_page_source":
            return self._get_page_source(payload)
        if verb == "get_aria_snapshot":
            return self._get_aria_snapshot(payload)
        if verb == "get_session_info":
            return self._get_session_info(payload)
        return {"success": False, "error": f"unknown verb: {verb}"}

    def _diagnostics(self) -> Dict[str, Any]:
        ctx = EXECUTION_CONTEXTS.current
        libs = []
        if (
            ctx
            and getattr(ctx, "namespace", None)
            and hasattr(ctx.namespace, "libraries")
        ):
            libraries = ctx.namespace.libraries
            if hasattr(libraries, "keys"):
                libs = list(libraries.keys())
            elif hasattr(libraries, "__iter__"):
                libs = [
                    getattr(li, "name", getattr(li, "__class__", type(li)).__name__)
                    for li in libraries
                ]
        return {"success": True, "result": {"libraries": libs, "context": bool(ctx)}}

    def _parse_run_keyword_payload(
        self, payload: Dict[str, Any]
    ) -> Tuple[str, list, Optional[Any]]:
        name = str(payload.get("name", ""))
        args = payload.get("args", []) or []
        assign_to = payload.get("assign_to")
        if not isinstance(args, list):
            raise ValueError("args must be a list of strings")
        return name, args, assign_to

    def _run_keyword(
        self, name: str, args: list, assign_to: Optional[Any]
    ) -> Dict[str, Any]:
        bi = BuiltIn()
        result: Any
        result = bi.run_keyword(name, *args)
        assigned = {}
        if assign_to:
            assigned = self._assign_variables(assign_to, result)
        # Sanitize result and assigned values for JSON serialization
        return {
            "success": True,
            "result": self._sanitize_for_json(result),
            "assigned": self._sanitize_for_json(assigned),
        }

    def _assign_variables(self, names: Any, value: Any) -> Dict[str, Any]:
        bi = BuiltIn()
        assigned: Dict[str, Any] = {}
        if isinstance(names, list):
            # If exactly one target name, assign the whole value to that variable
            if len(names) == 1:
                n = self._norm_var(names[0])
                bi.set_test_variable(n, value)
                assigned[n] = value
            else:
                # Multiple targets: element-wise assignment, missing values -> None
                if isinstance(value, (list, tuple)):
                    for i, n in enumerate(names):
                        v = value[i] if i < len(value) else None
                        bi.set_test_variable(self._norm_var(n), v)
                        assigned[self._norm_var(n)] = v
                else:
                    # Single scalar across multiple names: assign to first, None to rest
                    for i, n in enumerate(names):
                        v = value if i == 0 else None
                        bi.set_test_variable(self._norm_var(n), v)
                        assigned[self._norm_var(n)] = v
        else:
            n = self._norm_var(str(names))
            bi.set_test_variable(n, value)
            assigned[n] = value
        return assigned

    def _norm_var(self, name: str) -> str:
        s = str(name)
        if not (s.startswith("${") and s.endswith("}")):
            return f"${{{s}}}"
        return s

    def _sanitize_for_json(self, val: Any) -> Any:
        """Convert a value to a JSON-serializable form.

        - None, str, int, float, bool are returned as-is
        - Path objects (anything with __fspath__) are converted to str
        - Lists/tuples are recursively sanitized
        - Dicts are recursively sanitized (keys converted to str)
        - Other objects are converted to str(val) to preserve useful info
        """
        if val is None or isinstance(val, (str, int, float, bool)):
            return val
        # Handle Path-like objects (pathlib.Path, os.PathLike, etc.)
        if hasattr(val, "__fspath__"):
            return str(val)
        if isinstance(val, (list, tuple)):
            return [self._sanitize_for_json(item) for item in val]
        if isinstance(val, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in val.items()}
        # Fallback: convert to string to preserve useful information
        return str(val)

    # --- Additional verbs ---
    def _import_library(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = str(payload.get("name_or_path", "")).strip()
        args = payload.get("args") or []
        alias = payload.get("alias")
        if not name:
            return {"success": False, "error": "name_or_path required"}
        ctx = EXECUTION_CONTEXTS.current
        if not ctx or not getattr(ctx, "namespace", None):
            return {"success": False, "error": "no active RF context"}
        try:
            ns = ctx.namespace
            ns.import_library(name, args=tuple(args), alias=alias)
            return {"success": True, "result": {"library": name, "alias": alias}}
        except Exception as e:  # pragma: no cover - bubble details
            return {"success": False, "error": str(e)}

    def _import_resource(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        path = str(payload.get("path", "")).strip()
        if not path:
            return {"success": False, "error": "path required"}
        from pathlib import Path

        p = Path(path)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        ctx = EXECUTION_CONTEXTS.current
        if not ctx or not getattr(ctx, "namespace", None):
            return {"success": False, "error": "no active RF context"}
        try:
            ctx.namespace.import_resource(p.as_posix())
            return {"success": True, "result": {"resource": p.as_posix()}}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _list_keywords(self) -> Dict[str, Any]:
        ctx = EXECUTION_CONTEXTS.current
        if not ctx or not getattr(ctx, "namespace", None):
            return {"success": False, "error": "no active RF context"}
        libs = ctx.namespace.libraries
        items = []
        try:
            if hasattr(libs, "items"):
                iter_libs = list(libs.items())
            elif hasattr(libs, "__iter__"):
                iter_libs = [
                    (
                        getattr(
                            li, "name", getattr(li, "__class__", type(li)).__name__
                        ),
                        li,
                    )
                    for li in libs
                ]
            else:
                iter_libs = []
            for lib_name, lib in iter_libs:
                kws = []
                # best-effort discovery across RF versions
                for attr in ("keywords", "handlers"):
                    try:
                        coll = getattr(lib, attr, None)
                        if not coll:
                            continue
                        for k in coll:
                            name = getattr(k, "name", None)
                            if not name and hasattr(k, "_orig_name"):
                                name = getattr(k, "_orig_name")
                            if name and name not in kws:
                                kws.append(name)
                    except Exception:
                        continue
                items.append({"library": str(lib_name), "keywords": kws})
        except Exception as e:
            return {"success": False, "error": str(e)}
        return {"success": True, "result": items}

    def _get_keyword_doc(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = str(payload.get("name", "")).strip()
        if not name:
            return {"success": False, "error": "name required"}
        # Try library docs via LibDoc for known libs in namespace
        ctx = EXECUTION_CONTEXTS.current
        try:
            from robot.libdoc import LibraryDocumentation
        except Exception as e:  # pragma: no cover
            return {"success": False, "error": f"libdoc unavailable: {e}"}
        try:
            libs = []
            if ctx and getattr(ctx, "namespace", None):
                L = ctx.namespace.libraries
                if hasattr(L, "keys"):
                    libs = list(L.keys())
            for libname in libs:
                try:
                    doc = LibraryDocumentation(libname)
                    for kw in getattr(doc, "keywords", []) or []:
                        if kw.name.lower() == name.lower():
                            return {
                                "success": True,
                                "result": {
                                    "name": kw.name,
                                    "doc": kw.doc,
                                    "args": [str(a) for a in kw.args],
                                    "source": libname,
                                },
                            }
                except Exception:
                    continue
        except Exception as e:  # pragma: no cover
            return {"success": False, "error": str(e)}
        return {"success": False, "error": f"keyword not found: {name}"}

    def _get_variables(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        names = payload.get("names")
        bi = BuiltIn()
        try:
            all_vars = bi.get_variables()
        except Exception as e:  # pragma: no cover
            return {"success": False, "error": str(e)}
        if names:
            out: Dict[str, Any] = {}
            for n in names:
                k = self._norm_var(n)
                if k in all_vars:
                    out[k] = self._sanitize_for_json(all_vars[k])
            return {"success": True, "result": out}
        # Limit size to avoid large payloads
        out = {}
        for i, (k, v) in enumerate(all_vars.items()):
            if i >= 200:
                break
            out[k] = self._sanitize_for_json(v)
        return {"success": True, "result": out, "truncated": True}

    def _set_variable(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = payload.get("name")
        value = payload.get("value")
        scope = str(payload.get("scope", "test")).lower()
        if not name:
            return {"success": False, "error": "name required"}
        bi = BuiltIn()
        norm_name = self._norm_var(str(name))
        if scope == "suite":
            bi.set_suite_variable(norm_name, value)
        elif scope == "global":
            bi.set_global_variable(norm_name, value)
        else:
            bi.set_test_variable(norm_name, value)
        return {"success": True, "scope": scope}

    def _import_variables(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Import variables from a variable file into the live RF context."""
        variable_file_path = str(payload.get("variable_file_path", "")).strip()
        args = payload.get("args") or []
        if not variable_file_path:
            return {"success": False, "error": "variable_file_path required"}
        ctx = EXECUTION_CONTEXTS.current
        if not ctx or not getattr(ctx, "namespace", None):
            return {"success": False, "error": "no active RF context"}
        try:
            ctx.namespace.import_variables(variable_file_path, args=args, overwrite=True)
            return {"success": True, "result": {"variable_file": variable_file_path}}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_page_source(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get page source directly - faster than run_keyword wrapper.

        Automatically detects Browser Library vs SeleniumLibrary and uses
        the appropriate keyword.
        """
        try:
            bi = BuiltIn()

            # Try Browser Library keyword first (Playwright)
            try:
                source = bi.run_keyword("Get Page Source")
                return {
                    "success": True,
                    "result": source,
                    "library": "Browser",
                }
            except Exception:
                pass

            # Fall back to SeleniumLibrary keyword
            try:
                source = bi.run_keyword("Get Source")
                return {
                    "success": True,
                    "result": source,
                    "library": "SeleniumLibrary",
                }
            except Exception:
                pass

            # Try AppiumLibrary (returns XML)
            try:
                source = bi.run_keyword("Get Source")
                return {
                    "success": True,
                    "result": source,
                    "library": "AppiumLibrary",
                    "format": "xml",
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"No browser library loaded or no page open: {e}",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_aria_snapshot(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get ARIA accessibility tree snapshot.

        Only available with Browser Library (Playwright).
        """
        selector = payload.get("selector", "css=html")
        format_type = payload.get("format", "yaml")

        try:
            bi = BuiltIn()
            result = bi.run_keyword(
                "Get Aria Snapshot",
                selector,
                f"return_type={format_type}"
            )
            return {
                "success": True,
                "result": result,
                "format": format_type,
                "selector": selector,
                "library": "Browser",
            }
        except Exception as e:
            error_str = str(e)
            if "No keyword with name" in error_str:
                return {
                    "success": False,
                    "error": "Aria snapshot not available",
                    "note": "Get Aria Snapshot is only available with Browser Library (Playwright)",
                }
            return {"success": False, "error": error_str}

    def _get_session_info(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get RF execution context information."""
        try:
            ctx = EXECUTION_CONTEXTS.current

            if ctx is None:
                return {
                    "success": False,
                    "error": "No active execution context",
                }

            bi = BuiltIn()
            variables = bi.get_variables()

            # Get libraries from namespace
            libs = []
            if getattr(ctx, "namespace", None) and hasattr(ctx.namespace, "libraries"):
                libraries = ctx.namespace.libraries
                if hasattr(libraries, "keys"):
                    libs = list(libraries.keys())
                elif hasattr(libraries, "__iter__"):
                    libs = [
                        getattr(li, "name", getattr(li, "__class__", type(li)).__name__)
                        for li in libraries
                    ]

            return {
                "success": True,
                "result": {
                    "context_active": True,
                    "variable_count": len(variables),
                    "suite_name": getattr(ctx, "suite", None) and getattr(ctx.suite, "name", None),
                    "test_name": getattr(ctx, "test", None) and getattr(ctx.test, "name", None),
                    "libraries": libs,
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
