# Docker images

Two images run rf-mcp as a remote test runner in server mode over streamable HTTP
(see [Execution modes](CONFIGURATION.md#execution-modes)):

| image | Dockerfile | contents | ports |
| --- | --- | --- | --- |
| `ghcr.io/manykarim/rf-mcp` | `docker/Dockerfile` | rf-mcp, robotcode, Browser (Playwright browsers), SeleniumLibrary with Chromium and Firefox, RequestsLibrary; headless | 8000 |
| `ghcr.io/manykarim/rf-mcp-vnc` | `docker/Dockerfile.vnc` | the same, plus an X display (Xvfb, fluxbox) you can watch over VNC or in the browser (noVNC) | 8000, 5900, 6080 |

Tags: `main` (latest commit on main), `<version>`, `<major>.<minor>`, `<major>` and `latest` (latest release).
Both images are built for `linux/amd64` and `linux/arm64`.

## Run

Mount the Robot Framework project (the folder with `robot.toml`) at `/workspace`:

```bash
docker run --rm --shm-size=1g -p 8000:8000 -v "$PWD:/workspace" ghcr.io/manykarim/rf-mcp
```

The server listens at `http://localhost:8000/mcp`. It runs as user `appuser` (uid 1000), which needs write
access to the mounted folder for results (`results/`, `output.xml`). On Linux, either own the folder by uid 1000
or start the container with `--user "$(id -u):$(id -g)"`.

Connect an agent:

```bash
rf-mcp init --agent claude --url http://localhost:8000/mcp --write
```

`init` writes the HTTP entry into the agent's configuration (here `.mcp.json`). Any MCP client that speaks
streamable HTTP works the same way.

Without a mounted folder the agent can still work: it sends the files it changed with the `files` argument of
`rf_analyze`, `rf_run`, `rf_repl` and `rf_debug`, and the server writes them below `/workspace` before the call.

The HTTP transport has no authentication. Publish the port only on a trusted network, or put an authenticating
proxy in front of it.

## Watch the browser (VNC image)

```bash
docker run --rm --shm-size=1g -p 8000:8000 -p 6080:6080 -p 5900:5900 -v "$PWD:/workspace" ghcr.io/manykarim/rf-mcp-vnc
```

`--shm-size=1g` matters here: with Docker's default 64 MB of shared memory, Chromium opens headful pages but
fails to take screenshots (`Page.captureScreenshot: Unable to capture screenshot`).

Open `http://localhost:6080/vnc.html` (noVNC) or connect a VNC viewer to `localhost:5900` (no password). Tests that
open a browser with `headless=False` show up on display `:99`. Supervisor runs Xvfb, fluxbox, x11vnc, noVNC and
rf-mcp; their logs are in `/var/log/supervisor/`.

## Build

```bash
docker build -f docker/Dockerfile -t rf-mcp .
docker build -f docker/Dockerfile.vnc -t rf-mcp-vnc .
```

rf-mcp runs robotcode with the image's Python, so the libraries your tests import must be installed in the image.
`RF_LIBRARIES` sets them (pip requirement strings, separated by spaces):

```bash
docker build -f docker/Dockerfile -t rf-mcp-api \
  --build-arg RF_LIBRARIES="robotframework-requests robotframework-jsonlibrary" .
```

`rfbrowser init` runs only when Browser is in the list. To add libraries to a published image instead, extend it:

```dockerfile
FROM ghcr.io/manykarim/rf-mcp
RUN uv pip install --python /app/.venv/bin/python robotframework-databaselibrary
```

The build fails unless `robotmcp --version` and `robotcode --version` succeed in the finished image.

## Server options

The entrypoint is
`robotmcp --transport http --host 0.0.0.0 --port 8000 --mode server --root /workspace`. Further options can be
appended, or set as environment variables (see [CONFIGURATION.md](CONFIGURATION.md)):

```bash
docker run --rm -p 8000:8000 -v "$PWD:/workspace" -e RF_MCP_PROFILE=ci \
  -v /opt/team-skills:/skills -e RF_MCP_SKILLS_DIR=/skills ghcr.io/manykarim/rf-mcp
```
