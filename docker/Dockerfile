FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# rf-mcp runs robotcode with the image's own Python, so the Robot Framework
# libraries a project needs must be installed here. Override at build time:
#   docker build --build-arg RF_LIBRARIES="robotframework-requests" ...
ARG RF_LIBRARIES="robotframework-browser robotframework-seleniumlibrary robotframework-requests"

RUN apt-get update && \
    apt-get install -y build-essential ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk1.0-0 \
    libcairo-gobject2 \
    libcairo2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnss3 \
    libnspr4 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libx11-6 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    wget \
    curl \
    gnupg \
    unzip \
    xdg-utils \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Install Chromium and Firefox (works on both amd64 and arm64)
RUN apt-get update \
    && apt-get install -y chromium firefox-esr \
    && rm -rf /var/lib/apt/lists/*

# Selenium Manager (included in Selenium 4.6+) will automatically download chromedriver/geckodriver

RUN npx --yes playwright install-deps

RUN groupadd -r appuser && useradd -r -g appuser -u 1000 -m -s /bin/bash appuser \
    && mkdir -p /workspace && chown appuser:appuser /workspace

WORKDIR /app

COPY --chown=appuser:appuser --chmod=755 pyproject.toml uv.lock README.md /app/

COPY --chown=appuser:appuser --chmod=755 src /app/src

RUN chown -R appuser:appuser /app

USER appuser

# The venv first on PATH: rf-mcp finds robotcode there, and robot finds the libraries.
ENV PATH="/app/.venv/bin:/home/appuser/.local/bin:$PATH"

RUN uv sync --frozen --no-dev --extra otel && \
    if [ -n "$RF_LIBRARIES" ]; then uv pip install --python /app/.venv/bin/python $RF_LIBRARIES; fi && \
    if command -v rfbrowser >/dev/null; then rfbrowser init; fi && \
    robotmcp --version && robotcode --version

# Mount the Robot Framework project (the folder with robot.toml) here.
VOLUME ["/workspace"]

# MCP over streamable HTTP at http://<host>:8000/mcp
EXPOSE 8000

ENTRYPOINT ["robotmcp", "--transport", "http", "--host", "0.0.0.0", "--port", "8000", "--mode", "server", "--root", "/workspace"]
