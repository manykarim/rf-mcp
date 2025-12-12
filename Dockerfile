FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

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
        xdg-utils \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

RUN npx --yes playwright install-deps

RUN groupadd -r appuser && useradd -r -g appuser -u 1000 -m -s /bin/bash appuser

WORKDIR /app

COPY --chown=appuser:appuser pyproject.toml uv.lock README.md /app/

COPY --chown=appuser:appuser src /app/src

USER appuser

ENV PATH="/home/appuser/.local/bin:$PATH"

RUN uv lock && \
    uv sync --all-extras --no-dev && \
    uv run -- rfbrowser init

ENTRYPOINT ["uv", "run", "--no-sync", "robotmcp"]
