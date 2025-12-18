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
    curl \
    gnupg \
    unzip \
    xdg-utils \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Install Chrome
RUN curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install Firefox
RUN apt-get update \
    && apt-get install -y firefox-esr \
    && rm -rf /var/lib/apt/lists/*

# Selenium Manager (included in Selenium 4.6+) will automatically download chromedriver/geckodriver

RUN npx --yes playwright install-deps

RUN groupadd -r appuser && useradd -r -g appuser -u 1000 -m -s /bin/bash appuser

WORKDIR /app

COPY --chown=appuser:appuser --chmod=755 pyproject.toml uv.lock README.md /app/

COPY --chown=appuser:appuser --chmod=755 src /app/src

RUN chown -R appuser:appuser /app

USER appuser

ENV PATH="/home/appuser/.local/bin:$PATH"

RUN uv lock && \
    uv sync --all-extras --no-dev && \
    uv run -- rfbrowser init

# Expose ports for HTTP transport (8000) and Frontend dashboard (8001)
EXPOSE 8000 8001

ENTRYPOINT ["uv", "run", "--no-sync", "robotmcp", "--transport", "http", "--host", "0.0.0.0", "--port", "8000", "--with-frontend", "--frontend-host", "0.0.0.0", "--frontend-port", "8001"]
