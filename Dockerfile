# ── ChessLens Dockerfile ──────────────────────────────────────────────────────
# Build:  docker build -t chesslens .
# Run:    docker run -p 8501:8501 -v chesslens_data:/app/data chesslens
# Open:   http://localhost:8501
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Install Stockfish (the actual chess engine binary)
RUN apt-get update \
    && apt-get install -y --no-install-recommends stockfish \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/games:${PATH}"

# Verify stockfish is on PATH (fail fast if install broke)
RUN which stockfish && echo "quit" | stockfish

WORKDIR /app

# Install Python dependencies first (cached layer unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY *.py ./
COPY chesslens_config.json ./

# Create data directories that the app expects
RUN mkdir -p data/raw_pgn data/processed data/analytics models

# Non-root user for security (Streamlit doesn't need root)
RUN useradd -m chesslens && chown -R chesslens:chesslens /app
USER chesslens

# Streamlit config — disable telemetry, set headless mode
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

# Healthcheck — Streamlit responds on /_stcore/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py"]
