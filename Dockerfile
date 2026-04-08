# ─────────────────────────────────────────────────────────────
# Dockerfile — CRM Lead Scoring OpenEnv
# Build:  docker build -t crm-lead-scoring-openenv .
# Run:    docker run -p 8000:8000 crm-lead-scoring-openenv
# Docs:   http://localhost:8000/docs
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL name="crm-lead-scoring-openenv"
LABEL version="1.0.0"
LABEL description="AI-Powered CRM Lead Scoring RL Environment (OpenEnv)"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create non-root user (security best practice)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]