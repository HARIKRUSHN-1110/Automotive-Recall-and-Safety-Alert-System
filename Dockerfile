# Dockerfile
# Base: python:3.11-slim

FROM python:3.11-slim

LABEL description="AI-powered automotive recall risk checker"

# Environment
# PYTHONDONTWRITEBYTECODE  — no .pyc files (keeps image clean)
# PYTHONUNBUFFERED         — print() goes straight to terminal
# PYTHONPATH               — lets `from src.x import y` work from any working directory
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# System dependencies
# build-essential  — needed to compile some Python C extensions
# curl             — used by the Docker healthcheck
# libgomp1         — OpenMP runtime required by LightGBM
# cleaning apt cache immediately to keep the layer small.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
# Copy requirements BEFORE the source code.
# Why? Docker caches each layer. If requirements.txt hasn't changed, Docker reuses the cached pip install layer and only
# rebuilds the layers after it. Putting COPY . . first would bust the cache on every single code change.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Download NLTK data required by text_processing.py
# Doing this at build time means the container never needs internet access at runtime.
RUN python -c "import nltk; nltk.download('punkt', quiet=True); \
    nltk.download('punkt_tab', quiet=True); \
    nltk.download('stopwords', quiet=True)"

# Application code
# Copy only what the app needs to run.
# data/ is intentionally not added — it is mounted as a volume at runtime so the container always uses the latest DB and model without needing a rebuild.
COPY src/       ./src/
COPY notebooks/ ./notebooks/
COPY scripts/   ./scripts/

# Runtime directories
# Create empty data directories so volume mounts have targets.
# If the host volume isn't mounted, the app fails gracefully with a clear "database not found" error rather than a cryptic OS error.
RUN mkdir -p data/raw \
             data/processed \
             data/models \
             logs

# Streamlit config
# headless     — no browser auto-open (useless in a container)
# enableCORS   — allow requests from outside the container
# enableXsrfProtection — off for Docker simplicity
RUN mkdir -p /root/.streamlit
RUN echo '\
[server]\n\
headless = true\n\
port = 8501\n\
address = "0.0.0.0"\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /root/.streamlit/config.toml

# Port
EXPOSE 8501

# Health check
# Docker will mark the container unhealthy if the app stops responding. start-period gives Streamlit time to boot before 
# health checks start (it can be slow on first load).
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=40s \
    --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start command─
CMD ["streamlit", "run", "src/app/streamlit_app.py"]