# API image: retrieval runs on CPU, generation goes to an OpenAI-compatible
# backend (vLLM, Ollama, hosted API) configured with QMR_LLM.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only torch keeps the image small (sentence-transformers depends on it).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY configs/ configs/
COPY src/ src/

# Bake the embedding model into the image so startup needs no network.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

ENV QMR_INDEX_DIR=/app/indexes/hotpotqa
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/readyz')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
