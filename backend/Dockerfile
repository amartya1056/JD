# Backend image: FastAPI + the jobguard core library + trained model bundle.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for lxml/xgboost wheels are already covered by manylinux wheels;
# keep the image slim. Add build-essential only if a source build is needed.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the shared library, backend app, and model artifacts.
COPY jobguard ./jobguard
COPY backend ./backend
COPY models ./models

# Non-root user (least privilege).
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# The model is loaded once at startup (see backend/app/main.py lifespan).
# Shell form so hosts that inject $PORT (Render, Railway, Fly, Cloud Run) are
# respected; falls back to 8000 for local docker-compose.
CMD uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}
