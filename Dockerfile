FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps untuk OpenCV + ffmpeg (video I/O)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY app ./app
COPY scripts ./scripts

# Model & .env di-mount via volume di docker-compose (jangan bake ke image)
RUN mkdir -p /app/models /app/outputs

EXPOSE 8000 8501

# Default: FastAPI. Override via `command:` di docker-compose untuk service UI.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
