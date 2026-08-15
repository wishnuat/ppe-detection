FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Matplotlib & Ultralytics menulis config ke $HOME saat import. Di HuggingFace
    # Spaces container jalan sebagai UID 1000 tanpa home yang writable, dan tanpa
    # ini import ultralytics gagal dengan PermissionError.
    HOME=/app \
    MPLCONFIGDIR=/tmp/mpl \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# System deps untuk OpenCV + ffmpeg (video I/O)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch CPU dipasang lebih dulu dari index khusus PyTorch. Kalau langkah ini
# dilewat, `pip install -r requirements.txt` akan menarik wheel CUDA dari PyPI
# (~2.5 GB) padahal proyek ini murni CPU — akselerasi ditangani OpenVINO.
# Versinya sama persis dengan pin di requirements.txt, jadi langkah berikutnya
# melihatnya sebagai "already satisfied" dan tidak mengunduh ulang.
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.13.0 torchvision==0.28.0

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY app ./app
COPY web ./web
COPY scripts ./scripts

# Weights ikut di-bake supaya image bisa jalan sendiri di HuggingFace Spaces /
# Railway, yang tidak punya volume mount. Di docker-compose, ./models di-mount
# menimpa folder ini — jadi pengembangan lokal tetap memakai file di host.
# Kalau models/ kosong saat build, image tetap jadi: GET /health akan
# mengembalikan 503 dengan pesan bahwa weights belum ada.
COPY models ./models

RUN mkdir -p /app/outputs /tmp/mpl /tmp/ultralytics && \
    chmod -R 777 /app/outputs

# 8000 = default lokal & docker-compose. 7860 = port yang diharapkan
# HuggingFace Spaces (di-set lewat env PORT, lihat README).
EXPOSE 8000 7860

# Shell form supaya ${PORT} diekspansi. Satu container melayani API *dan*
# frontend statis — app/api.py me-mount web/ di "/".
CMD uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}
