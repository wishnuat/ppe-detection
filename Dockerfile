FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

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

# HuggingFace Spaces menjalankan container sebagai UID 1000, bukan root.
# Matplotlib, Ultralytics, dan fontconfig semuanya menulis cache ke $HOME saat
# import — kalau tidak writable, ketiganya jatuh ke fallback dan membanjiri log
# dengan warning, plus font cache matplotlib dibangun ulang tiap start.
#
# Membuat user 1000 dengan home sungguhan menyelesaikan ketiganya sekaligus,
# dan lebih rapi daripada menyebar chmod 777 ke /tmp.
RUN useradd --create-home --uid 1000 appuser

ENV HOME=/home/appuser \
    XDG_CACHE_HOME=/home/appuser/.cache \
    MPLCONFIGDIR=/home/appuser/.cache/matplotlib \
    YOLO_CONFIG_DIR=/home/appuser/.config/Ultralytics

RUN mkdir -p /app/outputs "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR" && \
    chown -R appuser:appuser /app /home/appuser

USER appuser

# 8000 = default lokal & docker-compose. 7860 = port yang diharapkan
# HuggingFace Spaces (di-set lewat env PORT, lihat README).
EXPOSE 8000 7860

# Satu container melayani API *dan* frontend statis — app/api.py me-mount
# web/ di "/".
#
# Dibungkus `sh -c` (bukan shell form telanjang) supaya ${PORT} tetap
# diekspansi tanpa memicu peringatan JSONArgsRecommended dari buildkit.
# `exec` membuat uvicorn menggantikan sh sebagai PID 1, jadi SIGTERM dari
# `docker stop` sampai ke uvicorn dan shutdown-nya bersih — tanpa itu sh
# menelan sinyalnya dan container baru mati setelah timeout 10 detik.
CMD ["sh", "-c", "exec uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
