# CUDA 12.1 + cuDNN 8 — required for PyTorch 2.4 + flash_attn
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System packages: Python 3.10, ffmpeg (video encoding), libGL (opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git ffmpeg \
    libgl1-mesa-glx libglib2.0-0 \
    wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python/pip
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python -m pip install --upgrade pip

WORKDIR /app

# ── 1. PyTorch 2.4.0 with CUDA 12.1 (must come before flash_attn) ──────────
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ── 2. flash_attn prebuilt wheel (Python 3.10 · CUDA 12 · PyTorch 2.4) ─────
# v2.7.3 uses "cu12" (covers 12.x); v2.6.3 had no cu121 wheel.
RUN pip install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"

# ── 3. Project dependencies (torch/flash_attn already satisfied above) ──────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 4. Source code ───────────────────────────────────────────────────────────
COPY . .

# Install the wan package so `import wan` works from anywhere in the container
RUN pip install --no-cache-dir -e .

# ── 5. Model cache directory ─────────────────────────────────────────────────
# Mount a RunPod Network Volume at /models for persistent weight storage.
# Without a volume, models are downloaded from HuggingFace on every cold start.
RUN mkdir -p /models

# ── 6. Entry point ───────────────────────────────────────────────────────────
CMD ["python", "-u", "handler.py"]
