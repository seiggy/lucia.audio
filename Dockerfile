FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf-cache \
    TORCH_HOME=/data/torch-cache

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Install PyTorch with CUDA 12.4 support first (layer cached)
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies (layer caching for faster rebuilds)
COPY pyproject.toml README.md ./
COPY wyoming_chatterbox/ ./wyoming_chatterbox/
COPY dashboard/ ./dashboard/
RUN pip install .

# Create data directories
RUN mkdir -p /data/voices /data/hf-cache /data/torch-cache

# Pre-download model weights during build (optional, enables offline startup)
# Uncomment the next line to bake model weights into the image (~1.5GB larger)
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('ResembleAI/chatterbox-turbo', allow_patterns=['*.safetensors','*.json','*.txt','*.pt','*.model'])"

EXPOSE 10200 8095

VOLUME /data

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8095/api/health')" || exit 1

ENTRYPOINT ["python", "-m", "wyoming_chatterbox"]
CMD ["--uri", "tcp://0.0.0.0:10200", "--voices-dir", "/data/voices", "--dashboard-port", "8095"]
