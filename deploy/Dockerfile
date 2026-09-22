# Multi-stage production Dockerfile for CountLoop
# Based on Ubuntu 22.04, Python 3.10, CUDA 12.2 (matching TMLR paper spec)

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1/12.2 support
RUN python -m pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install core and web service dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Install diffusion and vision libraries
RUN python -m pip install --no-cache-dir \
    diffusers>=0.28.0 \
    transformers>=4.40.0 \
    accelerate>=0.28.0 \
    safetensors>=0.4.0

# Copy codebase
COPY . /app

# Install package in editable mode
RUN python -m pip install -e .

EXPOSE 8000

# Default command: launch production API server
CMD ["python", "-m", "countloop.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
