# Mask2Former Hospital Segmentation — DGX Spark Container
# ARM64 (aarch64) — uses nvcr.io PyTorch base image for NVIDIA DGX Spark
FROM nvcr.io/nvidia/pytorch:25.11-py3

LABEL maintainer="mask2former-seg"
LABEL description="Mask2Former hospital segmentation pipeline (DGX Spark)"

# System dependencies (Ubuntu 24.04 inside nvcr pytorch image)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies
# Note: PyTorch and torchvision are already provided by the base image
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        transformers>=4.40.0 \
        peft>=0.9.0 \
        accelerate \
        albumentations \
        torchmetrics \
        scipy \
        opencv-python-headless \
        tqdm \
        numpy \
        matplotlib \
        pillow \
        pycocotools \
        sentencepiece \
        protobuf \
        roboflow

# Copy project scripts only (data/output are mounted as volumes)
COPY scripts/ /workspace/scripts/

# Make scripts importable as a package
RUN touch /workspace/scripts/__init__.py 2>/dev/null || true

# Environment: persist HuggingFace model cache on the host
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

# Default: show help
CMD ["python", "-m", "scripts.mask2former_seg.train", "--help"]
