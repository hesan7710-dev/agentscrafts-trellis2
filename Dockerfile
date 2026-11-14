FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda-12.4 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget curl build-essential cmake ninja-build sudo \
    libgl1 libglib2.0-0 libxrender1 libxext6 libsm6 libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

WORKDIR /opt/TRELLIS.2
RUN git clone --depth 1 --recursive https://github.com/microsoft/TRELLIS.2.git /opt/TRELLIS.2

# Upstream setup.sh checks for nvidia-smi while building. Docker build machines
# usually have CUDA toolkit but no attached GPU/driver, so provide a build-time
# detection shim. This selects the CUDA install path; it does not run inference.
RUN printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/nvidia-smi \
    && chmod +x /usr/local/bin/nvidia-smi \
    && bash -lc 'source /opt/conda/etc/profile.d/conda.sh && cd /opt/TRELLIS.2 && source ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm && conda run -n trellis2 pip install runpod boto3 pillow opencv-python-headless imageio' \
    && rm -f /usr/local/bin/nvidia-smi

WORKDIR /app
COPY worker.py ./worker.py
CMD ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "trellis2", "python", "-u", "/app/worker.py"]
