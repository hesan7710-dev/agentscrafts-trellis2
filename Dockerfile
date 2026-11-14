FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda-12.4 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget curl build-essential cmake ninja-build \
    libgl1 libglib2.0-0 libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh
WORKDIR /opt/TRELLIS.2
RUN git clone --depth 1 --recursive https://github.com/microsoft/TRELLIS.2.git /opt/TRELLIS.2
# Upstream setup script installs the project's CUDA 12.4/PyTorch 2.6 stack and compiled extensions.
RUN bash -lc 'source /opt/conda/etc/profile.d/conda.sh && cd /opt/TRELLIS.2 && source ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm && conda run -n trellis2 pip install runpod boto3 pillow opencv-python-headless imageio'
WORKDIR /app
COPY worker.py ./worker.py
CMD ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "trellis2", "python", "-u", "/app/worker.py"]
