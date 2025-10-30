FROM nvidia/cuda:11.7.1-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Australia/Sydney

# Set timezone
RUN apt-get update && apt-get install -y --no-install-recommends tzdata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Basic deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates curl git wget bzip2 libgl1 libglib2.0-0

# Install Python3.9
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.9 python3.9-venv python3.9-distutils python3.9-dev

# Set default python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    ln -sf /usr/local/bin/pip3.9 /usr/local/bin/pip

# Copy project into image 
RUN git clone https://github.com/zhipeixu/FakeShield.git /workspace/FakeShield
WORKDIR /workspace/FakeShield
RUN mkdir data

# Install dependencies
RUN apt-get update && apt-get install -y git && \
    pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --index-url https://download.pytorch.org/whl/cu117 && \
    pip install flash-attn==2.3.6 --no-build-isolation && \
    pip install -r requirements.txt && \
    pip install mmcv-full==1.4.7

# Install DTE-FDM and FlashAttention
RUN cd /workspace/FakeShield/DTE-FDM && \
    pip install -e . && \
    pip install -e ".[train]"

# Default command: show versions at container start
CMD ["bash", "-lc", "python - <<'PY'\nimport torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())\nPY"]

