FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility
ENV PYTHONPATH $PYTHONPATH:/workdir

WORKDIR /workdir

RUN apt-get update && \
    apt-get -y install \
    build-essential cmake unzip git wget tmux nano curl \
    python3 python3-pip python3-dev python3-setuptools && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/archives/*

# Install PyTorch
RUN pip3 install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    torchaudio==2.3.0

# Install requirements
COPY ./ ./
RUN pip3 install --no-cache-dir -e .[tests,docs,examples]
