FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility

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
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Docs requirements
COPY ./docs/requirements.txt /docs_requirements.txt
RUN pip3 install --no-cache-dir -r /docs_requirements.txt

# Tests requirements
COPY ./tests/requirements.txt /tests_requirements.txt
RUN pip3 install --no-cache-dir -r /tests_requirements.txt

# Examples requirements
COPY ./examples/requirements.txt /examples_requirements.txt
RUN pip3 install --no-cache-dir -r /examples_requirements.txt

ENV PYTHONPATH $PYTHONPATH:/workdir
WORKDIR /workdir
