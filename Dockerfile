FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility

RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm \
    cmake unzip git wget tmux nano curl \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool \
    python3 python3-pip python3-dev python3-setuptools \
    libsm6 libxext6 libxrender-dev &&\
    ln -sf /usr/bin/python3 /usr/bin/python &&\
    ln -sf /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir numpy==1.21.3

# Install PyTorch
RUN pip3 install --no-cache-dir \
    torch==1.10.0+cu113 \
    torchvision==0.11.1+cu113 \
    torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

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
