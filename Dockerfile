FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm \
    cmake unzip git wget tmux nano \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool \
    python3 python3-pip python3-dev python3-setuptools \
    libsm6 libxext6 libxrender-dev &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir numpy==1.19.1

# Install PyTorch
RUN pip3 install --no-cache-dir \
    torch==1.5.1 \
    torchvision==0.6.1

# Install Apex
RUN git clone https://github.com/NVIDIA/apex && cd apex &&\
    pip install -v --no-cache-dir \
    --global-option="--cpp_ext" \
    --global-option="--cuda_ext" ./

# Install python ML packages
RUN pip3 install --no-cache-dir \
    notebook==6.0.3 \
    cnn-finetune==0.6.0 \
    timm==0.1.30 \
    flake8==3.8.3 \
    pytest==6.0.1 \
    pytest-cov==2.10.1

# Docs requirements
COPY ./docs/requirements.txt /docs_requirements.txt
RUN pip3 install --no-cache-dir -r /docs_requirements.txt

ENV PYTHONPATH $PYTHONPATH:/workdir

WORKDIR /workdir
