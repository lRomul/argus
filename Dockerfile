FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm \
    cmake unzip git wget tmux nano curl \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool \
    python3 python3-pip python3-dev python3-setuptools \
    libsm6 libxext6 libxrender-dev &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir numpy==1.19.4

# Install PyTorch
RUN pip3 install --no-cache-dir \
    torch==1.7.1+cu110 \
    torchvision==0.8.2+cu110 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install Apex
RUN git clone https://github.com/NVIDIA/apex && cd apex &&\
    pip install -v --no-cache-dir \
    --global-option="--cpp_ext" \
    --global-option="--cuda_ext" ./

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
