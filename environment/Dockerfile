# Base image
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /workspace

# 日本語設定
RUN apt-get update && apt-get install -y locales \
    && localedef -i ja_JP -f UTF-8 ja_JP.UTF-8

# Install dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y git wget libgl1-mesa-glx libglib2.0-0

RUN apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    libsm6 \
    libxext6 \
    libxrender-dev

# Install Python packages
RUN pip install --no-cache-dir \
    transformers==4.41.2 \
    accelerate==0.30.1 \
    sentencepiece==0.2.0 \
    einops==0.8.0 \
    timm==0.9.16 \
    peft==0.11.1 \
    datasets==2.19.1 \
    gradio==4.31.5 \
    matplotlib==3.9.0 \
    scipy==1.13.1 \
    qwen_vl_utils==0.0.11

# Set environment variables
ENV PYTHONPATH=/workspace

CMD ["/bin/bash"]