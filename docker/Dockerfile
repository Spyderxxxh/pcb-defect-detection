# DefectGuard MVP Pipeline Dockerfile
# 支持 CUDA 11.8 + PyTorch 2.0

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 设置Python3为默认
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 升级pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 安装PyTorch (CUDA 11.8版本)
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 设置工作目录
WORKDIR /workspace

# 复制项目文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY . .

# 创建必要的目录
RUN mkdir -p data models outputs logs

# 设置环境变量
ENV PYTHONPATH=/workspace:$PYTHONPATH

# 暴露端口
EXPOSE 8000

# 默认命令
CMD ["python", "inference.py", "--help"]
