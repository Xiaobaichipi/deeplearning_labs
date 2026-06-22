# DeepLearning Labs — Docker 镜像
# 构建时可指定 pip 镜像源加速中国用户下载:
#   docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ -t deeplearning-labs .

FROM python:3.12-slim

ARG PIP_INDEX_URL=https://pypi.org/simple/
ARG PIP_TRUSTED_HOST=pypi.org

WORKDIR /app

# install git so users can pull updates inside the container
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# all Python dependencies ship pre-built wheels (torch, numpy, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary :all: \
    --index-url $PIP_INDEX_URL \
    --trusted-host $PIP_TRUSTED_HOST \
    -r requirements.txt

COPY . .

EXPOSE 5000

ENV HOST=127.0.0.1

CMD ["python", "main.py"]
