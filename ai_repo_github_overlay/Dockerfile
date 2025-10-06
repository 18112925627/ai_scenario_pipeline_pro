FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install -r requirements.txt

# 拷贝代码
COPY . /app

# 首次容器构建时可训练一次（如训练耗时长，建议在构建阶段或启动前准备好模型）
# 可切换到启动脚本里按需触发
RUN python -m model.train --train_csv data/seed_dataset.csv --model_dir artifacts || true

EXPOSE 8080
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8080"]
