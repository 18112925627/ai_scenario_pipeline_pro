# FoodPack AI Scenario Pipeline · PRO
- FastAPI 推理服务 + 轻量标注台（/label）+ 主动学习 + 经典模型/可选 BERT
- 端口：8080，环境变量：
  - MODEL_DIR: 经典模型目录（默认 artifacts）
  - USE_BERT: '1' 则使用 BERT 模型（目录 artifacts_bert）
  - BERT_MODEL_NAME: 训练脚本使用的 HuggingFace 模型名（默认 hfl/chinese-macbert-base）

## 本地启动（经典模型）
```
pip install -r requirements.txt
python -m model.train --train_csv data/seed_dataset.csv --model_dir artifacts
uvicorn service.app:app --host 0.0.0.0 --port 8080
```

## BERT 训练/部署（可选，需 GPU 或较强 CPU）
```
pip install -r requirements-bert.txt
python -m model.train_bert --train_csv data/seed_dataset.csv --model_name hfl/chinese-macbert-base --model_dir artifacts_bert
USE_BERT=1 uvicorn service.app:app --host 0.0.0.0 --port 8080
```

## 标注台
打开 `http://localhost:8080/label`：拉取未标注 → 勾选标签 → 提交；
导出标注集：`/export.csv`。将其并入训练集继续训练。

## Docker
```
# 经典
docker build -t scenario-ai:classic -f Dockerfile .
docker run -p 8080:8080 scenario-ai:classic
# BERT
docker build -t scenario-ai:bert -f Dockerfile.bert .
docker run -p 8080:8080 -e USE_BERT=1 scenario-ai:bert
```
