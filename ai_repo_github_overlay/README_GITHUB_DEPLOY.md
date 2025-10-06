# GitHub 仓库部署指引（适用于 `ai_scenario_pipeline_pro`）

把本压缩包解压到你的项目根目录（与 `service/`、`model/`、`requirements.txt` 同级），提交并推送到 GitHub。
随后可选择下述任一部署方案。

> 后端需暴露 FastAPI 应用 `service/app.py` 中的 `app` 对象；`/labels` 路径用于健康检查。

---

## 方案 A：Render（推荐新手）
1. GitHub 连接到 Render：**New → Web Service**，选择你的仓库。
2. 设置：
   - Build Command
     ```bash
     pip install -r requirements.txt && python -m model.train --train_csv data/seed_dataset.csv --model_dir artifacts
     ```
   - Start Command
     ```bash
     uvicorn service.app:app --host 0.0.0.0 --port $PORT
     ```
   - Health Check Path: `/labels`
   - 环境变量：`MODEL_DIR=artifacts`
3. 部署完成后得到 `https://xxx.onrender.com`，把小程序云函数的 `AI_BASE_URL` 指向该域名。

> 也可直接在 Render 选择 **Blueprints**，导入仓库中的 `render.yaml` 一键创建。

---

## 方案 B：Google Cloud Run（GitHub Actions 自动部署）
前置：一个 GCP 项目、启用 Cloud Run 与 Artifact Registry；创建服务账号并下载 JSON 密钥。

1. 在 GitHub 项目的 **Settings → Secrets and variables → Actions → New repository secret** 添加：
   - `GCP_PROJECT_ID`：你的项目 ID
   - `GCP_REGION`：如 `asia-east1` / `us-central1`
   - `GCP_SA_KEY`：服务账号 JSON（内容整段复制粘贴）
   - `SERVICE_NAME`：如 `foodpack-ai`
2. 合并到 `main` 分支，Actions 会触发 `.github/workflows/cloud-run.yml`，自动：
   - 使用 Cloud Build 构建镜像
   - 部署到 Cloud Run（全网可访问）
3. 部署成功后在 Cloud Run 控制台能看到 HTTPS URL，填入 `AI_BASE_URL`。

---

## 方案 C：Fly.io（GitHub Actions 自动部署）
1. 安装并配置 `flyctl`（本地创建应用）：
   ```bash
   fly launch --no-deploy
   # 按提示生成 fly.toml（确保内部端口为 8080）
   ```
2. 在 GitHub 的 Secrets 添加：
   - `FLY_API_TOKEN`：`flyctl auth token` 获取的 token
3. 推送后 Actions 会使用 `.github/workflows/flyio.yml` 自动 `fly deploy`。

---

## Docker 本地测试
```bash
docker build -t foodpack-ai:latest .
docker run -p 8080:8080 foodpack-ai:latest
# http://127.0.0.1:8080/labels
```

---

## 目录要求（关键文件）
```
.
├─ service/app.py        # FastAPI app，暴露变量 app
├─ model/                # 训练脚本与代码
├─ data/seed_dataset.csv # 初始数据
├─ requirements.txt
├─ Dockerfile            # 本包提供
├─ render.yaml           # 本包提供
└─ .github/workflows/    # 本包提供（Cloud Run / Fly.io）
```

**提示**：若你的入口模块不叫 `service.app:app`，记得在 `render.yaml`、`Dockerfile`、工作流里把启动命令改为你的模块路径。
