FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app
RUN python -m model.train --train_csv data/seed_dataset.csv --model_dir artifacts || true

EXPOSE 8080
CMD ["uvicorn","service.app:app","--host","0.0.0.0","--port","8080"]
