# ---- stage 1: download model assets and extract embeddings for every model in app/models.py (needs network)
FROM python:3.12-slim AS builder
ENV DATA_DIR=/data HF_HOME=/tmp/hf
RUN pip install --no-cache-dir huggingface_hub==0.27.0 numpy==2.2.1
WORKDIR /work
COPY app/__init__.py app/models.py app/
COPY scripts scripts
RUN python scripts/extract_embeddings.py

# ---- stage 2: slim, fully offline runtime
FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY --from=builder /data ./data
ENV DATA_DIR=/srv/data HF_HUB_OFFLINE=1
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
