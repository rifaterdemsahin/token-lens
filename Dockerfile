# ---- stage 1: download model assets and extract embeddings (needs network)
FROM python:3.12-slim AS builder
ARG MODEL_ID=openai-community/gpt2
ARG EMBED_TENSOR=wte.weight
ENV MODEL_ID=${MODEL_ID} EMBED_TENSOR=${EMBED_TENSOR} DATA_DIR=/data HF_HOME=/tmp/hf
RUN pip install --no-cache-dir huggingface_hub==0.27.0 safetensors==0.5.0 numpy==2.2.1
COPY scripts/extract_embeddings.py /scripts/extract_embeddings.py
RUN python /scripts/extract_embeddings.py

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
