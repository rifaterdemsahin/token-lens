PORT ?= 8000
.PHONY: setup data dev test docker deploy
setup:
	uv venv --python 3.12 .venv && uv pip install -p .venv -r requirements-dev.txt
data:
	.venv/bin/python scripts/extract_embeddings.py
dev: 
	@test -f data/embeddings.f16.npy || $(MAKE) data
	.venv/bin/uvicorn app.main:app --reload --port $(PORT)
test:
	.venv/bin/python -m pytest -q tests
docker:
	docker build -t token-lens . && docker run --rm -p 8080:8080 token-lens
deploy:
	fly deploy --remote-only
