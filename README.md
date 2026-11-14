# Token Lens

Inspect **real GPT-2 token IDs and embedding vectors**: type text, see each token's integer ID, its 768-number
vector as a heatmap, a 3D PCA map, nearest neighbours, analogies, and a named-axes view
(Royalty / Femininity / Fruit / Power) computed from the real vectors.

![screenshot placeholder](docs/screenshot.png)

## Run locally
```
make setup && make dev        # http://localhost:8000 (extracts embeddings on first run)
make test
# or
docker build -t token-lens . && docker run --rm -p 8080:8080 token-lens
```

## How the numbers are obtained
Text → GPT-2 BPE tokenizer (`tokenizer.json`) → integer **token ID** → that ID is a row index into the embedding
matrix `wte.weight` (50,257 × 768, from `model.safetensors`) → the row is the vector.
`scripts/extract_embeddings.py` does this once at Docker build time (no PyTorch) and writes
`data/embeddings.f16.npy`, `pca3.npy`, `tokenizer.json`; the runtime image is offline and memory-maps the matrix.
Models are registered in `app/models.py` (slug → HF repo + embedding tensor name): GPT-2, Qwen2.5-0.5B, Gemma-3-270m. Pick one in the UI or pass `?model=slug` to any API endpoint. Add a model by adding a registry entry and re-running extraction.

- **Token IDs are specific to each tokenizer**: GPT-2 `king`=3364, Qwen `king`=10566, Gemma `king`=8971; ID 9246 is `cat` in GPT-2 (not `king`). See the in-app *Models* page.
- UI is available in English and Turkish (toggle top right). Pages: `/`, `/models`, `/architecture`.
- Real token IDs are **arbitrary**: `king` (3364) / ` king` (5822) / ` queen` (16599) are not adjacent.
- A leading space is part of the token: `king` ≠ ` king`.
- Learned weights, not database rows. Individual dimensions are unlabeled.
- **Named axes are derived**: each is a normalised direction built from word pairs (shown in the UI);
  score = cosine(token vector, axis). Custom axes can be added in the UI.

## Deploy to Fly.io
```
fly auth login
fly launch --no-deploy --copy-config
fly tokens create deploy      # add as FLY_API_TOKEN GitHub secret, then push to main
```
CI (`.github/workflows/deploy.yml`) runs tests then `flyctl deploy --remote-only`.

> Image size: all three models are baked in (~1 GB of embeddings). Matrices are memory-mapped, so RAM stays within 1 GB. For a smaller image, remove entries from `app/models.py`.
