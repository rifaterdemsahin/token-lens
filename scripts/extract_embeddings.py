"""Download tokenizer + embedding matrix from Hugging Face and write offline artifacts to data/."""
import json
import os
import shutil
import sys

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

MODEL_ID = os.environ.get("MODEL_ID", "openai-community/gpt2")
EMBED_TENSOR = os.environ.get("EMBED_TENSOR", "wte.weight")
OUT = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    tok = hf_hub_download(MODEL_ID, "tokenizer.json")
    shutil.copy(tok, os.path.join(OUT, "tokenizer.json"))
    st = hf_hub_download(MODEL_ID, "model.safetensors")
    with safe_open(st, framework="np") as f:
        names = list(f.keys())
        name = EMBED_TENSOR if EMBED_TENSOR in names else next(
            (n for n in names if n.endswith(EMBED_TENSOR)), None)
        if name is None:
            sys.exit(f"tensor {EMBED_TENSOR!r} not found; available: {names[:10]}")
        w = f.get_tensor(name).astype(np.float32)
    print("embedding matrix", w.shape)
    np.save(os.path.join(OUT, "embeddings.f16.npy"), w.astype(np.float16))

    # PCA via SVD on the covariance matrix (dims x dims) - cheap, no sklearn.
    mean = w.mean(axis=0)
    xc = w - mean
    cov = (xc.T @ xc) / len(xc)
    vals, vecs = np.linalg.eigh(cov)
    comps = vecs[:, ::-1][:, :3]
    pca = (xc @ comps).astype(np.float32)
    np.save(os.path.join(OUT, "pca3.npy"), pca)
    np.save(os.path.join(OUT, "pca_mean.npy"), mean.astype(np.float32))
    with open(os.path.join(OUT, "meta.json"), "w") as fh:
        json.dump({"model_id": MODEL_ID, "tensor": name, "vocab": w.shape[0], "dims": w.shape[1]}, fh)
    print("done ->", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
