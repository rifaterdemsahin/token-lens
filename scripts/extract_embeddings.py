"""Download tokenizer + embedding matrix for each registered model and write offline artifacts to
data/<slug>/. No PyTorch: the safetensors container is parsed directly so bf16 weights work with numpy.

usage: extract_embeddings.py [slug ...]   (default: all models in app/models.py)
"""
import json
import os
import shutil
import sys

import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from app.models import MODELS  # noqa: E402

OUT = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))


def read_tensor(path: str, name: str) -> np.ndarray:
    """Read one tensor from a .safetensors file as float32 (supports F32/F16/BF16)."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    if name not in header:
        cands = [k for k in header if k.endswith(name)]
        if not cands:
            sys.exit(f"tensor {name!r} not found in {path}; e.g. {list(header)[:8]}")
        name = cands[0]
    info = header[name]
    a, b = info["data_offsets"]
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n + a, shape=(b - a,))
    shape, dt = info["shape"], info["dtype"]
    if dt == "BF16":
        u16 = raw.view(np.uint16)
        out = (u16.astype(np.uint32) << 16).view(np.float32)
    elif dt == "F16":
        out = raw.view(np.float16).astype(np.float32)
    elif dt == "F32":
        out = np.array(raw.view(np.float32))
    else:
        sys.exit(f"unsupported dtype {dt}")
    return out.reshape(shape)


def extract(slug: str) -> None:
    cfg = MODELS[slug]
    out = os.path.join(OUT, slug)
    os.makedirs(out, exist_ok=True)
    if cfg.get("tokenizer") == "wordpiece":  # repo ships only vocab.txt: build a fast tokenizer from it
        from tokenizers import BertWordPieceTokenizer
        BertWordPieceTokenizer(hf_hub_download(cfg["hf"], "vocab.txt"), lowercase=False,
                               strip_accents=False).save(os.path.join(out, "tokenizer.json"))
    else:
        shutil.copy(hf_hub_download(cfg["hf"], "tokenizer.json"), os.path.join(out, "tokenizer.json"))
    w = read_tensor(hf_hub_download(cfg["hf"], "model.safetensors"), cfg["tensor"])
    print(slug, "embedding matrix", w.shape)
    np.save(os.path.join(out, "embeddings.f16.npy"), w.astype(np.float16))

    # PCA (top 3) via covariance eigendecomposition, accumulated in chunks to bound memory.
    mean = w.mean(axis=0, dtype=np.float64).astype(np.float32)
    cov = np.zeros((w.shape[1],) * 2, dtype=np.float64)
    for i in range(0, len(w), 16384):
        c = w[i:i + 16384] - mean
        cov += c.T.astype(np.float64) @ c.astype(np.float64)
    vals, vecs = np.linalg.eigh(cov / len(w))
    comps = vecs[:, ::-1][:, :3].astype(np.float32)
    pca = np.concatenate([(w[i:i + 16384] - mean) @ comps for i in range(0, len(w), 16384)])
    np.save(os.path.join(out, "pca3.npy"), pca.astype(np.float32))
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"slug": slug, "model_id": cfg["hf"], "label": cfg["label"],
                   "tensor": cfg["tensor"], "lang": cfg.get("lang", "en"), "vocab": w.shape[0], "dims": w.shape[1]}, fh)
    print("done ->", os.path.abspath(out))


if __name__ == "__main__":
    for s in (sys.argv[1:] or list(MODELS)):
        extract(s)
