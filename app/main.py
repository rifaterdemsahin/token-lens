import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .embeddings import AXES, NEUTRAL, available_models, get_lens as _get_lens
from .models import DEFAULT_MODEL

STATIC = os.path.join(os.path.dirname(__file__), "static")
app = FastAPI(title="Token Lens")


def get_lens(model: str = DEFAULT_MODEL):
    try:
        return _get_lens(model)
    except KeyError:
        raise HTTPException(404, f"unknown or unavailable model {model!r}")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/api/meta")
def meta(model: str = DEFAULT_MODEL):
    L = get_lens(model)
    return {**L.meta, "axes": AXES, "neutral": NEUTRAL}


@app.get("/api/models")
def models():
    return {"default": DEFAULT_MODEL, "models": available_models()}


@app.get("/api/tokenize")
def tokenize(text: str = Query(..., max_length=2000), model: str = DEFAULT_MODEL):
    return get_lens(model).tokenize(text)


@app.get("/api/words")
def words(words: str, model: str = DEFAULT_MODEL):
    """Resolve words to token variants (' king' vs 'king') so the UI can show both."""
    L = get_lens(model)
    out = []
    for w in [x.strip() for x in words.split(",") if x.strip()][:50]:
        row = {"word": w}
        for key, cand in (("with_space", " " + w), ("no_space", w)):
            ids = L.enc(cand).ids
            row[key] = {"ids": ids, "tokens": [L.display(L.token_str(i)) for i in ids]}
        out.append(row)
    return out


@app.get("/api/token/{token_id}")
def token(token_id: int, model: str = DEFAULT_MODEL):
    L = get_lens(model)
    if not 0 <= token_id < L.vocab:
        raise HTTPException(404, "token id out of range")
    v = L.vector(token_id)
    return {"token": L.display(L.token_str(token_id)), "token_id": token_id, "dims": L.dims,
            "vector": [float(x) for x in v], "norm": float((v ** 2).sum() ** 0.5)}


@app.get("/api/neighbors/{token_id}")
def neighbors(token_id: int, k: int = Query(10, ge=1, le=50), model: str = DEFAULT_MODEL):
    L = get_lens(model)
    if not 0 <= token_id < L.vocab:
        raise HTTPException(404, "token id out of range")
    return L.neighbors(token_id, k)


@app.get("/api/axes")
def axes(tokens: str, custom_name: Optional[str] = None, pos: Optional[str] = None, neg: Optional[str] = None, model: str = DEFAULT_MODEL):
    L = get_lens(model)
    ws = [t.strip() for t in tokens.split(",") if t.strip()][:60]
    custom = None
    if custom_name and pos:
        custom = {custom_name: {"pos": [p.strip() for p in pos.split(",") if p.strip()],
                                "neg": [n.strip() for n in (neg or "").split(",") if n.strip()]}}
    return {"axes": L.axes_info(), "scores": L.axes_scores(ws, custom)}


@app.get("/api/analogy")
def analogy(a: str, b: str, c: str, k: int = Query(10, ge=1, le=30), model: str = DEFAULT_MODEL):
    return get_lens(model).analogy(a, b, c, k)


@app.get("/api/pca")
def pca(ids: str, model: str = DEFAULT_MODEL):
    L = get_lens(model)
    try:
        idl = [int(i) for i in ids.split(",") if i.strip()][:200]
    except ValueError:
        raise HTTPException(400, "ids must be integers")
    if any(not 0 <= i < L.vocab for i in idl):
        raise HTTPException(404, "token id out of range")
    return L.pca_coords(idl)


@app.get("/models")
def models_page():
    return FileResponse(os.path.join(STATIC, "models.html"))


@app.get("/architecture")
def architecture():
    return FileResponse(os.path.join(STATIC, "architecture.html"))


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")
