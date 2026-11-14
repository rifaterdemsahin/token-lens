import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .embeddings import available_models, get_tokenizer, get_lens as _get_lens
from .models import DEFAULT_MODEL

STATIC = os.path.join(os.path.dirname(__file__), "static")
import time

import psutil

app = FastAPI(title="Token Lens")
_proc = psutil.Process()
_proc.cpu_percent(None)  # prime so the next call measures since this point
psutil.cpu_percent(None)


def get_lens(model: str = DEFAULT_MODEL):
    try:
        return _get_lens(model)
    except KeyError:
        raise HTTPException(404, f"unknown or unavailable model {model!r}")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/api/stats")
def stats():
    """Live server resource usage (polled by the UI)."""
    vm = psutil.virtual_memory()
    mi = _proc.memory_full_info() if hasattr(_proc, "memory_full_info") else _proc.memory_info()
    return {"t": time.time(), "cpu_proc": _proc.cpu_percent(None), "cpu_sys": psutil.cpu_percent(None),
            "cpus": psutil.cpu_count(), "rss_mb": mi.rss / 2**20,
            "mem_used_mb": (vm.total - vm.available) / 2**20, "mem_total_mb": vm.total / 2**20,
            "mem_pct": vm.percent, "models_loaded": _get_lens.cache_info().currsize}


SENTENCES = {  # the same sentence in 14 languages (hand-written parallel text)
    "en": ("English", "Hello, my name is Sarah and I would like to order two coffees, please."),
    "es": ("Spanish", "Hola, me llamo Sarah y me gustaría pedir dos cafés, por favor."),
    "tr": ("Turkish", "Merhaba, benim adım Sarah ve iki kahve sipariş etmek istiyorum, lütfen."),
    "ko": ("Korean", "안녕하세요, 제 이름은 사라입니다. 커피 두 잔을 주문하고 싶습니다."),
    "ja": ("Japanese", "こんにちは、サラです。コーヒーを二つ注文したいのですが、お願いします。"),
    "zh": ("Chinese", "你好，我叫莎拉，我想点两杯咖啡，谢谢。"),
    "ru": ("Russian", "Здравствуйте, меня зовут Сара, и я хотела бы заказать два кофе, пожалуйста."),
    "el": ("Greek", "Γεια σας, με λένε Σάρα και θα ήθελα να παραγγείλω δύο καφέδες, παρακαλώ."),
    "hi": ("Hindi", "नमस्ते, मेरा नाम सारा है और मैं दो कॉफ़ी ऑर्डर करना चाहती हूँ, कृपया।"),
    "bn": ("Bengali", "হ্যালো, আমার নাম সারা এবং আমি দুটি কফি অর্ডার করতে চাই, দয়া করে।"),
    "ta": ("Tamil", "வணக்கம், என் பெயர் சாரா, எனக்கு இரண்டு காபி ஆர்டர் செய்ய வேண்டும், தயவுசெய்து."),
    "te": ("Telugu", "నమస్కారం, నా పేరు సారా, దయచేసి నాకు రెండు కాఫీలు ఆర్డర్ చేయాలి."),
    "mr": ("Marathi", "नमस्कार, माझे नाव सारा आहे आणि मला दोन कॉफी मागवायच्या आहेत, कृपया."),
    "th": ("Thai", "สวัสดี ฉันชื่อซาร่า ฉันอยากสั่งกาแฟสองแก้ว ค่ะ"),
}


# Anthropic's tokenizer is not public, so Claude counts are ESTIMATES (not measured).
# English baseline = chars / 3.5 (Anthropic's rough guidance) x 1.30 (reported ~30% more tokens for the
# generation-5 tokenizer; unverified). Other languages = baseline x a ratio taken from third-party
# reported ranges where one exists ("reported", midpoint used), else an assumption ("assumed").
ESTIMATED = {"claude-fable-5-2": "Claude Fable 5.2", "claude-opus-5-5": "Claude Opus 5.5"}
GEN5_FACTOR = 1.30
EST_RATIO = {  # code: (ratio vs English, basis, note)
    "en": (1.0, "baseline", ""), "es": (1.43, "reported", "reported 1.42-1.45x for European text"),
    "zh": (1.15, "reported", "reported ~1.15x"), "ko": (2.25, "reported", "reported 1.5-3.0x, midpoint"),
    "ja": (2.25, "reported", "reported 1.5-3.0x, midpoint"), "hi": (3.25, "reported", "reported 2.5-4x+, midpoint"),
    "bn": (3.25, "reported", "reported 2.5-4x+, midpoint"),
    "ta": (3.25, "assumed", "no reported figure; Hindi/Bengali range assumed"),
    "te": (3.25, "assumed", "no reported figure; Hindi/Bengali range assumed"),
    "mr": (3.25, "assumed", "no reported figure; Hindi/Bengali range assumed"),
    "tr": (1.8, "assumed", "no reported figure"), "ru": (1.9, "assumed", "no reported figure"),
    "el": (1.9, "assumed", "no reported figure"), "th": (2.5, "assumed", "no reported figure"),
}


def estimate_claude_tokens(code: str, english_text: str) -> int:
    base = len(english_text) / 3.5 * GEN5_FACTOR
    return max(1, round(base * EST_RATIO[code][0]))


@app.get("/api/language-tax")
def language_tax():
    """Token counts for the same sentence per language per model, relative to English."""
    models = available_models()
    models_out = models + [{"slug": k, "label": v + " - ESTIMATE (not measured)", "hf": None, "estimate": True}
                           for k, v in ESTIMATED.items()]
    toks = {m["slug"]: get_tokenizer(m["slug"]) for m in models}
    rows = []
    for code, (name, text) in SENTENCES.items():
        row = {"code": code, "language": name, "indic": code in ("hi", "bn", "ta", "te", "mr"), "text": text, "chars": len(text),
               "bytes": len(text.encode("utf-8")), "models": {}}
        for slug, t in toks.items():
            enc = t.encode(text, add_special_tokens=False)
            row["models"][slug] = {"tokens": len(enc.ids),
                                   "unk": sum(1 for x in enc.tokens if x in ("[UNK]", "<unk>"))}
        for k in ESTIMATED:
            row["models"][k] = {"tokens": estimate_claude_tokens(code, SENTENCES["en"][1]), "unk": 0, "estimate": True,
                                "basis": EST_RATIO[code][1], "basis_note": EST_RATIO[code][2]}
        rows.append(row)
    en = rows[0]["models"]
    for r in rows:
        for slug, v in r["models"].items():
            v["ratio"] = v["tokens"] / en[slug]["tokens"]
    return {"models": models_out, "rows": rows,
            "estimate_note": "Claude tokenizer is not public; counts for Claude models are heuristic estimates."}


@app.get("/language-tax")
def language_tax_page():
    return FileResponse(os.path.join(STATIC, "language-tax.html"))


@app.get("/api/meta")
def meta(model: str = DEFAULT_MODEL):
    L = get_lens(model)
    return {**L.meta, "axes": L.axes, "neutral": L.neutral}


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
