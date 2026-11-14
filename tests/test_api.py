import os, sys
import numpy as np
import pytest
from fastapi.testclient import TestClient
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.main import app  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
c = TestClient(app)
tok = Tokenizer.from_file(os.path.join(DATA, "gpt2", "tokenizer.json"))


def test_healthz():
    assert c.get("/healthz").status_code == 200


def test_tokenize_matches_real_tokenizer():
    r = c.get("/api/tokenize", params={"text": "king queen man woman apple"}).json()
    assert [t["token_id"] for t in r] == tok.encode("king queen man woman apple").ids
    assert r[0]["token"] == "king" and r[1]["token"] == " queen" and r[1]["leading_space"]


def test_space_variants_differ():
    assert tok.encode("king").ids != tok.encode(" king").ids
    assert tok.encode(" king").ids == [5822]


def test_vector_768():
    t = c.get("/api/token/5822").json()
    assert t["dims"] == 768 and len(t["vector"]) == 768 and t["token"] == " king"


def test_neighbors():
    r = c.get("/api/neighbors/5822?k=5").json()
    assert len(r) == 5 and all(n["id"] != 5822 for n in r)


def test_analogy_queen_top5():
    r = c.get("/api/analogy", params={"a": "king", "b": "man", "c": "woman", "k": 5}).json()
    assert any(x["token"].strip() == "queen" for x in r), r


def test_femininity_axis():
    s = c.get("/api/axes", params={"tokens": "king,queen,man,woman"}).json()["scores"]
    assert s["queen"]["Femininity"] > s["king"]["Femininity"]
    assert s["woman"]["Femininity"] > s["man"]["Femininity"]


def test_fruit_axis():
    s = c.get("/api/axes", params={"tokens": "apple,pear,grape,king,queen"}).json()["scores"]
    lo = max(s["king"]["Fruit"], s["queen"]["Fruit"])
    for f in ("apple", "pear", "grape"):
        assert s[f]["Fruit"] > lo + 0.1, (f, s[f]["Fruit"], lo)


def test_pca_and_bad_id():
    assert len(c.get("/api/pca?ids=5822,14710").json()) == 2
    assert c.get("/api/token/999999").status_code == 404


def test_models_listed():
    r = c.get("/api/models").json()
    assert r["default"] == "gpt2" and any(m["slug"] == "gpt2" for m in r["models"])


def test_unknown_model_404():
    assert c.get("/api/tokenize", params={"text": "a", "model": "nope"}).status_code == 404


def test_same_id_differs_across_models():
    # id 9246 is 'cat' in GPT-2, not 'king' - IDs are tokenizer-specific.
    assert c.get("/api/token/9246").json()["token"] == "cat"
    slugs = {m["slug"] for m in c.get("/api/models").json()["models"]}
    if "qwen2.5-0.5b" not in slugs:
        pytest.skip("qwen data not extracted")
    q = c.get("/api/tokenize", params={"text": "king", "model": "qwen2.5-0.5b"}).json()
    assert q[0]["token_id"] == 10566 != 3364
    g = c.get("/api/token/10566", params={"model": "qwen2.5-0.5b"}).json()
    assert g["dims"] == 896 and len(g["vector"]) == 896


def test_stats():
    r = c.get("/api/stats").json()
    assert r["rss_mb"] > 0 and r["mem_total_mb"] > 0 and "cpu_proc" in r


def test_turkish_model():
    slugs = {m["slug"] for m in c.get("/api/models").json()["models"]}
    if "bert-turkish" not in slugs:
        pytest.skip("bert-turkish data not extracted")
    t = c.get("/api/tokenize", params={"text": "kral kadın elma", "model": "bert-turkish"}).json()
    assert [x["token"] for x in t] == ["kral", "kadın", "elma"]
    assert c.get("/api/meta", params={"model": "bert-turkish"}).json()["lang"] == "tr"
    s = c.get("/api/axes", params={"tokens": "kral,kraliçe,adam,kadın,elma,armut,üzüm", "model": "bert-turkish"}).json()["scores"]
    assert s["kadın"]["Femininity"] > s["adam"]["Femininity"] and s["kraliçe"]["Femininity"] > s["kral"]["Femininity"]
    assert min(s[f]["Fruit"] for f in ("elma", "armut", "üzüm")) > max(s["kral"]["Fruit"], s["kraliçe"]["Fruit"]) + 0.1
    r = c.get("/api/analogy", params={"a": "kral", "b": "adam", "c": "kadın", "k": 5, "model": "bert-turkish"}).json()
    assert any(x["token"].lower().startswith("kraliçe") for x in r)


def test_language_tax():
    r = c.get("/api/language-tax").json()
    en = next(x for x in r["rows"] if x["code"] == "en")
    hi = next(x for x in r["rows"] if x["code"] == "hi")
    assert en["models"]["gpt2"]["ratio"] == 1.0
    assert hi["models"]["gpt2"]["tokens"] > 2 * en["models"]["gpt2"]["tokens"]
    assert c.get("/language-tax").status_code == 200 and c.get("/static/language-tax.jpg").status_code == 200


def test_claude_estimates_are_flagged():
    r = c.get("/api/language-tax").json()
    est = [m for m in r["models"] if m.get("estimate")]
    assert {m["slug"] for m in est} == {"claude-fable-5-2", "claude-opus-5-5"}
    assert all("ESTIMATE" in m["label"] for m in est)
    for row in r["rows"]:
        assert row["models"]["claude-opus-5-5"]["estimate"] is True
    by = {x["code"]: x["models"]["claude-opus-5-5"] for x in r["rows"]}
    assert by["en"]["ratio"] == 1.0 and by["es"]["ratio"] > 1.3 and 1.1 < by["zh"]["ratio"] < 1.2
    assert by["hi"]["ratio"] >= 2.5 and by["ko"]["basis"] == "reported" and by["tr"]["basis"] == "assumed"


def test_top5_taxed_languages_are_indic():
    # The page text claims this for the measured tokenizers; keep the claim honest.
    rows = c.get("/api/language-tax").json()["rows"]
    meas = [k for k in ("gpt2", "qwen2.5-0.5b", "gemma-3-270m") if k in rows[0]["models"]]
    if len(meas) < 3:
        pytest.skip("needs all three general-purpose models extracted")
    top = sorted(rows, key=lambda r: -sum(r["models"][k]["ratio"] for k in meas))[:5]
    assert all(r["indic"] for r in top), [r["language"] for r in top]
