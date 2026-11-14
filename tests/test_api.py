import os, sys
import numpy as np
import pytest
from fastapi.testclient import TestClient
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.main import app  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
c = TestClient(app)
tok = Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))


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
