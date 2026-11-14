"""Loading, cosine similarity, neighbors, concept axes and analogies over a real embedding matrix."""
import json
import os
from functools import lru_cache

import numpy as np
from tokenizers import Tokenizer

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))

# Axis definitions: (positive terms, negative terms). Each term is a word; axis = mean(pos) - mean(neg).
# For pair axes the pairs are listed explicitly so the UI can show them.
NEUTRAL = ["table", "car", "house", "chair", "book", "door", "road", "paper"]
AXES = {
    "Femininity": {"pairs": [["queen", "king"], ["woman", "man"], ["she", "he"], ["girl", "boy"]]},
    "Royalty": {"pairs": [["king", "man"], ["queen", "woman"], ["prince", "boy"], ["crown", "hat"]]},
    "Fruit": {"pos": ["apple", "pear", "grape", "banana"], "neg": NEUTRAL},
    "Power": {"pairs": [["powerful", "weak"], ["strong", "weak"], ["ruler", "servant"]]},
}


class Lens:
    def __init__(self, data_dir: str = DATA_DIR):
        self.tok = Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))
        self.emb = np.load(os.path.join(data_dir, "embeddings.f16.npy"), mmap_mode="r")
        self.pca = np.load(os.path.join(data_dir, "pca3.npy"), mmap_mode="r")
        with open(os.path.join(data_dir, "meta.json")) as fh:
            self.meta = json.load(fh)
        self.vocab, self.dims = self.emb.shape
        # Row norms, computed in chunks to keep memory flat.
        self.norms = np.empty(self.vocab, dtype=np.float32)
        for i in range(0, self.vocab, 8192):
            self.norms[i:i + 8192] = np.linalg.norm(self.emb[i:i + 8192].astype(np.float32), axis=1)
        self.norms[self.norms == 0] = 1.0
        self._axes_cache = {}

    # ---- tokens
    def token_str(self, i: int) -> str:
        return self.tok.id_to_token(int(i)) or ""

    @staticmethod
    def display(tok: str) -> str:
        """GPT-2 byte-level BPE marks a leading space as 'Ġ' (and newline as 'Ċ')."""
        return tok.replace("Ġ", " ").replace("Ċ", "\\n")

    def tokenize(self, text: str):
        enc = self.tok.encode(text)
        out = []
        for tid, raw in zip(enc.ids, enc.tokens):
            out.append({
                "token": self.display(raw),
                "token_id": tid,
                "byte_repr": list(self.display(raw).encode("utf-8", errors="replace")),
                "leading_space": raw.startswith("Ġ"),
            })
        return out

    def vector(self, i: int) -> np.ndarray:
        return np.asarray(self.emb[int(i)], dtype=np.float32)

    def word_ids(self, word: str):
        """Return (ids, variant): prefer the ' word' (leading space) single token, else 'word'."""
        for cand in (" " + word, word):
            ids = self.tok.encode(cand).ids
            if len(ids) == 1:
                return ids
        return self.tok.encode(" " + word).ids  # multi-token word: caller averages

    def word_vec(self, word: str) -> np.ndarray:
        ids = self.word_ids(word)
        return np.mean([self.vector(i) for i in ids], axis=0)

    # ---- similarity
    def neighbors_of_vec(self, v: np.ndarray, k=10, exclude=()):
        v = v / (np.linalg.norm(v) or 1.0)
        sims = np.empty(self.vocab, dtype=np.float32)
        for i in range(0, self.vocab, 8192):
            sims[i:i + 8192] = self.emb[i:i + 8192].astype(np.float32) @ v
        sims /= self.norms
        for e in exclude:
            sims[int(e)] = -np.inf
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        return [{"id": int(t), "token": self.display(self.token_str(t)), "similarity": float(sims[t])} for t in top]

    def neighbors(self, i: int, k=10):
        return self.neighbors_of_vec(self.vector(i), k, exclude=[i])

    # ---- axes
    def axis_vector(self, defn: dict) -> np.ndarray:
        if "pairs" in defn:
            a = np.mean([self.word_vec(p) - self.word_vec(n) for p, n in defn["pairs"]], axis=0)
        else:
            a = np.mean([self.word_vec(w) for w in defn["pos"]], axis=0) - \
                np.mean([self.word_vec(w) for w in defn["neg"]], axis=0)
        return a / (np.linalg.norm(a) or 1.0)

    def axes_info(self):
        return {name: d for name, d in AXES.items()}

    def named_axis(self, name: str) -> np.ndarray:
        if name not in self._axes_cache:
            self._axes_cache[name] = self.axis_vector(AXES[name])
        return self._axes_cache[name]

    def score(self, v: np.ndarray, axis: np.ndarray) -> float:
        return float(v @ axis / (np.linalg.norm(v) or 1.0))

    def axes_scores(self, words, custom=None):
        """custom: {name: {"pos": [...], "neg": [...]}}"""
        axes = {n: self.named_axis(n) for n in AXES}
        for n, d in (custom or {}).items():
            axes[n] = self.axis_vector({"pos": d["pos"], "neg": d.get("neg") or NEUTRAL})
        return {w: {n: self.score(self.word_vec(w), a) for n, a in axes.items()} for w in words}

    # ---- analogy
    def analogy(self, a, b, c, k=10):
        v = self.word_vec(a) - self.word_vec(b) + self.word_vec(c)
        excl = set(self.word_ids(a) + self.word_ids(b) + self.word_ids(c))
        # also exclude the no-space variants of the inputs
        for w in (a, b, c):
            excl.update(self.tok.encode(w).ids)
        return self.neighbors_of_vec(v, k, exclude=excl)

    def pca_coords(self, ids):
        return [{"id": int(i), "token": self.display(self.token_str(i)),
                 "x": float(self.pca[i][0]), "y": float(self.pca[i][1]), "z": float(self.pca[i][2])} for i in ids]


@lru_cache(maxsize=1)
def get_lens() -> Lens:
    return Lens()
