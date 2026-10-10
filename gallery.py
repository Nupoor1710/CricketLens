"""A growing gallery of face fingerprints (embeddings), one or more per player.

Identification = find the gallery face most similar to the query face. Adding a new player
just means adding their fingerprints. No retraining, which is what makes the app dynamic.
Pure numpy: importing this does not load PyTorch.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np

GALLERY_PATH = Path("models/gallery.npz")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60]


def score_from_sim(sim: float) -> float:
    """Map cosine similarity to a 0-1 'match strength' (same person is usually 0.6-0.9)."""
    return float(min(1.0, max(0.0, (sim - 0.2) / 0.6)))


class Gallery:
    def __init__(self, path: Path = GALLERY_PATH) -> None:
        self.path = path
        self.names: list[str] = []
        self.E = np.empty((0, 512), dtype=np.float32)
        if path.exists():
            z = np.load(path, allow_pickle=True)
            self.names = [str(n) for n in z["names"]]
            self.E = z["E"].astype(np.float32)

    def people(self) -> dict[str, int]:
        return dict(sorted(Counter(self.names).items()))

    def add(self, slug: str, emb: np.ndarray) -> None:
        self.names.append(slug)
        self.E = np.vstack([self.E, emb.reshape(1, -1).astype(np.float32)])

    def save(self) -> None:
        self.path.parent.mkdir(exist_ok=True)
        np.savez(self.path, names=np.array(self.names), E=self.E)

    def match(self, emb: np.ndarray, k: int = 3) -> list[list[tuple[str, float]]]:
        """For each query embedding: the k best players as (slug, best similarity)."""
        sims = emb @ self.E.T
        results = []
        for row in sims:
            best: dict[str, float] = {}
            for n, s in zip(self.names, row):
                if s > best.get(n, -1.0):
                    best[n] = float(s)
            results.append(sorted(best.items(), key=lambda t: -t[1])[:k])
        return results
