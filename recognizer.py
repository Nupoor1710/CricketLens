"""Identifies faces by matching them against the gallery, and learns new faces on request."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from PIL import Image

from .core import FaceEmbedder, pretty_name
from .gallery import Gallery, score_from_sim, slugify

DATASET = Path("dataset")


def _area(b) -> float:
    return float((b[2] - b[0]) * (b[3] - b[1]))


class Recognizer:
    def __init__(self, gallery: Gallery) -> None:
        self.gallery = gallery
        self.embedder = FaceEmbedder()
        self.lock = threading.Lock()

    def identify(self, file) -> dict:
        img = self.embedder.load_image(file)
        w, h = img.size
        boxes, det, emb = self.embedder.faces(img)
        faces = []
        if len(emb) and len(self.gallery.names):
            for box, d, top in zip(boxes, det, self.gallery.match(emb)):
                faces.append(
                    {
                        "box": [float(box[0] / w), float(box[1] / h), float(box[2] / w), float(box[3] / h)],
                        "detection": float(d),
                        "top": [
                            {"name": pretty_name(n), "prob": score_from_sim(s), "sim": s}
                            for n, s in top
                        ],
                    }
                )
        elif len(emb):  # faces found but nobody in the gallery yet
            for box, d in zip(boxes, det):
                faces.append({"box": [float(box[0] / w), float(box[1] / h), float(box[2] / w), float(box[3] / h)],
                              "detection": float(d), "top": [{"name": "Nobody yet", "prob": 0.0, "sim": 0.0}]})
        faces.sort(key=lambda f: -(f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
        return {"width": w, "height": h, "faces": faces}

    def teach(self, files, name: str, face: int | None = None) -> dict:
        """Add faces from the uploaded photos to a player (created if new)."""
        slug = slugify(name)
        if not slug:
            raise ValueError("Please enter a player name.")
        folder = DATASET / slug
        added = skipped = 0
        with self.lock:
            for f in files:
                try:
                    img = self.embedder.load_image(f)
                    boxes, _, emb = self.embedder.faces(img)
                except Exception:
                    skipped += 1
                    continue
                if len(emb) == 0:
                    skipped += 1
                    continue
                order = sorted(range(len(boxes)), key=lambda i: -_area(boxes[i]))
                i = order[face] if face is not None and 0 <= face < len(order) else order[0]
                x1, y1, x2, y2 = boxes[i]
                m = 0.3 * max(x2 - x1, y2 - y1)  # keep some context around the face
                crop = img.crop((max(0, x1 - m), max(0, y1 - m), min(img.width, x2 + m), min(img.height, y2 + m)))
                folder.mkdir(parents=True, exist_ok=True)
                crop.save(folder / f"added_{int(time.time() * 1000)}_{added}.jpg", quality=92)
                self.gallery.add(slug, emb[i])
                added += 1
            if added:
                self.gallery.save()
        return {"name": pretty_name(slug), "added": added, "skipped": skipped}
