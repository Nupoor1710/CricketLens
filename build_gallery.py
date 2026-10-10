"""Build the face gallery from the dataset folder (one folder per player, 1+ photos each).

Usage:  python build_gallery.py
Takes seconds to a few minutes. Run again after adding photos by hand.
(Photos added through the app's "Add a player" / "Teach it" buttons are indexed instantly.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from faceid.core import FaceEmbedder
from faceid.gallery import Gallery, slugify

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main(data_dir: str = "dataset") -> None:
    root = Path(data_dir)
    players = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    embedder = FaceEmbedder()
    names, vecs, skipped = [], [], 0
    for p in players:
        files = [f for f in p.iterdir() if f.suffix.lower() in IMG_EXT]
        for f in tqdm(files, desc=p.name, unit="img", leave=False):
            try:
                _, _, emb = embedder.faces(embedder.load_image(f), largest_only=True)
            except Exception:
                emb = []
            if len(emb) == 0:
                skipped += 1
                continue
            names.append(slugify(p.name))
            vecs.append(emb[0])
    if not names:
        sys.exit("No usable photos found. Put photos in dataset\\<player_name>\\ "
                 "(or run fetch_cricketers.py), then run this again.")
    g = Gallery()
    g.names, g.E = names, np.array(vecs, dtype=np.float32)
    g.save()
    print(f"\nGallery built: {len(set(names))} players, {len(names)} reference faces, {skipped} photos skipped.")
    thin = [n for n, c in g.people().items() if c < 3]
    if thin:
        print(f"Only 1-2 photos (works, but more is better): {', '.join(thin)}")
    print("Now run:  python app.py")


if __name__ == "__main__":
    main()
