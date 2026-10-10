"""Download reference photos of cricketers from Wikipedia + Wikimedia Commons (free-licensed images).

Usage:
    python fetch_cricketers.py                 # everyone in cricketers.txt
    python fetch_cricketers.py --max 8         # up to 8 photos per player
    python fetch_cricketers.py --only "Virat Kohli"

For each player it takes the article's lead photo as the anchor, then adds Commons photos ONLY if
the face matches the anchor, so photos of other people are discarded automatically.
Sources are written to dataset/<player>/SOURCES.txt. Credit them if you publish your work.
Afterwards run:  python build_gallery.py
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import requests

from faceid.core import FaceEmbedder
from faceid.gallery import slugify

UA = {"User-Agent": "CricketLens/1.0 (student project; educational use)"}
WIKI = "https://en.wikipedia.org/w/api.php"
COMMONS = "https://commons.wikimedia.org/w/api.php"


def api(url: str, **params) -> dict:
    r = requests.get(url, params={"format": "json", **params}, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()


def lead_image(title: str) -> str | None:
    d = api(WIKI, action="query", redirects=1, titles=title, prop="pageimages",
            piprop="thumbnail", pithumbsize=900)
    for page in d.get("query", {}).get("pages", {}).values():
        if page.get("thumbnail"):
            return page["thumbnail"]["source"]
    return None


def commons_images(name: str, limit: int = 25) -> list[str]:
    d = api(COMMONS, action="query", generator="search", gsrsearch=f"{name} cricketer filetype:bitmap",
            gsrnamespace=6, gsrlimit=limit, prop="imageinfo", iiprop="url|mime", iiurlwidth=900)
    urls = []
    for page in d.get("query", {}).get("pages", {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        if info.get("mime") in ("image/jpeg", "image/png"):
            urls.append(info.get("thumburl") or info.get("url"))
    return [u for u in urls if u]


def download(url: str) -> bytes:
    for attempt in range(3):
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code == 429:  # polite back-off if rate limited
            time.sleep(3 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.content
    raise RuntimeError("rate limited")


def fetch_player(embedder, display: str, title: str, out_root: Path, max_photos: int, min_sim: float) -> int:
    lead = lead_image(title)
    urls = ([lead] if lead else []) + [u for u in commons_images(display) if u != lead]
    cands = []  # (url, PIL image, embedding)
    for u in urls:
        try:
            img = embedder.load_image(io.BytesIO(download(u)))
            _, _, emb = embedder.faces(img, largest_only=True)
        except Exception:
            continue
        if len(emb):
            cands.append((u, img, emb[0]))
        time.sleep(0.3)
    if not cands:
        return 0
    E = np.array([c[2] for c in cands])
    ref = E[0] if (lead and cands[0][0] == lead) else E[(E @ E.T).mean(1).argmax()]
    keep = [c for c in cands if float(c[2] @ ref) >= min_sim][:max_photos]

    folder = out_root / slugify(display)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "SOURCES.txt", "w", encoding="utf-8") as src:
        for n, (u, img, _) in enumerate(keep):
            img.save(folder / f"wiki_{n:02d}.jpg", quality=92)
            src.write(f"wiki_{n:02d}.jpg  {u}\n")
    return len(keep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="cricketers.txt")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--max", type=int, default=8, help="max photos per player")
    ap.add_argument("--min-sim", type=float, default=0.55, help="face match needed to keep a Commons photo")
    ap.add_argument("--only", help="fetch just this player")
    ap.add_argument("--force", action="store_true", help="re-fetch players that already have photos")
    args = ap.parse_args()

    players = []
    for line in Path(args.list).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            display, _, title = (p.strip() for p in line.partition("|"))
            players.append((display, title or display))
    if args.only:
        players = [p for p in players if p[0].lower() == args.only.lower()] or [(args.only, args.only)]

    embedder = FaceEmbedder()
    out_root = Path(args.out)
    total = 0
    for i, (display, title) in enumerate(players, 1):
        folder = out_root / slugify(display)
        if folder.exists() and any(folder.glob("*.jpg")) and not args.force:
            print(f"[{i}/{len(players)}] {display}: already has photos, skipping (use --force)")
            continue
        try:
            n = fetch_player(embedder, display, title, out_root, args.max, args.min_sim)
        except Exception as e:
            print(f"[{i}/{len(players)}] {display}: failed ({e})")
            continue
        total += n
        print(f"[{i}/{len(players)}] {display}: {n} photo(s)" + ("" if n else "  (none found; add photos by hand)"))
    print(f"\nDone. {total} photos saved to {out_root}/. Now run:  python build_gallery.py")


if __name__ == "__main__":
    main()
