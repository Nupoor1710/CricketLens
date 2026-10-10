"""Train the cricketer face classifier.

Usage:
    python train.py                    # uses ./dataset, caches embeddings
    python train.py --rebuild          # re-extract embeddings (after adding photos)
    python train.py --data my_folder

Dataset layout:
    dataset/<player_name>/*.jpg       e.g. dataset/virat_kohli/img01.jpg
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tqdm import tqdm

from faceid.core import FaceEmbedder

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODELS = Path("models")


def check_dataset(data_dir: Path) -> None:
    """Count photos per player and stop early with a clear message if there are too few."""
    if not data_dir.exists():
        raise SystemExit(f"Folder '{data_dir}' not found.")
    print("Photos found per player folder:")
    usable = 0
    for p in sorted(d for d in data_dir.iterdir() if d.is_dir()):
        n = sum(1 for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        flag = "" if n >= 6 else "   <-- needs at least 6 (aim for 40+)"
        print(f"  {p.name:20s} {n:4d}{flag}")
        usable += n >= 6
    if usable < 2:
        raise SystemExit(
            "\nNeed at least 2 player folders with 6+ photos each. "
            "Add photos to dataset\\<player_name>\\ and run again."
        )
    print()


def extract_embeddings(data_dir: Path, cache: Path, rebuild: bool):
    if cache.exists() and not rebuild:
        z = np.load(cache, allow_pickle=True)
        if len(z["y"]) > 0:
            print(f"Loaded cached embeddings: {len(z['y'])} faces  (use --rebuild to refresh)")
            return z["X"], z["y"], int(z["skipped"])

    embedder = FaceEmbedder()
    X, y, skipped = [], [], 0
    players = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not players:
        raise SystemExit(f"No player folders found in {data_dir}/")

    for player in players:
        files = [f for f in player.iterdir() if f.suffix.lower() in IMG_EXT]
        for f in tqdm(files, desc=player.name, unit="img"):
            try:
                img = embedder.load_image(f)
                _, _, emb = embedder.faces(img, largest_only=True)
                if len(emb) == 0:
                    skipped += 1  # no clear face -> leave it out of training
                    continue
                X.append(emb[0])
                y.append(player.name)
            except Exception as exc:  # corrupt file etc.
                skipped += 1
                print(f"  skipped {f.name}: {exc}")

    X, y = np.array(X), np.array(y)
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, X=X, y=y, skipped=skipped)
    return X, y, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    t0 = time.time()
    check_dataset(Path(args.data))
    X, y, skipped = extract_embeddings(Path(args.data), MODELS / "embeddings.npz", args.rebuild)

    counts = Counter(y)
    print("\nFaces used per player:", dict(counts))
    if len(counts) < 2:
        raise SystemExit("Need at least 2 players with photos.")
    smallest = min(counts.values())
    if smallest < 6:
        raise SystemExit(f"Every player needs at least 6 usable photos (smallest has {smallest}).")
    if smallest < 30:
        print("Tip: 30-100 photos per player gives noticeably better results.")

    # 1) compare candidate classifiers with cross-validation
    candidates = {
        "SVM (linear)": SVC(kernel="linear", probability=True, random_state=42),
        "KNN (k=3, cosine)": KNeighborsClassifier(n_neighbors=3, metric="cosine", weights="distance"),
        "Logistic regression": LogisticRegression(max_iter=2000),
    }
    folds = StratifiedKFold(n_splits=min(5, smallest), shuffle=True, random_state=42)
    comparison = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X, y, cv=folds)
        comparison[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
        print(f"  {name:22s} CV accuracy {scores.mean():.3f} +/- {scores.std():.3f}")
    best_name = max(comparison, key=lambda k: comparison[k]["mean"])
    print(f"Best: {best_name}")

    # 2) honest held-out evaluation of the winner
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )
    best = candidates[best_name]
    best.fit(Xtr, ytr)
    pred = best.predict(Xte)
    classes = sorted(counts)
    report = classification_report(yte, pred, labels=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=classes)
    accuracy = float((pred == yte).mean())
    print(f"\nHeld-out accuracy: {accuracy:.3f}  ({len(yte)} test faces)")

    # 3) refit on everything for the deployed model
    best.fit(X, y)
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"clf": best, "classes": list(best.classes_)}, MODELS / "classifier.joblib")

    metrics = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "model": best_name,
        "accuracy": accuracy,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "skipped": skipped,
        "classes": classes,
        "counts": {c: int(counts[c]) for c in classes},
        "report": {c: report[c] for c in classes},
        "confusion": cm.tolist(),
        "comparison": comparison,
    }
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved models/classifier.joblib and models/metrics.json  ({time.time() - t0:.0f}s)")
    print("Now run:  python app.py")


if __name__ == "__main__":
    main()
