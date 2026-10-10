"""Face detection (MTCNN) and embedding (FaceNet / InceptionResnetV1) helpers.

Pipeline:  image -> detect faces -> align + crop -> 512-d embedding (L2-normalised)
The embeddings are what the classifier (SVM / KNN / LogReg) is trained on.
"""
from __future__ import annotations

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageOps

from .core_names import pretty_name  # noqa: F401  (re-exported)

MAX_SIDE = 1600  # large photos are downscaled for speed; boxes are returned normalised


class FaceEmbedder:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mtcnn = MTCNN(
            image_size=160, margin=20, keep_all=True, post_process=True, device=self.device
        )
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    @staticmethod
    def load_image(path_or_file) -> Image.Image:
        img = Image.open(path_or_file)
        img = ImageOps.exif_transpose(img).convert("RGB")
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE))
        return img

    def faces(self, img: Image.Image, largest_only: bool = False, min_prob: float = 0.90):
        """Return (boxes[N,4] in pixels, detection_probs[N], embeddings[N,512])."""
        boxes, probs = self.mtcnn.detect(img)
        empty = (np.empty((0, 4)), np.empty((0,)), np.empty((0, 512)))
        if boxes is None:
            return empty

        keep = [i for i, p in enumerate(probs) if p is not None and p >= min_prob]
        if not keep:
            return empty
        if largest_only:
            area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
            keep = [max(keep, key=lambda i: probs[i] * area(boxes[i]))]

        boxes, probs = boxes[keep], probs[keep]
        crops = self.mtcnn.extract(img, boxes, save_path=None)  # (N, 3, 160, 160)
        if crops.ndim == 3:
            crops = crops.unsqueeze(0)
        with torch.no_grad():
            emb = self.net(crops.to(self.device)).cpu().numpy()
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return boxes, probs, emb
