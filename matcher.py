"""
matcher.py
----------
Core visual-similarity engine for the necklace -> earrings recommender.

Approach (no internet / no pretrained-model downloads required):
  1. Foreground segmentation with OpenCV GrabCut, so the (often coloured,
     sometimes textured) product-photo background doesn't pollute the
     comparison. We only assume the item is roughly centred in the frame,
     which holds for all 20 provided images.
  2. Two complementary, hand-crafted feature descriptors are extracted from
     the foreground pixels only:
       a) HSV colour histogram  -> captures metal tone + gemstone colour
          palette (gold vs silver/oxidised, green/red/pink stones, etc.)
          This is the primary signal for "does this look like it belongs
          to the same set" the way a jewellery stylist would judge it.
       b) HOG (Histogram of Oriented Gradients) on the masked grayscale
          image -> captures shape/silhouette style (studs vs jhumkis vs
          chandbali vs long drops, filigree density, etc.)
  3. Similarity between a necklace and a candidate earring = a weighted
     blend of colour-histogram correlation and HOG cosine similarity.
     Colour is weighted higher by default (COLOR_WEIGHT) because for
     jewellery, matching metal/gemstone tone is what visually reads as
     "goes together" first; shape is a secondary refinement.

Everything here runs on CPU in well under a second per image and has zero
external model downloads, so it works fully offline.
"""

from __future__ import annotations

import csv
import os
import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Dict, Tuple

import cv2
import numpy as np
from skimage.feature import hog

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

APP_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(APP_DIR, "static", "images")
CSV_PATH = os.path.join(APP_DIR, "candidate_dataset.csv")
CACHE_PATH = os.path.join(APP_DIR, ".feature_cache.pkl")

SEG_SIZE = 320          # image size used for GrabCut + colour histogram
HOG_SIZE = 128           # image size used for HOG (shape) descriptor
GRABCUT_ITERS = 5
BORDER_MARGIN_FRAC = 0.08  # assume item sits within the central 84% of frame

H_BINS, S_BINS, V_BINS = 30, 32, 32

COLOR_WEIGHT = 0.65       # weight of colour similarity in the final score
SHAPE_WEIGHT = 1 - COLOR_WEIGHT


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class Product:
    id: str
    product_type: str
    image_file: str

    @property
    def image_path(self) -> str:
        return os.path.join(IMAGES_DIR, self.image_file)


@dataclass
class Features:
    hist: np.ndarray            # flattened, normalised HSV histogram (foreground only)
    hog_vec: np.ndarray         # HOG descriptor (foreground-masked grayscale)
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)  # BGR, for display


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #

def load_catalog() -> List[Product]:
    products = []
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            products.append(Product(row["id"], row["product_type"].strip(), row["image_file"].strip()))
    return products


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #

def _foreground_mask(bgr: np.ndarray) -> np.ndarray:
    """GrabCut segmentation seeded with a centred rectangle.

    Falls back to 'everything is foreground' if GrabCut fails for any
    reason (e.g. a degenerate rectangle), so the pipeline never crashes.
    """
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    mx, my = int(w * BORDER_MARGIN_FRAC), int(h * BORDER_MARGIN_FRAC)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    try:
        cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, GRABCUT_ITERS, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        if fg.sum() < 0.02 * h * w:   # segmentation collapsed to (near) nothing
            fg = np.ones((h, w), np.uint8)
    except cv2.error:
        fg = np.ones((h, w), np.uint8)
    return fg


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #

def _dominant_colors(bgr: np.ndarray, mask: np.ndarray, k: int = 4) -> List[Tuple[int, int, int]]:
    pixels = bgr[mask.astype(bool)].reshape(-1, 3).astype(np.float32)
    if len(pixels) < k:
        return []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten())
    order = np.argsort(-counts)
    return [tuple(int(c) for c in centers[i]) for i in order]


def extract_features(image_path: str) -> Features:
    raw = cv2.imread(image_path)
    if raw is None:
        raise FileNotFoundError(image_path)

    seg_img = cv2.resize(raw, (SEG_SIZE, SEG_SIZE))
    fg_mask = _foreground_mask(seg_img)

    # --- colour histogram (foreground only) ---
    hsv = cv2.cvtColor(seg_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], fg_mask,
        [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256],
    )
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist = hist.flatten()

    # --- shape/texture (HOG on masked grayscale) ---
    hog_img = cv2.resize(raw, (HOG_SIZE, HOG_SIZE))
    hog_mask = cv2.resize(fg_mask, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(hog_img, cv2.COLOR_BGR2GRAY)
    gray = np.where(hog_mask.astype(bool), gray, 0).astype(np.uint8)
    hog_vec = hog(
        gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
        block_norm="L2-Hys", feature_vector=True,
    )

    dom = _dominant_colors(seg_img, fg_mask, k=4)

    return Features(hist=hist, hog_vec=hog_vec, dominant_colors=dom)


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def similarity(feat_a: Features, feat_b: Features) -> Dict[str, float]:
    # Correlation is naturally in [-1, 1]; rescale to [0, 1].
    color_corr = cv2.compareHist(
        feat_a.hist.astype(np.float32), feat_b.hist.astype(np.float32), cv2.HISTCMP_CORREL
    )
    color_score = (color_corr + 1) / 2

    shape_score = (_cosine(feat_a.hog_vec, feat_b.hog_vec) + 1) / 2

    combined = COLOR_WEIGHT * color_score + SHAPE_WEIGHT * shape_score
    return {"color": color_score, "shape": shape_score, "combined": combined}


# --------------------------------------------------------------------------- #
# Index (compute-once, reuse) + recommendation
# --------------------------------------------------------------------------- #

class CatalogIndex:
    """Loads the catalog and pre-computes features for every image once."""

    def __init__(self):
        self.products: List[Product] = load_catalog()
        self.by_id: Dict[str, Product] = {p.id: p for p in self.products}
        self.features: Dict[str, Features] = {}

        cache = self._load_cache()
        dirty = False
        for p in self.products:
            mtime = os.path.getmtime(p.image_path)
            cached = cache.get(p.id)
            if cached is not None and cached[0] == mtime:
                self.features[p.id] = cached[1]
            else:
                feat = extract_features(p.image_path)
                self.features[p.id] = feat
                cache[p.id] = (mtime, feat)
                dirty = True
        if dirty:
            self._save_cache(cache)

    @staticmethod
    def _load_cache() -> dict:
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}

    @staticmethod
    def _save_cache(cache: dict) -> None:
        try:
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(cache, f)
        except Exception:
            pass  # caching is a pure optimisation; never fail the request over it

    @property
    def necklaces(self) -> List[Product]:
        return [p for p in self.products if p.product_type == "Necklace"]

    @property
    def earrings(self) -> List[Product]:
        return [p for p in self.products if p.product_type == "Earrings"]

    def recommend(self, necklace_id: str, top_k: int = 3) -> List[dict]:
        if necklace_id not in self.by_id:
            raise KeyError(f"Unknown product id: {necklace_id}")
        query_feat = self.features[necklace_id]

        scored = []
        for e in self.earrings:
            s = similarity(query_feat, self.features[e.id])
            scored.append({
                "id": e.id,
                "image_file": e.image_file,
                "score": round(s["combined"], 4),
                "color_score": round(s["color"], 4),
                "shape_score": round(s["shape"], 4),
            })
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]


@lru_cache(maxsize=1)
def get_index() -> CatalogIndex:
    """Process-wide singleton so features are computed only once."""
    return CatalogIndex()
