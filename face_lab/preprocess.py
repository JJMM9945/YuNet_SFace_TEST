"""Android port boundary: decode/orient -> BGR -> detection resize -> coordinates -> align -> feature."""
from io import BytesIO
import cv2
import numpy as np
from PIL import Image, ImageOps


def decode_oriented(data):
    # Pillow does NOT auto-rotate on open. Apply EXIF exactly once, then discard metadata.
    with Image.open(BytesIO(data)) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def detection_input(bgr, max_side):
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / max(w, h))
    dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA) if (dw, dh) != (w, h) else bgr
    return small, w / dw, h / dh


def to_original(row, sx, sy):
    mapped = np.asarray(row, dtype=np.float32).copy()
    mapped[[0, 2, 4, 6, 8, 10, 12]] *= sx
    mapped[[1, 3, 5, 7, 9, 11, 13]] *= sy
    return mapped


def extract(recognizer, bgr, row):
    crop = recognizer.alignCrop(bgr, row)
    feature = recognizer.feature(crop).reshape(-1).astype(np.float32).copy()
    norm = np.linalg.norm(feature)
    if feature.size != 128 or not np.isfinite(feature).all() or not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid_feature_vector")
    return crop, feature / norm


def save_bgr(path, bgr):
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path, quality=92)
