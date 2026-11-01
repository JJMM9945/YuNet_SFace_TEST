# 역할: 여러 크기 YuNet 검출과 원본 좌표 복원 및 중복 상자 제거
"""Optional multiscale YuNet detection in oriented original pixel coordinates."""
import numpy as np
from .preprocess import detection_input, to_original

MULTISCALE_VERSION = 'yunet-multiscale-global-nms-v1'


def multiscale_detect(detector, bgr, sides, nms_threshold):
    rows, inputs, seen = [], [], set()
    for side in sorted(set(sides)):
        small, sx, sy = detection_input(bgr, side)
        h, w = small.shape[:2]
        if (w, h) in seen:
            continue
        seen.add((w, h))
        inputs.append(dict(max_side=side, width=w, height=h, scale_x=sx, scale_y=sy))
        detector.setInputSize((w, h))
        _, found = detector.detect(small)
        if found is not None:
            rows.extend(to_original(r, sx, sy) for r in found)
    ordered = sorted(rows, key=lambda r: (-float(r[14]), *map(float, r[:4])))
    kept = []
    for row in ordered:
        if not np.isfinite(row).all() or min(row[2:4]) <= 0:
            continue
        suppress = False
        for old in kept:
            lo = np.maximum(row[:2], old[:2])
            hi = np.minimum(row[:2]+row[2:4], old[:2]+old[2:4])
            inter = float(np.prod(np.maximum(0, hi-lo)))
            union = float(row[2]*row[3]+old[2]*old[3])-inter
            if union > 0 and inter/union > nms_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(row)
    return kept, inputs
