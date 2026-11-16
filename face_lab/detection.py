# 역할: 여러 크기 YuNet 검출과 원본 좌표 복원 및 중복 상자 제거
"""Optional multiscale YuNet detection in oriented original pixel coordinates."""
import numpy as np
from .preprocess import detection_input, to_original

MULTISCALE_VERSION = 'yunet-multiscale-global-nms-v1'
DETECTION_POLICY_VERSION = 'yunet-conditional1280-preserve-v1'


def box_iou(first, second):
    lo = np.maximum(first[:2], second[:2])
    hi = np.minimum(first[:2] + first[2:4], second[:2] + second[2:4])
    intersection = float(np.prod(np.maximum(0, hi - lo)))
    union = float(first[2] * first[3] + second[2] * second[3]) - intersection
    return intersection / union if union > 0 else 0.0


def detect_with_policy(detector, bgr, settings):
    """Preserve primary rows; add non-overlapping supplemental rows before SFace.

    Counts before quality filtering trigger the second pass. No result from the
    supplemental pass can replace primary boxes, scores or landmarks.
    """
    primary, inputs = multiscale_detect(detector, bgr, settings.detection_sides, settings.nms_threshold)
    triggered = settings.conditional_detection and (len(primary) == 0 or len(primary) >= settings.conditional_min_faces)
    kept = list(primary)
    supplemental, added = [], 0
    if triggered:
        supplemental, extra_inputs = multiscale_detect(
            detector, bgr, settings.supplemental_detection_sides, settings.nms_threshold)
        inputs.extend(extra_inputs)
        for candidate in sorted(supplemental, key=lambda row: (-float(row[14]), *map(float, row[:4]))):
            if all(box_iou(candidate, old) <= settings.nms_threshold for old in kept):
                kept.append(candidate)
                added += 1
    return kept, dict(policy_version=DETECTION_POLICY_VERSION,
                      primary_faces=len(primary), conditional_triggered=bool(triggered),
                      supplemental_faces=len(supplemental), added_faces=added,
                      trigger_reason=('disabled' if not settings.conditional_detection else
                                      'zero_faces' if not primary else
                                      'at_least_min_faces' if triggered else 'not_triggered'),
                      actual_detection_inputs=inputs)


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
