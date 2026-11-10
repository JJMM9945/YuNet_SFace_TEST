"""Selected on-device port reference: no reports, network or identity experiments."""
from dataclasses import dataclass
import cv2
import numpy as np

from .detection import multiscale_detect
from .models import load
from .preprocess import decode_oriented, extract
from .grouping import group_faces


@dataclass(frozen=True)
class Settings:
    detection_sides: tuple = (960, 1600, 2400)
    score_threshold: float = .75
    nms_threshold: float = .30
    top_k: int = 5000
    min_face_size: float = 40
    min_blur: float = 50
    similarity_threshold: float = .40
    ambiguity_margin: float = .01
    min_group_size: int = 1
    linkage: str = 'average'


class FaceEngine:
    """Use one instance on a serial worker; OpenCV detector input size is mutable."""

    def __init__(self, models='models', settings=None):
        self.settings = settings or Settings()
        cv2.setNumThreads(1)
        cv2.setRNGSeed(0)
        cv2.ocl.setUseOpenCL(False)
        s = self.settings
        self.detector, self.recognizer = load(models, s.score_threshold, s.nms_threshold, s.top_k)

    def extract_photo(self, encoded, photo_id):
        """Return face records and normalized vectors keyed by face_id.

        Caller owns stable photo IDs, persistence, change detection and job resumption.
        Quality exclusions retain successfully extracted vectors but cannot group.
        """
        bgr = decode_oriented(encoded)
        h, w = bgr.shape[:2]
        s = self.settings
        rows, _ = multiscale_detect(self.detector, bgr, s.detection_sides, s.nms_threshold)
        rows.sort(key=lambda r: (*map(float, r[:4]), -float(r[14])))
        faces, vectors = [], {}
        for number, row in enumerate(rows, 1):
            fid = f'{photo_id}_f{number:03d}'
            face = dict(face_id=fid, photo_id=photo_id, x=float(row[0]), y=float(row[1]),
                        width=float(row[2]), height=float(row[3]), detection_score=float(row[14]),
                        landmarks=row[4:14].tolist(), blur_variance=None, quality_reason='')
            try:
                crop, vector = extract(self.recognizer, bgr, row)
                vectors[fid] = vector
                blur = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                face['blur_variance'] = blur
                reasons = []
                if min(row[2:4]) < s.min_face_size:
                    reasons.append('small_face')
                if blur < s.min_blur:
                    reasons.append('blurry_face')
                if (row[0] < -.1*row[2] or row[1] < -.1*row[3]
                        or row[0]+row[2] > w+.1*row[2] or row[1]+row[3] > h+.1*row[3]):
                    reasons.append('clipped_face')
                face['quality_reason'] = ';'.join(reasons)
            except (cv2.error, ValueError) as error:
                face['quality_reason'] = f'feature_failed:{type(error).__name__}:{error}'
            faces.append(face)
        return faces, vectors

    def initial_groups(self, faces, vectors):
        """Mutates faces with provisional groups. Not permanent identity assignment.

        Pairwise hierarchy uses quadratic memory; caller must bound batch sizes.
        """
        s = self.settings
        return group_faces(faces, vectors, threshold=s.similarity_threshold,
                           min_group=s.min_group_size, margin=s.ambiguity_margin, method=s.linkage)
