import argparse
import sys


def parser():
    p = argparse.ArgumentParser(description="Offline YuNet + SFace personal photo validation (CPU).")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-models", help="Explicit network download and SHA-256 validation only.")
    prep.add_argument("--models", default="models")
    check = sub.add_parser("check", help="Dependency, model loading and blank-image forward-pass check.")
    check.add_argument("--models", default="models")
    for name in ("analyze", "regroup"):
        s = sub.add_parser(name, help="Local analysis" if name == "analyze" else "Regroup existing features after strict cache validation")
        s.add_argument("--input", default="ljm_dataset")
        s.add_argument("--output", default="results")
        s.add_argument("--models", default="models")
        s.add_argument("--similarity-threshold", type=float, default=0.45, help="Cosine similarity >= this; experimental starting value")
        s.add_argument("--min-group-size", type=int, default=2)
        s.add_argument("--ambiguity-margin", type=float, default=0.03)
        if name == "analyze":
            s.add_argument("--score-threshold", type=float, default=0.9)
            s.add_argument("--nms-threshold", type=float, default=0.3)
            s.add_argument("--top-k", type=int, default=5000)
            s.add_argument("--max-side", type=int, default=1600)
            s.add_argument("--min-face-size", type=float, default=40, help="Oriented original pixels")
            s.add_argument("--min-blur", type=float, default=50, help="Laplacian variance on aligned 112x112 crop")
        else:
            s.add_argument("--source", required=True, help="Previous run directory containing cache.json and features.npz")
    evaluation = sub.add_parser("evaluate", help="Pairwise precision/recall/F1 only with manual ground truth")
    evaluation.add_argument("--run", required=True)
    evaluation.add_argument("--ground-truth", required=True)
    evaluation.add_argument("--output", help="Evaluation JSON directory")
    review = sub.add_parser('review',help='Create an offline manual merge/split review HTML next to a run')
    review.add_argument('--run',required=True)
    for name in ('apply-review','correct-groups'):
        s=sub.add_parser(name,help='Apply explicit human annotations or pair constraints to a new result')
        s.add_argument('--source',required=True)
        s.add_argument('--input',default='ljm_dataset')
        s.add_argument('--output',default='results')
        s.add_argument('--models',default='models')
        if name=='apply-review':
            s.add_argument('--ground-truth',required=True)
        else:
            s.add_argument('--corrections',required=True)
    return p


def main():
    p = parser()
    a = p.parse_args()
    for name in ("score_threshold", "nms_threshold"):
        if hasattr(a, name) and not 0 < getattr(a, name) <= 1:
            p.error(f"{name} must be in (0,1]")
    if hasattr(a, "similarity_threshold"):
        if not -1 <= a.similarity_threshold <= 1 or not 0 <= a.ambiguity_margin <= 2:
            p.error("similarity threshold must be [-1,1], margin [0,2]")
        if a.min_group_size < 1:
            p.error("min-group-size must be >= 1")
    for name in ("max_side", "top_k"):
        if hasattr(a, name) and getattr(a, name) < (32 if name == "max_side" else 1):
            p.error(f"{name} out of range")
    for name in ("min_face_size", "min_blur"):
        if hasattr(a, name) and not getattr(a, name) >= 0:
            p.error(f"{name} must be nonnegative")
    try:
        import cv2
        import numpy as np
        cv2.setNumThreads(1)
        cv2.setRNGSeed(0)
        cv2.ocl.setUseOpenCL(False)
        from .models import load, prepare
        if a.command == "prepare-models":
            prepare(a.models)
            print("Models verified and ready.")
        elif a.command == "check":
            d, r = load(a.models)
            d.setInputSize((320,320))
            d.detect(np.zeros((320,320,3), np.uint8))
            v = r.feature(np.zeros((112,112,3), np.uint8))
            if v.size != 128 or not np.isfinite(v).all():
                raise ValueError("SFace forward pass returned invalid features")
            print(f"OK OpenCV {cv2.__version__}; CPU model load and blank-image forward passes (not accuracy test)")
        elif a.command == "evaluate":
            from .evaluation import evaluate
            evaluate(a)
        elif a.command == 'review':
            from .review import make_review
            make_review(a)
        elif a.command == 'apply-review':
            from .review import apply_review
            apply_review(a)
        elif a.command == 'correct-groups':
            from .corrections import correct_groups
            correct_groups(a)
        else:
            from .pipeline import analyze, regroup
            (analyze if a.command == "analyze" else regroup)(a)
        return 0
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}\nUse Python 3.11/3.12 and install requirements.txt in a clean venv; prepare-models is the only network command.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
