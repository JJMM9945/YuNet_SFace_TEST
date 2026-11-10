# 역할: CLI 명령과 현재 기본 설정 검증 및 분석·검토 명령 분기
import argparse
import sys


def parser():
    p = argparse.ArgumentParser(description="Offline YuNet + SFace personal photo validation (CPU).")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-models", help="Explicit network download and SHA-256 validation only.")
    prep.add_argument("--models", default="models")
    aura_prep=sub.add_parser('prepare-auraface',help='Explicit download and checksum verification of official AuraFace weights')
    aura_prep.add_argument('--models',default='models/auraface')
    aura_test=sub.add_parser('compare-auraface',help='Paired local SFace/AuraFace experiment with frozen YuNet detections')
    aura_test.add_argument('--input',default='ljm_dataset')
    aura_test.add_argument('--source',default='results/최초 그룹화 유사도 비교/유사도 0.40')
    aura_test.add_argument('--ground-truth',default='results/얼굴크기별_오류진단/ground_truth_all_validated.csv')
    aura_test.add_argument('--output',default='results/AuraFace_비교')
    aura_test.add_argument('--models',default='models')
    aura_test.add_argument('--aura-models',default='models/auraface')
    check = sub.add_parser("check", help="Dependency, model loading and blank-image forward-pass check.")
    check.add_argument("--models", default="models")
    for name in ("analyze", "regroup"):
        s = sub.add_parser(name, help="Local analysis" if name == "analyze" else "Regroup existing features after strict cache validation")
        s.add_argument("--input", default="ljm_dataset")
        s.add_argument("--output", default="results")
        s.add_argument("--models", default="models")
        s.add_argument("--similarity-threshold", type=float, default=0.40, help="Cosine similarity >= this; experimental starting value")
        s.add_argument("--min-group-size", type=int, default=1)
        s.add_argument("--ambiguity-margin", type=float, default=0.01)
        s.add_argument('--linkage', choices=['complete','average'], default='average')
        s.set_defaults(reassign_unknown=False)
        if name == "analyze":
            s.add_argument("--score-threshold", type=float, default=0.75)
            s.add_argument("--nms-threshold", type=float, default=0.3)
            s.add_argument("--top-k", type=int, default=5000)
            s.add_argument("--max-side", type=int, default=1600)
            s.add_argument('--detection-sides', type=int, nargs='+', default=[960,1600,2400], help='Selected default: 960 1600 2400; use --detection-sides 1600 for a single scale; global original-coordinate NMS')
            s.add_argument("--min-face-size", type=float, default=40, help="Oriented original pixels")
            s.add_argument("--min-blur", type=float, default=50, help="Laplacian variance on aligned 112x112 crop")
        else:
            s.add_argument("--source", required=True, help="Previous run directory containing cache.json and features.npz")
    evaluation = sub.add_parser("evaluate", help="Pairwise precision/recall/F1 only with manual ground truth")
    evaluation.add_argument("--run", required=True)
    evaluation.add_argument("--ground-truth", required=True)
    evaluation.add_argument("--output", help="Evaluation JSON directory")
    review = sub.add_parser('review',help='Create offline ground-truth annotation HTML next to a run')
    review.add_argument('--run',required=True)
    corrections = sub.add_parser('correct-groups', help='Apply explicit face-ID corrections to a new local result')
    corrections.add_argument('--source', required=True)
    corrections.add_argument('--corrections', required=True)
    corrections.add_argument('--output', default='results/manual_corrections')
    template = sub.add_parser('corrections-template', help='Create a source-bound empty face-ID correction file')
    template.add_argument('--source', required=True)
    template.add_argument('--output', required=True)
    experiment = sub.add_parser('compare-representatives', help='Compare known false-detection exclusion and frozen representative reassignment')
    experiment.add_argument('--source', required=True)
    experiment.add_argument('--corrections', required=True)
    experiment.add_argument('--ground-truth', required=True)
    experiment.add_argument('--input', default='ljm_dataset')
    experiment.add_argument('--models', default='models')
    experiment.add_argument('--output', default='results/representative_comparison')
    experiment.add_argument('--representatives', type=int, default=3, help='Maximum fixed representatives per group')
    experiment.add_argument('--representative-score-threshold', type=float, default=.45, help='Minimum mean cosine similarity for the representative experiments')
    experiment.add_argument('--compare-with', help='Previous compatible representative-comparison directory')
    singleton = sub.add_parser('singleton-review', help='Create an offline, pair-by-pair singleton review from an existing comparison')
    singleton.add_argument('--comparison', required=True)
    singleton.add_argument('--source', help='Original automatic run, if it has moved')
    singleton.add_argument('--ground-truth', help='Optional source-bound partial truth to display previous confirmations')
    singleton.add_argument('--output', default='results/singleton_review')
    singleton.add_argument('--top-candidates', type=int, default=3)
    quality = sub.add_parser('diagnose-quality', help='Diagnose grouping errors by original-coordinate face size')
    quality.add_argument('--run', required=True)
    quality.add_argument('--ground-truth', required=True)
    quality.add_argument('--review-decisions')
    quality.add_argument('--review-manifest')
    quality.add_argument('--size-boundaries', type=float, nargs='+', default=[60, 100, 200])
    quality.add_argument('--output', required=True)
    quality_review = sub.add_parser('quality-review', help='Create a source-bound ground-truth page filtered by face size')
    quality_review.add_argument('--run', required=True)
    quality_review.add_argument('--ground-truth', required=True)
    quality_review.add_argument('--min-face-size', type=float, default=0)
    quality_review.add_argument('--max-face-size', type=float)
    quality_review.add_argument('--only-unlabeled', action='store_true')
    quality_review.add_argument('--output', required=True)
    normalize_truth = sub.add_parser('normalize-quality-truth', help='Normalize IDs exported by the filtered quality review')
    normalize_truth.add_argument('--run', required=True)
    normalize_truth.add_argument('--input', required=True)
    normalize_truth.add_argument('--output', required=True)
    merge_split = sub.add_parser('diagnose-merge-split', help='Compare conservative group merge and split-review rules')
    merge_split.add_argument('--run', required=True)
    merge_split.add_argument('--ground-truth', required=True)
    merge_split.add_argument('--output', required=True)
    identity_test = sub.add_parser('test-identity-links', help='Test explicit identity collections and seed-only predictions')
    identity_test.add_argument('--run', required=True)
    identity_test.add_argument('--ground-truth', required=True)
    identity_test.add_argument('--output', default='results/identity_links')
    identity_edit = sub.add_parser('edit-identity-links', help='Apply explicit face-ID link changes to a new store revision')
    identity_edit.add_argument('--store', required=True)
    identity_edit.add_argument('--cache', required=True)
    identity_edit.add_argument('--changes')
    identity_edit.add_argument('--undo', action='store_true')
    identity_edit.add_argument('--output', required=True)
    identity_apply = sub.add_parser('apply-identity-dataset', help='Analyze a new dataset with frozen confirmed identity references')
    identity_apply.add_argument('--reference-run', required=True)
    identity_apply.add_argument('--identity-store', required=True)
    identity_apply.add_argument('--input', required=True)
    identity_apply.add_argument('--models', default='models')
    identity_apply.add_argument('--output', required=True)
    automatic = sub.add_parser('compare-automatic-anchors', help='Compare frozen automatic anchors without confirmed person identities')
    automatic.add_argument('--source', required=True)
    automatic.add_argument('--ground-truth', required=True)
    automatic.add_argument('--target-source')
    automatic.add_argument('--output', required=True)
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
    if getattr(a, 'detection_sides', None) and any(side < 32 for side in a.detection_sides):
        p.error('detection-sides must each be >=32')
    if hasattr(a, 'representatives') and a.representatives < 2:
        p.error('representatives must be at least 2')
    if hasattr(a, 'top_candidates') and not 1 <= a.top_candidates <= 10:
        p.error('top-candidates must be between 1 and 10')
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
        elif a.command == 'prepare-auraface':
            from .auraface import prepare_auraface
            prepare_auraface(a.models)
        elif a.command == 'compare-auraface':
            from .auraface_experiment import run_experiment
            run_experiment(a)
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
        elif a.command == 'correct-groups':
            from .corrections import correct_groups
            correct_groups(a)
        elif a.command == 'corrections-template':
            from .corrections import create_template
            create_template(a)
        elif a.command == 'compare-representatives':
            from .representative_experiment import compare_representatives
            compare_representatives(a)
        elif a.command == 'singleton-review':
            from .singleton_review import create_singleton_review
            create_singleton_review(a)
        elif a.command == 'diagnose-quality':
            from .quality_diagnosis import diagnose_quality
            diagnose_quality(a)
        elif a.command == 'quality-review':
            from .quality_diagnosis import make_quality_review
            make_quality_review(a)
        elif a.command == 'normalize-quality-truth':
            from .quality_diagnosis import normalize_quality_truth
            normalize_quality_truth(a)
        elif a.command == 'diagnose-merge-split':
            from .merge_split_diagnosis import diagnose_merge_split
            diagnose_merge_split(a)
        elif a.command == 'test-identity-links':
            from .identity_experiment import test_identity_links
            test_identity_links(a)
        elif a.command == 'edit-identity-links':
            from .identity_experiment import edit_identity_links
            edit_identity_links(a)
        elif a.command == 'apply-identity-dataset':
            from .identity_dataset import apply_identity_dataset
            apply_identity_dataset(a)
        elif a.command == 'compare-automatic-anchors':
            from .automatic_representatives import run_auto_comparison
            run_auto_comparison(a)
        else:
            from .pipeline import analyze, regroup
            (analyze if a.command == "analyze" else regroup)(a)
        return 0
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}\nUse Python 3.11/3.12 and install requirements.txt in a clean venv; prepare-models is the only network command.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
