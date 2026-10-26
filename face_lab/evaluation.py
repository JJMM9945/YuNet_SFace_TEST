import csv
import json
from collections import Counter
from pathlib import Path
from .pipeline import write_json


def choose2(n):
    return n * (n - 1) // 2


def evaluate(args):
    run = Path(args.run)
    summary_path=run/'summary.json'
    if summary_path.exists() and json.loads(summary_path.read_text(encoding='utf-8')).get('mode') in {'apply-review','correct-groups'}:
        raise ValueError('This run has human-corrected labels. Evaluate the original automatic run to measure model/grouping performance.')
    with (run / "faces.csv").open(encoding="utf-8-sig", newline="") as file:
        faces = {f["face_id"]: f for f in csv.DictReader(file)}
    with Path(args.ground_truth).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"face_id", "manual_person_id", "include"} <= set(reader.fieldnames or []):
            raise ValueError("Ground truth needs face_id, manual_person_id, include columns.")
        rows = list(reader)
    if rows and '_context_sha256' in rows[0]:
        from .review import context_hash
        cache=json.loads((run/'cache.json').read_text(encoding='utf-8'))
        if any(r.get('_context_sha256')!=context_hash(cache) for r in rows):
            raise ValueError('Ground-truth extraction mismatch; evaluate the exact source extraction.')
    seen, selected = set(), []
    for r in rows:
        fid = r["face_id"]
        if fid not in faces or fid in seen:
            raise ValueError(f"Unknown or duplicate ground-truth face_id: {fid}")
        seen.add(fid)
        if r["include"].strip() not in {"0", "1"}:
            raise ValueError(f"include must be 0 or 1: {fid}")
        manual = r["manual_person_id"].strip()
        if manual and r["include"].strip() == "1":
            # Unknown predictions are distinct singleton identities, not one shared unknown person.
            selected.append((fid, manual, faces[fid]["group_id"] or "unknown:"+fid))
    if not selected:
        raise ValueError("No filled, included ground truth; metrics are not computed.")
    truth = Counter(t for _, t, _ in selected)
    predicted = Counter(p for _, _, p in selected)
    joint = Counter((t, p) for _, t, p in selected)
    tp = sum(choose2(n) for n in joint.values())
    predicted_pairs, true_pairs = sum(map(choose2, predicted.values())), sum(map(choose2, truth.values()))
    fp, fn = predicted_pairs-tp, true_pairs-tp
    precision = tp/predicted_pairs if predicted_pairs else None
    recall = tp/true_pairs if true_pairs else None
    denom = 2*tp+fp+fn
    f1 = 2*tp/denom if denom else None
    result = dict(evaluated_faces=len(selected), total_detected_faces=len(faces), omitted_faces=len(faces)-len(selected),
                  evaluated_unknown_faces=sum(p.startswith("unknown:") for _, _, p in selected),
                  true_positive_pairs=tp, false_positive_pairs=fp, false_negative_pairs=fn,
                  precision=precision, recall=recall, f1=f1,
                  unknown_policy="included as separate singleton predictions; same-person unknown pairs count as false negatives",
                  undefined_policy="zero denominator is null", limitation="Missed detections cannot be evaluated from this CSV.")
    output = Path(args.output) if args.output else run / "evaluations"
    from datetime import datetime, timezone
    output.mkdir(parents=True, exist_ok=True)
    target = output / ("evaluation_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+".json")
    result["ground_truth_sha256"] = __import__("hashlib").sha256(Path(args.ground_truth).read_bytes()).hexdigest()
    write_json(target, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Evaluation: {target}")
