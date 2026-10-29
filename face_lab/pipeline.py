import csv
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
import uuid

import cv2
import numpy as np
import PIL
import scipy

from . import VERSION, PREPROCESS_VERSION, GROUPING_VERSION
from .models import MODELS, load, sha256, verify
from .preprocess import decode_oriented, detection_input, to_original, extract, save_bgr
from .grouping import group_faces, reassign_unknown, validate_vectors

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Excel formula-injection protection for user-controlled path/text fields.
            writer.writerow({k: ("'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t", "\r")) else v)
                             for k, v in row.items()})


def inventory(root):
    if not root.is_dir():
        raise ValueError(f"Input directory not found: {root}")
    paths = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS),
                   key=lambda p: p.relative_to(root).as_posix())
    result = []
    for p in paths:
        rel = p.relative_to(root).as_posix()
        try:
            digest, error = sha256(p), ""
        except OSError as e:
            digest, error = "", str(e)
        result.append(dict(photo_id="p" + hashlib.sha256(rel.encode("utf-8")).hexdigest()[:16],
                           relative_path=rel, sha256=digest, read_error=error))
    return result


def runtime():
    return dict(project=VERSION, preprocessing=PREPROCESS_VERSION, opencv=cv2.__version__, numpy=np.__version__,
                pillow=PIL.__version__, scipy=scipy.__version__, python=platform.python_version(),
                platform=platform.platform(), threads=1, rng_seed=0, backend="OpenCV", target="CPU",
                model_loading="Python bytes -> OpenCV ONNX buffers")


def new_run(output, input_root):
    output = Path(output).resolve()
    input_root = Path(input_root).resolve()
    if output == input_root or input_root in output.parents:
        raise ValueError("Output must be outside the input directory (recursive input contamination).")
    output.mkdir(parents=True, exist_ok=True)
    run = output / (datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:6])
    run.mkdir()
    for name in ("photos", "annotated", "crops", "montages"):
        (run / name).mkdir()
    return run


def extraction_signature(config):
    keys = ("score_threshold", "nms_threshold", "top_k", "max_side", "min_face_size", "min_blur")
    return dict(runtime=runtime(), models=MODELS, extraction={k: config[k] for k in keys})


def analyze(args):
    start = time.perf_counter()
    root = Path(args.input).resolve()
    files = inventory(root)
    config = vars(args).copy()
    config.pop("handler", None)
    detector, recognizer = load(args.models, args.score_threshold, args.nms_threshold, args.top_k)
    run = new_run(args.output, root)
    photos, faces, vectors = [], [], {}
    detection_seconds = feature_seconds = 0.0
    for item in files:
        photo = dict(item, status="success", error="", detected_faces=0, people="")
        photos.append(photo)
        try:
            if item["read_error"]:
                raise OSError(item["read_error"])
            data = (root / item["relative_path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("input_changed_during_run")
            bgr = decode_oriented(data)
            h, w = bgr.shape[:2]
            photo.update(width=w, height=h)
            save_bgr(run / "photos" / (item["photo_id"] + ".jpg"), bgr)
            small, sx, sy = detection_input(bgr, args.max_side)
            photo.update(detect_width=small.shape[1], detect_height=small.shape[0], scale_x=sx, scale_y=sy)
            t = time.perf_counter()
            try:
                detector.setInputSize((small.shape[1], small.shape[0]))
                _, detected = detector.detect(small)
            finally:
                detection_seconds += time.perf_counter() - t
            rows = [] if detected is None else [to_original(r, sx, sy) for r in detected]
            rows.sort(key=lambda r: (float(r[0]), float(r[1]), float(r[2]), float(r[3]), -float(r[14])))
            photo["detected_faces"] = len(rows)
            for number, row in enumerate(rows, 1):
                fid = f"{item['photo_id']}_f{number:03d}"
                f = dict(face_id=fid, photo_id=item["photo_id"], relative_path=item["relative_path"],
                         x=float(row[0]), y=float(row[1]), width=float(row[2]), height=float(row[3]),
                         detection_score=float(row[14]), landmarks=row[4:14].tolist(), quality_reason="",
                         blur_variance=None, crop_path="")
                faces.append(f)
                t = time.perf_counter()
                try:
                    if not np.isfinite(row).all() or min(row[2:4]) <= 0:
                        raise ValueError("invalid_detection_geometry")
                    crop, vector = extract(recognizer, bgr, row)
                    vectors[fid] = vector
                    f["crop_path"] = f"crops/{fid}.jpg"
                    save_bgr(run / f["crop_path"], crop)
                    blur = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                    f["blur_variance"] = blur
                    reasons = []
                    if min(row[2:4]) < args.min_face_size:
                        reasons.append("small_face")
                    if blur < args.min_blur:
                        reasons.append("blurry_face")
                    # Large clipping makes alignment unreliable; tolerate minor detector overshoot.
                    if row[0] < -0.1*row[2] or row[1] < -0.1*row[3] or row[0]+row[2] > w+0.1*row[2] or row[1]+row[3] > h+0.1*row[3]:
                        reasons.append("clipped_face")
                    f["quality_reason"] = ";".join(reasons)
                except Exception as e:
                    f["quality_reason"] = f"feature_failed:{type(e).__name__}:{e}"
                finally:
                    feature_seconds += time.perf_counter() - t
        except Exception as e:
            photo.update(status="failed", error=f"{type(e).__name__}: {e}")
    ids = sorted(vectors)
    np.savez_compressed(run / "features.npz", face_ids=np.array(ids, dtype=str),
                        vectors=np.stack([vectors[i] for i in ids]) if ids else np.empty((0, 128), np.float32))
    cache = dict(schema=1, signature=extraction_signature(config), input_root=str(root), input_files=files,
                 config=config, photos=photos, faces=faces, features_sha256=sha256(run / "features.npz"))
    write_json(run / "cache.json", cache)
    return finish(run, photos, faces, vectors, args, start, detection_seconds, feature_seconds, cache, "analyze")


def load_vectors(source, cache):
    if sha256(source / "features.npz") != cache['features_sha256']:
        raise ValueError('Feature cache checksum mismatch')
    with np.load(source / "features.npz", allow_pickle=False) as data:
        ids, matrix = data['face_ids'].tolist(), data['vectors']
        if matrix.ndim != 2 or matrix.shape != (len(ids), 128) or len(set(ids)) != len(ids):
            raise ValueError('Invalid feature cache mapping/shape')
        face_ids = {f['face_id'] for f in cache['faces']}
        if not set(ids) <= face_ids or len(face_ids) != len(cache['faces']):
            raise ValueError('Feature cache has unknown/duplicate face IDs')
        vectors = dict(zip(ids, matrix))
        validate_vectors(vectors, dimension=128)
        if any(not f.get('quality_reason') and f['face_id'] not in vectors for f in cache['faces']):
            raise ValueError('Eligible face missing embedding')
    return vectors


def regroup(args):
    start = time.perf_counter()
    source, root = Path(args.source).resolve(), Path(args.input).resolve()
    cache = json.loads((source / "cache.json").read_text(encoding="utf-8"))
    if cache.get("schema") != 1 or cache["signature"] != extraction_signature(cache["config"]):
        raise ValueError("Cache incompatible: model/preprocessing/runtime version changed; run analyze again.")
    for spec in MODELS.values():
        verify(Path(args.models) / spec["name"], spec)
    if inventory(root) != cache["input_files"]:
        raise ValueError("Cache input mismatch: photo content/path/list changed; run analyze again.")
    if sha256(source / "features.npz") != cache["features_sha256"]:
        raise ValueError("Feature cache checksum mismatch.")
    vectors = load_vectors(source, cache)
    run = new_run(args.output, root)
    for name in ("photos", "crops"):
        shutil.copytree(source / name, run / name, dirs_exist_ok=True)
    shutil.copy2(source / "features.npz", run / "features.npz")
    write_json(run / "cache.json", cache)
    return finish(run, cache["photos"], cache["faces"], vectors, args, start, 0.0, 0.0, cache, "regroup")


def finish(run, photos, faces, vectors, args, start, detection_seconds, feature_seconds, cache, mode, assignments=None):
    from .reports import render
    t = time.perf_counter()
    diagnostics, reassignment = {}, None
    if mode == 'apply-review':
        from .review import assign_review
        assign_review(faces, assignments)
    elif mode != 'correct-groups':
        diagnostics = group_faces(faces, vectors, args.similarity_threshold, args.min_group_size, args.ambiguity_margin,
                                  method=getattr(args, 'linkage', 'complete'))
        if getattr(args, 'reassign_unknown', False):
            reassignment = reassign_unknown(faces, vectors, args.reassign_threshold, args.reassign_margin,
                                            args.representatives, args.reassign_min_support,
                                            args.reassign_support_threshold, args.diversity_weight)
        write_json(run / 'diagnostics.json', diagnostics)
        flat = []
        for f in faces:
            d = diagnostics[f['face_id']]
            row = {k:v for k,v in d.items() if not isinstance(v, (dict,list))}
            row['stages'] = json.dumps(d['stages'], ensure_ascii=False)
            row['candidates'] = json.dumps(d['candidates'], ensure_ascii=False)
            row['reassignment'] = json.dumps(d.get('reassignment'), ensure_ascii=False)
            flat.append(row)
        fields = sorted({k for row in flat for k in row}) or ['face_id']
        write_csv(run / 'diagnostics.csv', flat, fields)
        if reassignment:
            write_json(run / 'representatives.json', reassignment)
    group_seconds = time.perf_counter() - t
    for photo in photos:
        own = [f for f in faces if f["photo_id"] == photo["photo_id"]]
        photo["people"] = ";".join(sorted({f["label"] for f in own}))
        photo["unknown_faces"] = sum(f["status"] == "unknown" for f in own)
        photo["review"] = ";".join(sorted({f["review"] for f in own if f["review"]}))
    write_csv(run / "faces.csv", faces, ["photo_id", "relative_path", "face_id", "x", "y", "width", "height",
              "landmarks", "detection_score", "group_id", "label", "status", "reason", "blur_variance", "review", "crop_path",
              "manual_person_id", "label_origin", "quality_reason", "decision_state"])
    write_csv(run / "photos.csv", photos, ["photo_id", "relative_path", "sha256", "status", "error", "detected_faces",
              "people", "unknown_faces", "review", "width", "height", "detect_width", "detect_height", "scale_x", "scale_y"])
    from .review import context_hash
    write_csv(run / "ground_truth_template.csv", [dict(f, manual_person_id=f.get('manual_person_id',''), include="1", review_type="", annotation_status="", notes="",
              _context_sha256=context_hash(cache)) for f in faces],
              ["face_id", "photo_id", "relative_path", "manual_person_id", "include", "annotation_status", "review_type", "notes", "_context_sha256"])
    write_csv(run / "photo_review_template.csv", [dict(p, missed_face_count="", false_detection_count="", notes="") for p in photos],
              ["photo_id", "relative_path", "status", "detected_faces", "missed_face_count", "false_detection_count", "notes"])
    render(run, photos, faces)
    config = vars(args).copy()
    config.pop("handler", None)
    from collections import Counter
    sizes = Counter(f['group_id'] for f in faces if f['group_id'])
    summary = dict(mode=mode, grouping_eligible_faces=sum(not f.get('quality_reason') and f['face_id'] in vectors for f in faces),
                   singleton_groups=sum(n == 1 for n in sizes.values()),
                   quality_excluded_faces=sum(bool(f.get('quality_reason')) for f in faces),
                   held_faces=sum(f['status'] == 'unknown' and not f.get('quality_reason') for f in faces),
                   reassigned_faces=sum(d.get('reassignment', {}).get('applied', False) for d in diagnostics.values()),
                   unknown_reasons=dict(Counter(f['reason'] for f in faces if f['status'] == 'unknown')),
                   same_photo_review_groups=sorted({f['group_id'] for f in faces if 'same_photo_same_group' in f['review']}),
                   automatic_assignment_rate=sum(bool(f['group_id']) for f in faces)/len(faces) if faces else None,
                   actual_grouping_config={k:getattr(args,k,None) for k in ('linkage','similarity_threshold','min_group_size','ambiguity_margin','reassign_unknown')}, total_photos=len(photos), successful_photos=sum(p["status"] == "success" for p in photos),
                   failed_photos=sum(p["status"] == "failed" for p in photos),
                   zero_face_photos=sum(p["status"] == "success" and p["detected_faces"] == 0 for p in photos),
                   detected_faces=len(faces), groups=len({f["group_id"] for f in faces if f["group_id"]}),
                   unknown_faces=sum(f["status"] == "unknown" for f in faces),
                   review_faces=sum(bool(f["review"]) for f in faces), detection_seconds=detection_seconds,
                   feature_seconds=feature_seconds, grouping_seconds=group_seconds, total_seconds=time.perf_counter()-start,
                   note=("Human corrections applied; not automatic accuracy or retraining." if mode in {'apply-review','correct-groups'}
                         else "Counts are not accuracy. No ground truth evaluation performed."))
    write_json(run / "summary.json", summary)
    write_json(run / "run_config.json", dict(options=config, extraction_signature=cache["signature"], runtime=runtime(),
               grouping_version=('human-correction-v1' if mode in {'apply-review','correct-groups'} else GROUPING_VERSION),
               ordering='face_id ascending; SciPy ties resolved on canonical condensed matrix; library version pinned',
               input_files=cache["input_files"], input_root=str(Path(args.input).resolve()),
               preprocessing=dict(channel_order="uint8 BGR, no manual normalization before OpenCV APIs", exif="Pillow exif_transpose once",
                resize="aspect preserved; round dimensions; INTER_AREA; no upscaling/padding", coordinates="oriented original pixels, x/y/w/h and five x/y landmarks",
                sface="OpenCV alignCrop (112x112) -> feature (128) -> L2 normalize", similarity="dot(unit vectors); larger is closer; distance=1-cosine")))
    (run / "summary.md").write_text("# Local face validation\n\n" + "\n".join(f"- {k}: {v}" for k, v in summary.items()) +
        "\n\n검출 0개는 보이는 얼굴 없음/미검출을 자동 구별하지 못합니다. unknown은 검출 후 그룹 미확인입니다.\n"
        "얼굴별 오류는 faces.csv, 사진별 오류는 photos.csv를 확인하세요. 개인 사진 정확도는 수동 정답 평가 전 알 수 없습니다.\n", encoding="utf-8")
    print(f"Result: {run}\nPhotos={len(photos)}, faces={len(faces)}, groups={summary['groups']}, unknown={summary['unknown_faces']}")
    return run
