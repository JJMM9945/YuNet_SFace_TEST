"""Dataset-independent fixed-.75 photo album workflow for PC/device parity."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np
import cv2

from . import VERSION, PREPROCESS_VERSION, GROUPING_VERSION
from .albums import build_collections, collection_summary
from .album_report import render
from .engine import FaceEngine, Settings
from .detection import DETECTION_POLICY_VERSION
from .models import MODELS
from .preprocess import decode_oriented, save_bgr

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def group_photos(photos, faces, vectors, engine):
    """Device entry point after extraction; no file I/O or test-package dependency.

    photos: photo_id, relative_path (+ optional status/error).
    faces/vectors: combined outputs of FaceEngine.extract_photo for these photos.
    Groups are provisional and specific to this batch, not permanent person IDs.
    """
    if len({p['photo_id'] for p in photos}) != len(photos):
        raise ValueError('Duplicate photo IDs')
    known = {p['photo_id'] for p in photos}
    if any(f['photo_id'] not in known for f in faces):
        raise ValueError('Face references an unknown photo')
    counts = Counter(f['photo_id'] for f in faces)
    results = {}
    for key in ('solo', 'pair', 'crowd'):
        subset = deepcopy([f for f in faces if (counts[f['photo_id']] == 1 if key == 'solo' else
                           counts[f['photo_id']] == 2 if key == 'pair' else counts[f['photo_id']] >= 3)])
        if key != 'crowd':
            engine.initial_groups(subset, {f['face_id']: vectors[f['face_id']] for f in subset if f['face_id'] in vectors})
        results[key] = subset
    collections = build_collections(results, photos, counts)
    return collections, {key: collection_summary(value) for key, value in collections.items()}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def run(input_folder, output_folder='results/photo_albums', models='models'):
    dataset, parent = Path(input_folder).expanduser().resolve(), Path(output_folder).expanduser().resolve()
    if not dataset.is_dir():
        raise ValueError(f'사진 폴더를 찾을 수 없습니다: {dataset}')
    if parent == dataset or dataset in parent.parents:
        raise ValueError('결과 폴더는 입력 사진 폴더 밖에 지정하세요.')
    paths = sorted((p for p in dataset.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS),
                   key=lambda p: p.relative_to(dataset).as_posix())
    if not paths:
        raise ValueError('지원하는 사진이 없습니다. JPG/PNG/BMP/WebP/TIFF 사진을 넣어 주세요. HEIC는 변환 후 사용하세요.')
    engine = FaceEngine(models, Settings())  # Fixed .75; no score/cache/source override.
    output = parent / (datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ_') + uuid.uuid4().hex[:6])
    (output/'photos').mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    photos, faces, vectors = [], [], {}
    analysis_seconds = 0.0
    for index, path in enumerate(paths, 1):
        relative = path.relative_to(dataset).as_posix()
        pid = 'p' + hashlib.sha256(relative.encode('utf-8')).hexdigest()[:20]
        photo = dict(photo_id=pid, relative_path=relative, status='success', error='')
        try:
            encoded = path.read_bytes()
            photo['sha256'] = hashlib.sha256(encoded).hexdigest()
            tick = time.perf_counter()
            # Decode for the original-photo report as well as the engine's own decode.
            oriented = decode_oriented(encoded)
            extracted, values = engine.extract_photo(encoded, pid)
            analysis_seconds += time.perf_counter() - tick
            if getattr(engine, 'last_detection', None) is not None:
                photo['detection'] = engine.last_detection
            for face in extracted:
                face['relative_path'] = relative
            save_bgr(output/'photos'/f'{pid}.jpg', oriented)
            faces.extend(extracted)
            vectors.update(values)
        except (OSError, ValueError, RuntimeError, cv2.error) as error:
            photo.update(status='failed', error=f'{type(error).__name__}: {error}')
        photos.append(photo)
        print(f'[{index}/{len(paths)}] {relative}: {photo["status"]}', flush=True)
    tick = time.perf_counter()
    collections, rows = group_photos(photos, faces, vectors, engine)
    grouping_seconds = time.perf_counter() - tick
    settings = asdict(engine.settings)
    summary = dict(total_photos=len(photos), successful_photos=sum(p['status'] == 'success' for p in photos),
                   failed_photos=sum(p['status'] == 'failed' for p in photos),
                   zero_face_photos=sum(p['status'] == 'success' and not any(f['photo_id'] == p['photo_id'] for f in faces) for p in photos),
                   detected_faces=len(faces), quality_excluded_faces=sum(bool(f.get('quality_reason')) for f in faces),
                   conditional_rechecked_photos=sum(p.get('detection', {}).get('conditional_triggered', False) for p in photos),
                   conditional_added_faces=sum(p.get('detection', {}).get('added_faces', 0) for p in photos),
                   detection={k: settings[k] for k in ('score_threshold','detection_sides','conditional_detection','supplemental_detection_sides','conditional_min_faces','nms_threshold','top_k')},
                   grouping={k: settings[k] for k in ('linkage','similarity_threshold','ambiguity_margin','min_group_size')},
                   settings=settings, cases=rows, mode='photo_albums_conditional_1280',
                   analysis_seconds=analysis_seconds, grouping_seconds=grouping_seconds,
                   versions=dict(project=VERSION,preprocessing=PREPROCESS_VERSION,grouping=GROUPING_VERSION,detection_policy=DETECTION_POLICY_VERSION),
                   models={key: dict(name=spec['name'],sha256=spec['sha256']) for key,spec in MODELS.items()},
                   total_seconds=time.perf_counter()-started,
                   note='Photo counts are detected face counts, not verified people. Groups are provisional; counts are not accuracy.')
    write_json(output/'summary.json', summary)
    write_json(output/'photos.json', photos)
    write_json(output/'faces.json', faces)
    write_json(output/'photo_collections.json', collections)
    ids = sorted(vectors)
    np.savez_compressed(output/'features.npz',face_ids=np.array(ids,dtype=str),
                        vectors=np.stack([vectors[k] for k in ids]) if ids else np.empty((0,128),np.float32))
    render(output, {}, summary, vectors, {}, collections)
    print('Summary:', json.dumps(rows, ensure_ascii=False))
    print(f'Result: {output / "index.html"}')
    return output
