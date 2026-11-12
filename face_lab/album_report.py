"""Build the offline photo-collection viewer."""
import json
from pathlib import Path


def render(output, results, summary, vectors, counts, collections=None):
    if collections is None:
        raise ValueError('Photo collections are required; run python -m face_lab albums --input PHOTO_FOLDER')
    # Send display fields only, not vectors or full diagnostic matrices.
    cases = {}
    for case, groups in collections.items():
        cases[case] = []
        for group in groups:
            item = {k: group[k] for k in ('id', 'title', 'identities')}
            item['photos'] = []
            for photo in group['photos']:
                record = {k: photo[k] for k in ('photo_id', 'path', 'count', 'state', 'reason')}
                record['faces'] = [dict(face_id=f['face_id'],
                                       box=[f[k] for k in ('x','y','width','height')],
                                       identity='인물 판정 안 함' if group['id'].startswith('crowd-') else f.get('group_id') or '미확인',
                                       reason='' if group['id'].startswith('crowd-') else f.get('reason', ''),
                                       own=None if group['id'].startswith('crowd-') else f.get('diagnostic', {}).get('own_group_score'))
                                   for f in photo['faces']]
                item['photos'].append(record)
            cases[case].append(item)
    payload = json.dumps(dict(summary=summary, cases=cases), ensure_ascii=False, allow_nan=False)
    payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
    detection_score = str(summary.get('detection', {}).get('score_threshold', .75))
    notice = f'분석 실패 {summary.get("failed_photos", 0)}장 · 정상 분석 후 얼굴 미탐지 {summary.get("zero_face_photos", 0)}장'
    (output/'index.html').write_text(template.replace('__REPORT_DATA__', payload)
                                   .replace('__DETECTION_SCORE__', detection_score)
                                   .replace('__RUN_NOTICE__', notice), encoding='utf-8')
