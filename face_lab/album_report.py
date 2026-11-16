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
    if summary.get('detection', {}).get('conditional_detection'):
        notice += f' · 조건부 추가 검사 {summary.get("conditional_rechecked_photos", 0)}장 · 추가 검출 {summary.get("conditional_added_faces", 0)}개'
    detection = summary.get('detection', {})
    sides = '·'.join(str(side) for side in detection.get('detection_sides', [1280]))
    policy = f'기본 검출 크기 {sides}px'
    if detection.get('conditional_detection'):
        extra = '·'.join(str(side) for side in detection.get('supplemental_detection_sides', [960,1600,2400]))
        policy += f' → 얼굴0개 또는{detection.get("conditional_min_faces", 3)}개 이상이면 {extra}px 추가 검사. 기존 상자 유지·중복 제거 후 특징 추출'
    (output/'index.html').write_text(template.replace('__REPORT_DATA__', payload)
                                   .replace('__DETECTION_SCORE__', detection_score)
                                   .replace('__DETECTION_POLICY__', policy)
                                   .replace('__RUN_NOTICE__', notice), encoding='utf-8')
