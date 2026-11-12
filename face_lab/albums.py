"""Photo collections: solo identity, unordered pair identity, one crowd album."""
from collections import defaultdict


def build_collections(results, photos, counts):
    cases = {'solo': [], 'pair': [], 'crowd': [], 'zero': []}
    for case in ('solo', 'pair', 'crowd', 'zero'):
        faces_by_photo = defaultdict(list)
        for f in results.get(case, []):
            faces_by_photo[f['photo_id']].append(f)
        grouped = {}
        selected = [p for p in photos if (counts[p['photo_id']] == 1 if case == 'solo' else
                    counts[p['photo_id']] == 2 if case == 'pair' else
                    counts[p['photo_id']] >= 3 if case == 'crowd' else counts[p['photo_id']] == 0)]
        for photo in selected:
            faces = faces_by_photo[photo['photo_id']]
            state, reason = 'assigned', ''
            if case in ('solo', 'pair'):
                if len(faces) != (1 if case == 'solo' else 2):
                    raise ValueError('Photo partition does not match face count')
                if any(f.get('quality_reason') for f in faces):
                    state, reason = 'quality_excluded', '작거나 흐리거나 잘린 얼굴이 있어요'
                elif any(not f.get('group_id') for f in faces):
                    state, reason = 'held', '인물 판단이 애매해요'
                elif case == 'pair' and len({f['group_id'] for f in faces}) != 2:
                    state, reason = 'held', '두 얼굴이 같은 인물 그룹에 배정되어 확인이 필요해요'
                signature = tuple(sorted(f['group_id'] for f in faces)) if state == 'assigned' else ('pending',)
            else:
                signature = ('failed',) if photo.get('status') == 'failed' else (case,)
                if signature == ('failed',):
                    state, reason = 'failed', photo.get('error', '사진 분석에 실패했습니다.')
            if signature not in grouped:
                number = len([s for s in grouped if s != ('pending',)]) + 1
                gid = f'{case}-{number:03d}' if signature != ('pending',) else f'{case}-pending'
                title = ('단체 사진' if case == 'crowd' else '얼굴 미탐지 사진' if case == 'zero' else
                         f'{"인물" if case == "solo" else "두 사람 조합"} {number:02d}')
                if signature == ('pending',):
                    title = '확인 필요'
                elif signature == ('failed',):
                    title = '분석 실패 사진'
                grouped[signature] = dict(id=gid, title=title, identities=list(signature) if case in ('solo','pair') and state == 'assigned' else [], photos=[])
            grouped[signature]['photos'].append(dict(photo_id=photo['photo_id'], path=photo['relative_path'],
                                                       count=counts[photo['photo_id']], faces=faces,
                                                       state=state, reason=reason))
        cases[case] = list(grouped.values())
    cases['all'] = cases['solo'] + cases['pair'] + cases['crowd'] + cases['zero']
    assert sum(len(g['photos']) for g in cases['all']) == len(photos)
    return cases


def collection_summary(groups):
    photos = [p for g in groups for p in g['photos']]
    return dict(photos=len(photos), faces=sum(p['count'] for p in photos),
                groups=sum(not g['id'].endswith('pending') and not g['id'].startswith('zero-') for g in groups),
                singleton_groups=sum(len(g['photos']) == 1 and not g['id'].endswith('pending') and not g['id'].startswith('zero-') for g in groups),
                held_photos=sum(p['state'] == 'held' for p in photos),
                quality_check_photos=sum(p['state'] == 'quality_excluded' for p in photos))
