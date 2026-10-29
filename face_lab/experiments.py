"""Offline, one-factor comparisons. Selection is explicit and restricted to dev."""
from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
import csv
import json
from pathlib import Path
import shutil
import time

from .evaluation import read_annotations, pair_metrics, read_split, split_hash
from .models import sha256
from .pipeline import new_run, load_vectors, finish, write_json, write_csv
from .review import validate_cache, context_hash

CONFIG_KEYS = ('linkage', 'similarity_threshold', 'min_group_size', 'ambiguity_margin', 'reassign_unknown',
               'reassign_threshold', 'reassign_margin', 'representatives', 'reassign_min_support',
               'reassign_support_threshold', 'diversity_weight')


def plan(args):
    base = {k: getattr(args, k) for k in CONFIG_KEYS}
    base['reassign_unknown'] = False
    runs = [('baseline_complete', dict(base, linkage='complete')),
            ('average_same_threshold', dict(base, linkage='average'))]
    for threshold in sorted(set(args.thresholds)):
        if threshold != args.similarity_threshold:
            runs.append((f'{args.linkage}_threshold_{threshold:g}', dict(base, similarity_threshold=threshold)))
    runs.append((f'{args.linkage}_reassignment', dict(base, reassign_unknown=True)))
    return runs


def memberships(faces):
    groups = defaultdict(list)
    for f in faces:
        if f['group_id']:
            groups[f['group_id']].append(f['face_id'])
    return {f['face_id']: tuple(sorted(groups[f['group_id']])) if f['group_id'] else None for f in faces}


def compare(args):
    source, root = Path(args.source).resolve(), Path(args.input).resolve()
    cache = validate_cache(source, root, args.models)
    vectors = load_vectors(source, cache)
    subset = args.subset
    splits = read_split(args.split_manifest, cache['photos']) if args.split_manifest else None
    if subset != 'all' and splits is None:
        raise ValueError('--subset dev/eval requires --split-manifest (photo-level disjoint sets)')
    if subset == 'all' and splits is not None:
        raise ValueError('With a split manifest choose --subset dev or eval; do not mix the sets')
    selection = None
    if subset == 'eval':
        if not args.selected_config:
            raise ValueError('Evaluation requires --selected-config frozen from a dev comparison; no eval sweep allowed')
        selection = json.loads(Path(args.selected_config).read_text())
        if (selection.get('schema') != 1 or selection.get('selected_on') != 'dev'
                or selection.get('split_sha256') != split_hash(splits)
                or selection.get('context_sha256') != context_hash(cache)
                or selection.get('features_sha256') != cache['features_sha256']):
            raise ValueError('Selected dev config does not match extraction or split manifest')
        configs = [('frozen_dev_selection', selection['config'])]
    else:
        if args.selected_config:
            raise ValueError('--selected-config is for eval only')
        configs = plan(args)
    photos = [p for p in cache['photos'] if splits is None or splits[p['photo_id']] == subset]
    pids = {p['photo_id'] for p in photos}
    faces = [f for f in cache['faces'] if f['photo_id'] in pids]
    if not photos or not faces:
        raise ValueError('No photos/faces in requested subset')
    annotations = read_annotations(args.ground_truth, cache['faces'], cache) if args.ground_truth else None
    parent = new_run(args.output, root)
    # Copies of existing artifacts only. Children share these new copies via local symlinks.
    for name in ('photos','crops'):
        shutil.copytree(source/name, parent/name, dirs_exist_ok=True)
    shutil.copy2(source/'features.npz', parent/'features.npz')
    write_json(parent/'cache.json', cache)
    if args.ground_truth:
        shutil.copy2(args.ground_truth, parent/'ground_truth_input.csv')
    if args.split_manifest:
        shutil.copy2(args.split_manifest, parent/'split_manifest.csv')
    if args.selected_config:
        shutil.copy2(args.selected_config, parent/'selected_config_input.json')
    write_csv(parent/'split_template.csv', [dict(p, split='') for p in cache['photos']], ['photo_id','relative_path','split'])
    write_csv(parent/'ground_truth_template.csv', [dict(f, manual_person_id='', include='1', annotation_status='', notes='',
              _context_sha256=context_hash(cache)) for f in cache['faces']],
              ['face_id','photo_id','relative_path','manual_person_id','include','annotation_status','notes','_context_sha256'])
    rows, results, changes = [], {}, []
    reference = reference_memberships = None
    for name, config in configs:
        started = time.perf_counter()
        child = parent/name
        child.mkdir()
        for folder in ('annotated','montages'):
            (child/folder).mkdir()
        for shared in ('photos','crops'):
            (child/shared).symlink_to(Path('..')/shared, target_is_directory=True)
        shutil.copy2(source/'features.npz', child/'features.npz')
        write_json(child/'cache.json', cache)
        metadata = dict(subset=subset, split_sha256=split_hash(splits) if splits else None,
                        face_ids=sorted(f['face_id'] for f in faces),
                        selected_config_sha256=sha256(Path(args.selected_config)) if selection else None)
        write_json(child/'experiment.json', metadata)
        options = vars(args).copy(); options.update(config)
        predicted = deepcopy(faces)
        finish(child, deepcopy(photos), predicted, vectors, Namespace(**options), started, 0., 0., cache, 'compare')
        summary = json.loads((child/'summary.json').read_text())
        metric = None
        if annotations:
            try:
                metric = pair_metrics(predicted, annotations)
            except ValueError as e:
                if 'No filled, included ground truth' not in str(e):
                    raise
            if metric:
                write_json(child/'evaluation.json', metric)
        row = dict(experiment=name, subset=subset, **config,
                   **{k: summary[k] for k in ('detected_faces','grouping_eligible_faces','groups','singleton_groups',
                     'unknown_faces','held_faces','quality_excluded_faces','reassigned_faces','automatic_assignment_rate',
                     'grouping_seconds','total_seconds')},
                   precision=metric['precision'] if metric else None,
                   recall=metric['recall'] if metric else None, f1=metric['f1'] if metric else None,
                   evaluated_faces=metric['evaluated_faces'] if metric else 0,
                   ground_truth_coverage=metric['ground_truth_coverage'] if metric else 0,
                   mixed_groups=len(metric['mixed_groups']) if metric else None,
                   split_people=len(metric['split_people']) if metric else None,
                   same_photo_review_groups=summary['same_photo_review_groups'])
        rows.append(row)
        results[name] = dict(config=config, summary=summary, evaluation=metric,
                             same_photo_review_faces=[f['face_id'] for f in predicted if f['review']])
        current = {f['face_id']: f for f in predicted}
        current_memberships = memberships(predicted)
        if reference is None:
            reference, reference_memberships = current, current_memberships
        for fid, f in current.items():
            old = reference[fid]
            changes.append(dict(experiment=name, face_id=fid, photo_id=f['photo_id'],
                                before_group=old['group_id'], after_group=f['group_id'],
                                group_id_changed=old['group_id'] != f['group_id'],
                                membership_changed=reference_memberships[fid] != current_memberships[fid],
                                before_state=old['decision_state'], after_state=f['decision_state'],
                                before_reason=old['reason'], after_reason=f['reason'],
                                reassigned=f['diagnostic'].get('reassignment',{}).get('applied',False)))
    write_csv(parent/'comparison.csv', rows, list(rows[0]))
    write_csv(parent/'membership_changes.csv', changes, list(changes[0]))
    extraction = cache['signature']['extraction']
    report = dict(schema=1, source=str(source), subset=subset, extraction=extraction,
                  split_sha256=split_hash(splits) if splits else None, context_sha256=context_hash(cache),
                  features_sha256=cache['features_sha256'], experiments=results,
                  counts_are_not_accuracy=True, automatic_best_selection=None,
                  timing='grouping_seconds includes diagnostics; total_seconds includes local rendering/copy, excludes source extraction',
                  evaluation_policy='all is exploratory only; dev chooses settings explicitly; eval applies frozen selection only',
                  changes_reference=configs[0][0])
    write_json(parent/'comparison.json', report)
    render_comparison(parent, rows, changes, extraction)
    print(f'Comparison: {parent}/index.html\nNo setting automatically selected.')
    return parent


def select_config(args):
    parent = Path(args.comparison).resolve()
    report = json.loads((parent/'comparison.json').read_text())
    if report.get('subset') != 'dev' or not report.get('split_sha256'):
        raise ValueError('Only a dev comparison with a photo split can produce a frozen selection')
    if args.experiment not in report['experiments']:
        raise ValueError('Experiment not found')
    config = report['experiments'][args.experiment]['config']
    target = parent/('selected_config_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json')
    write_json(target, dict(schema=1, selected_on='dev', selection='explicit user choice; no automatic count ranking',
                           experiment=args.experiment, config=config, split_sha256=report['split_sha256'],
                           context_sha256=report['context_sha256'], features_sha256=report['features_sha256'],
                           comparison_sha256=sha256(parent/'comparison.json')))
    print(f'Frozen selection: {target}')
    return target


def render_comparison(parent, rows, changes, extraction):
    columns = [('experiment','실험'),('linkage','방식'),('similarity_threshold','유사도 기준'),('ambiguity_margin','모호성 폭'),
               ('min_group_size','최소 크기'),('detected_faces','검출'),('grouping_eligible_faces','대상'),('groups','그룹'),
               ('singleton_groups','단독'),('unknown_faces','미확인 전체'),('held_faces','품질통과 보류'),
               ('quality_excluded_faces','품질 제외'),('reassigned_faces','재배정'),('precision','P'),('recall','R'),('f1','F1'),
               ('evaluated_faces','평가 얼굴'),('ground_truth_coverage','정답 비율'),('automatic_assignment_rate','자동 배정 비율'),
               ('grouping_seconds','그룹화 초'),('total_seconds','결과 생성 초'),('mixed_groups','혼합 오류'),('split_people','분리 오류')]
    page = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>얼굴 그룹화 비교</title>',
            '<style>body{font:15px sans-serif;margin:24px}td,th{padding:7px;border:1px solid #ddd}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>',
            '<h1>동일 임베딩 얼굴 그룹화 비교</h1><p>그룹 수는 실제 인원 수가 아닙니다. 정답 없는 오류·지표는 미평가입니다. 최적 설정을 자동 선택하지 않습니다.</p>',
            '<p>미확인 전체 = 품질통과 보류 + 품질 제외. 실험 간 번호 변경과 실제 구성원 변경을 구별합니다. 같은 사진 중복은 검토 표시이며 병합 금지 규칙이 아닙니다.</p>',
            '<p><a href="comparison.csv">요약 CSV</a> · <a href="comparison.json">전체 설정·평가·오류 사례 JSON</a> · <a href="membership_changes.csv">얼굴별 소속 변경 CSV</a> · <a href="ground_truth_template.csv">정답 양식</a> · <a href="split_template.csv">사진별 dev/eval 양식</a></p>',
            '<p>공통 추출 설정: '+escape(json.dumps(extraction, ensure_ascii=False))+'</p>',
            '<table><tr>'+''.join('<th>'+title+'</th>' for _,title in columns)+'</tr>']
    for name, title in [('analysis_report.md','진단·실행 안내'),('preprocessing_audit.json','전처리·보존 검사'),('preprocessing_preview.jpg','좌표·정렬 확인 예')]:
        if (parent/name).exists():
            page.insert(-1, f'<p><a href="{name}">{title}</a></p>')
    for row in rows:
        cells = []
        for key,_ in columns:
            value = row[key]
            text = '미평가' if value is None else f'{value:.5f}' if isinstance(value,float) else str(value)
            cell = escape(text)
            if key == 'experiment':
                cell = f'<a href="{escape(row[key])}/index.html">{cell}</a>'
            cells.append('<td>'+cell+'</td>')
        page.append('<tr>'+''.join(cells)+'</tr>')
    page.append('</table><h2>실제 적용한 전체 설정</h2><pre>'+escape(json.dumps(rows, ensure_ascii=False, indent=2))+'</pre>')
    page.append('<h2>구성원 변경·재배정 얼굴</h2><table><tr><th>실험</th><th>얼굴</th><th>기준선 → 현재</th><th>사유</th></tr>')
    for c in changes:
        if not c['membership_changed'] and not c['reassigned']:
            continue
        name, fid = c['experiment'], c['face_id']
        page.append(f'<tr><td>{escape(name)}</td><td><a href="{escape(name)}/index.html#{fid}">{fid}</a></td>'
                    f'<td>{escape(c["before_group"] or "미확인")} → {escape(c["after_group"] or "미확인")}</td>'
                    f'<td>{escape(c["before_reason"])} → {escape(c["after_reason"])}; 재배정={c["reassigned"]}</td></tr>')
    page.append('</table><h2>오류·검토 사례</h2>')
    report = json.loads((parent/'comparison.json').read_text())
    for name, result in report['experiments'].items():
        metric = result['evaluation']
        page.append('<h3>'+escape(name)+'</h3>')
        page.append('<pre>'+escape(json.dumps(dict(mixed_groups=metric['mixed_groups'] if metric else '정답 미입력: 미평가',
                    split_people=metric['split_people'] if metric else '정답 미입력: 미평가',
                    same_photo_review_faces=result['same_photo_review_faces']), ensure_ascii=False, indent=2))+'</pre>')
    page.append('</html>')
    (parent/'index.html').write_text('\n'.join(page), encoding='utf-8')
