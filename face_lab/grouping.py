"""Deterministic agglomerative cosine grouping and frozen multi-reference recovery.

Complete: max cross distance = 1 - min cross similarity.
Average (UPGMA): mean cross distance = 1 - mean cross similarity.
Both cut at distance <= 1-threshold; neither is single linkage.
"""
from collections import Counter
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def validate_vectors(vectors, dimension=None):
    for fid, value in vectors.items():
        v = np.asarray(value)
        if (v.ndim != 1 or not v.size or (dimension is not None and v.size != dimension)
                or not np.isfinite(v).all() or not np.isclose(np.linalg.norm(v), 1, atol=1e-4)):
            raise ValueError(f"Invalid/non-unit embedding: {fid}")
    if len({len(v) for v in vectors.values()}) > 1:
        raise ValueError("Embedding dimensions differ")


def mark_same_photo(faces):
    counts = Counter((f['photo_id'], f['group_id']) for f in faces if f['group_id'])
    for f in faces:
        f['review'] = 'same_photo_same_group' if f['group_id'] and counts[(f['photo_id'], f['group_id'])] > 1 else ''


def candidate_fields(candidates, threshold, margin):
    best = candidates[0] if candidates else None
    second = candidates[1] if len(candidates) > 1 else None
    gap = best['score'] - second['score'] if second else None
    return dict(top1_group_id=best['group_id'] if best else None, top1_score=best['score'] if best else None,
                top2_group_id=second['group_id'] if second else None, top2_score=second['score'] if second else None,
                candidate_gap=gap, score_threshold=threshold, margin_threshold=margin,
                group_criterion_unmet=best['score'] < threshold if best else None,
                candidate_margin_insufficient=gap < margin if gap is not None else None,
                candidates=candidates)


def group_faces(faces, vectors, threshold=0.45, min_group=2, margin=0.03, diagnostics=None, method='complete'):
    if method not in {'complete', 'average'}:
        raise ValueError('linkage must be complete or average')
    if not -1 <= threshold <= 1 or not 0 <= margin <= 2 or min_group < 1:
        raise ValueError('Invalid grouping thresholds')
    if len({f['face_id'] for f in faces}) != len(faces):
        raise ValueError('Duplicate face IDs')
    validate_vectors(vectors)
    records = {}
    if diagnostics is not None:
        diagnostics.clear()
    for f in faces:
        reason = f.get('quality_reason', '') or ('' if f['face_id'] in vectors else 'missing_embedding')
        f.update(group_id='', label='미확인', status='unknown', review='', reason=reason)
        d = dict(face_id=f['face_id'], photo_id=f['photo_id'], relative_path=f.get('relative_path'),
                 quality_reason=reason, width=f.get('width'), height=f.get('height'),
                 face_size=min(f.get('width', 0), f.get('height', 0)), detection_score=f.get('detection_score'),
                 blur_variance=f.get('blur_variance'), linkage=method,
                 individual_similarity_definition='dot(L2-normalized embeddings); larger is closer',
                 group_score_definition=('minimum' if method == 'complete' else 'arithmetic mean') + ' of member similarities; self excluded',
                 nearest_face_id=None, nearest_cosine_similarity=None, initial_cluster_id=None,
                 own_group_score=None, singleton_prior=None, own_competitor_gap=None,
                 **candidate_fields([], threshold, margin),
                 stages=[dict(stage='quality', state='quality_excluded' if reason else 'eligible', reason=reason)])
        records[f['face_id']] = d
    # Canonical face-ID order fixes input-order and SciPy's deterministic tie handling.
    eligible = sorted((i for i, f in enumerate(faces) if not f['reason']), key=lambda i: faces[i]['face_id'])
    if eligible:
        x = np.stack([vectors[faces[i]['face_id']] for i in eligible])
        sim = np.clip(x @ x.T, -1, 1)
        if len(x) == 1:
            labels = [1]
        else:
            distance = np.clip(1 - sim.astype(np.float64), 0, 2)
            distance = (distance + distance.T) / 2
            np.fill_diagonal(distance, 0)
            labels = fcluster(linkage(squareform(distance), method=method), t=1-threshold, criterion='distance')
        clusters = {}
        for local, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(local)
        ordered = sorted(clusters.values(), key=lambda m: faces[eligible[m[0]]]['face_id'])
        gids = {id(m): f'person{n:03d}' for n, m in enumerate(ordered, 1)}
        aggregate = np.min if method == 'complete' else np.mean
        ambiguous = set()
        for members in ordered:
            gid = gids[id(members)]
            for i in members:
                f = faces[eligible[i]]; d = records[f['face_id']]
                peers = [j for j in members if j != i]
                own = float(aggregate(sim[i, peers])) if peers else None
                # Baseline singleton prior is explicit, NEVER a measured candidate score.
                effective_own = own if own is not None else 1.0
                competitors = []
                candidates = []
                for other in ordered:
                    refs = [j for j in other if j != i]
                    if not refs:
                        continue
                    score = float(aggregate(sim[i, refs]))
                    c = dict(group_id=gids[id(other)], score=score,
                             face_ids=[faces[eligible[j]]['face_id'] for j in refs],
                             min_similarity=float(np.min(sim[i, refs])), max_similarity=float(np.max(sim[i, refs])),
                             is_own_cluster=other is members)
                    candidates.append(c)
                    if other is not members:
                        competitors.append(c)
                candidates.sort(key=lambda c: (-c['score'], c['group_id']))
                competitors.sort(key=lambda c: (-c['score'], c['group_id']))
                other_score = competitors[0]['score'] if competitors else None
                matched = bool(len(members) >= min_group and other_score is not None
                               and other_score >= threshold-margin and effective_own-other_score <= margin)
                if matched:
                    ambiguous.add(i)
                nearest = sorted((j for j in range(len(x)) if j != i),
                                 key=lambda j: (-float(sim[i,j]), faces[eligible[j]]['face_id']))
                d.update(candidate_fields(candidates, threshold, margin))
                d.update(initial_cluster_id=gid, own_group_score=own, singleton_prior=1.0 if not peers else None,
                         own_competitor_gap=effective_own-other_score if other_score is not None else None,
                         competitor_floor=threshold-margin, ambiguity_rule_matched=matched,
                         own_face_ids=[faces[eligible[j]]['face_id'] for j in members],
                         # Compatibility fields for the existing diagnostic utility.
                         own_min_similarity=float(np.min(sim[i,peers])) if peers else None,
                         own_singleton=not peers, competing_min_similarity=other_score,
                         gap=effective_own-other_score if other_score is not None else None, competitors=competitors,
                         nearest_face_id=faces[eligible[nearest[0]]]['face_id'] if nearest else None,
                         nearest_cosine_similarity=float(sim[i,nearest[0]]) if nearest else None)
                d['stages'].append(dict(stage='clustering', state='provisional', group_id=gid,
                                        reason=f'{method}_distance_cut', distance_threshold=1-threshold))
        for members in ordered:
            kept = [i for i in members if i not in ambiguous]
            gid = gids[id(members)]
            for i in members:
                f = faces[eligible[i]]; d = records[f['face_id']]
                if i in ambiguous:
                    f['reason'] = 'ambiguous_competing_cluster'
                elif len(kept) < min_group:
                    f['reason'] = 'below_min_group_size_after_ambiguity' if len(members) >= min_group else 'below_min_group_size'
                else:
                    f.update(group_id=gid, label='인물'+gid[6:], status='grouped', reason='')
                d['stages'].append(dict(stage='ambiguity', state='held' if i in ambiguous else 'passed',
                                        reason='ambiguous_competing_cluster' if i in ambiguous else 'no_competing_ambiguity'))
                d['stages'].append(dict(stage='minimum_size', state='assigned' if f['group_id'] else 'held',
                                        group_id=f['group_id'], reason=f['reason'], kept_cluster_size=len(kept)))
        surviving = {f['group_id'] for f in faces if f['group_id']}
        for d in records.values():
            for c in d['candidates']:
                c['surviving_group_id'] = c['group_id'] if c['group_id'] in surviving else None
    # Nearest individual uses every valid cached vector (including quality exclusions),
    # independently of whether that face can participate in automatic grouping.
    indexed = {f['face_id']: f for f in faces}
    for fid, f in indexed.items():
        d = records[fid]
        if fid in vectors:
            nearest = sorted((other for other in indexed if other != fid and other in vectors),
                             key=lambda other: (-float(np.clip(vectors[fid] @ vectors[other], -1, 1)), other))
            d['nearest_face_id'] = nearest[0] if nearest else None
            d['nearest_cosine_similarity'] = float(np.clip(vectors[fid] @ vectors[nearest[0]], -1, 1)) if nearest else None
            if d['quality_reason'] and eligible:
                candidates = []
                for members in ordered:
                    refs = [faces[eligible[j]]['face_id'] for j in members]
                    scores = [float(np.clip(vectors[fid] @ vectors[ref], -1, 1)) for ref in refs]
                    candidates.append(dict(group_id=gids[id(members)], score=float(aggregate(scores)), face_ids=refs,
                                           min_similarity=min(scores), max_similarity=max(scores), is_own_cluster=False,
                                           surviving_group_id=gids[id(members)] if gids[id(members)] in surviving else None))
                candidates.sort(key=lambda c: (-c['score'], c['group_id']))
                d.update(candidate_fields(candidates, threshold, margin))
                d['candidates_diagnostic_only'] = True
        d['nearest_scope'] = 'all detected faces with valid embeddings in this experiment, including quality exclusions'
        d['own_group_criterion_unmet'] = d['own_group_score'] < threshold if d['own_group_score'] is not None else None
    mark_same_photo(faces)
    for f in faces:
        d = records[f['face_id']]
        d.update(final_group_id=f['group_id'], state='assigned' if f['group_id'] else ('quality_excluded' if d['quality_reason'] else 'held'), reason=f['reason'])
        f['decision_state'] = d['state']
        f['diagnostic'] = d
    if diagnostics is not None:
        diagnostics.update(records)
    return records


def quality_score(f):
    """Geometric mean: detector confidence, min-side/160, aligned blur/200 (all clipped to [0,1])."""
    values = [f.get('detection_score', 0) or 0, min(f.get('width',0), f.get('height',0))/160,
              (f.get('blur_variance',0) or 0)/200]
    return float(np.prod(np.clip(values, 0, 1)) ** (1/3))


def select_representatives(members, vectors, count=3, diversity_weight=0.3):
    remaining = sorted(members, key=lambda f: f['face_id'])
    selected, audit = [], []
    while remaining and len(selected) < count:
        scored = []
        for f in remaining:
            q = quality_score(f)
            diversity = min((float(1-np.clip(vectors[f['face_id']] @ vectors[s['face_id']], -1, 1))/2 for s in selected), default=0)
            value = (1-diversity_weight)*q + diversity_weight*diversity if selected else q
            scored.append((value, f, q, diversity))
        value, f, q, diversity = min(scored, key=lambda t: (-t[0], t[1]['face_id']))
        audit.append(dict(face_id=f['face_id'], quality_score=q, diversity=diversity, selection_score=value))
        selected.append(f); remaining.remove(f)
    return selected, audit


def reassign_unknown(faces, vectors, threshold=.45, margin=.03, representative_count=3,
                     min_support=2, support_threshold=.4, diversity_weight=.3):
    if representative_count < min_support or min_support < 2:
        raise ValueError('Reassignment requires at least two supporting representatives')
    groups = {}
    for f in faces:
        if f['group_id'] and not f.get('quality_reason'):
            groups.setdefault(f['group_id'], []).append(f)
    frozen, selections = {}, {}
    for gid, members in sorted(groups.items()):
        frozen[gid], selections[gid] = select_representatives(members, vectors, representative_count, diversity_weight)
    # All scores refer ONLY to this frozen snapshot. Recoveries cannot propagate.
    for f in sorted(faces, key=lambda f: f['face_id']):
        d = f['diagnostic']
        before = dict(state=d['state'], group_id=f['group_id'], reason=f['reason'])
        d['before_reassignment'] = before
        if f['group_id'] or d['quality_reason']:
            d['reassignment'] = dict(applied=False, reason='already_assigned' if f['group_id'] else 'quality_excluded')
            continue
        candidates = []
        for gid, reps in frozen.items():
            scores = [float(np.clip(vectors[f['face_id']] @ vectors[r['face_id']], -1, 1)) for r in reps]
            candidates.append(dict(group_id=gid, score=float(np.mean(scores)),
                                   representatives=[dict(face_id=r['face_id'], cosine_similarity=s) for r,s in zip(reps,scores)],
                                   support_count=sum(s >= support_threshold for s in scores)))
        candidates.sort(key=lambda c: (-c['score'], c['group_id']))
        record = candidate_fields(candidates, threshold, margin)
        best = candidates[0] if candidates else None
        if not best:
            reason = 'no_candidate_groups'
        elif best['score'] < threshold:
            reason = 'representative_group_score_below_threshold'
        elif best['support_count'] < min_support:
            reason = 'insufficient_representative_support'
        elif record['candidate_gap'] is not None and record['candidate_gap'] < margin:
            reason = 'representative_candidate_margin_insufficient'
        else:
            reason = 'frozen_multi_representative_assignment'
            f.update(group_id=best['group_id'], label='인물'+best['group_id'][6:], status='grouped', reason='')
        applied = bool(f['group_id'])
        if not applied:
            f['reason'] = reason
        record.update(applied=applied, reason=reason, min_support=min_support, support_threshold=support_threshold,
                      score_definition='arithmetic mean of ALL frozen representative cosine similarities',
                      missing_runner_up_policy='no competition test when only one group; score and >=2 supports still required')
        d['reassignment'] = record
        d['stages'].append(dict(stage='reassignment', state='assigned' if applied else 'held', group_id=f['group_id'], reason=reason))
        d.update(final_group_id=f['group_id'], state='assigned' if applied else 'held', reason=f['reason'])
        f['decision_state'] = d['state']
    mark_same_photo(faces)
    return dict(representatives=selections, score='mean_all_representatives', threshold=threshold, margin=margin,
                representative_count=representative_count, min_support=min_support, support_threshold=support_threshold,
                diversity_weight=diversity_weight, quality_formula='cuberoot(clip(det,0,1)*clip(min_side/160,0,1)*clip(blur/200,0,1))',
                selection_formula='first: quality; subsequent: (1-diversity_weight)*quality + diversity_weight*min((1-cos)/2)',
                frozen=True)
