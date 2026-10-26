"""Deterministic complete-linkage on cosine distance, with conservative rejection."""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def group_faces(faces, vectors, threshold=0.45, min_group=2, margin=0.03, diagnostics=None):
    for f in faces:
        f.update(group_id="", label="미확인", status="unknown", review="", reason=f.get("quality_reason", ""))
    eligible = [i for i, f in enumerate(faces) if not f["reason"] and f["face_id"] in vectors]
    if not eligible:
        return
    x = np.stack([vectors[faces[i]["face_id"]] for i in eligible])
    sim = np.clip(x @ x.T, -1, 1)
    if len(x) == 1:
        labels = [1]
    else:
        distance = np.clip(1 - sim.astype(np.float64), 0, 2)
        distance = (distance + distance.T) / 2
        np.fill_diagonal(distance, 0)
        labels = fcluster(linkage(squareform(distance), method="complete"), t=1-threshold, criterion="distance")
    clusters = {}
    for local, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(local)
    if diagnostics is not None:
        diagnostics.clear()
        for members in clusters.values():
            for i in members:
                own_members=[j for j in members if j!=i]
                competitors=[]
                for other_members in clusters.values():
                    if other_members is members:
                        continue
                    competitors.append(dict(face_ids=[faces[eligible[j]]['face_id'] for j in other_members],
                        min_similarity=float(min(sim[i,j] for j in other_members)),
                        max_similarity=float(max(sim[i,j] for j in other_members))))
                competitors.sort(key=lambda c:(-c['min_similarity'],c['face_ids'][0]))
                own=float(min((sim[i,j] for j in own_members),default=1.0))
                other=competitors[0]['min_similarity'] if competitors else -1.0
                diagnostics[faces[eligible[i]]['face_id']]=dict(
                    own_face_ids=[faces[eligible[j]]['face_id'] for j in members],
                    own_min_similarity=own,own_singleton=not own_members,
                    competing_min_similarity=other,gap=own-other,
                    competitors=competitors,
                    ambiguity_rule_matched=bool(len(members)>=min_group and other>=threshold-margin and own-other<=margin))
    # Use the same complete-link criterion for both own and competing clusters.
    # A single strong edge to an otherwise incompatible cluster is not sufficient
    # evidence of ambiguous identity (and would reintroduce single-link chaining).
    ambiguous = set()
    for members in clusters.values():
        if len(members) < min_group:
            continue
        for i in members:
            own = min((sim[i, j] for j in members if j != i), default=1.0)
            other = max((min(sim[i, j] for j in competitor)
                         for competitor in clusters.values() if competitor is not members), default=-1.0)
            if other >= threshold - margin and own - other <= margin:
                ambiguous.add(i)
    ordered = sorted(clusters.values(), key=lambda members: faces[eligible[min(members)]]["face_id"])
    number = 0
    for members in ordered:
        kept = [i for i in members if i not in ambiguous]
        for i in members:
            faces[eligible[i]]["reason"] = "ambiguous_competing_cluster" if i in ambiguous else "below_min_group_size"
        if len(kept) < min_group:
            continue
        number += 1
        gid = f"person{number:03d}"
        seen = [faces[eligible[i]]["photo_id"] for i in kept]
        for i in kept:
            f = faces[eligible[i]]
            f.update(group_id=gid, label=f"인물{number:03d}", status="grouped", reason="")
            if seen.count(f["photo_id"]) > 1:
                f["review"] = "same_photo_same_group"
