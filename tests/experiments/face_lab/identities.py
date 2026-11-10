"""Persistent, explicit face-to-identity links; grouping IDs are never identity IDs."""
import hashlib
import json
import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone

from .review import context_hash


def extraction_binding(cache):
    return dict(context_sha256=context_hash(cache), features_sha256=cache['features_sha256'])


def empty_store(cache, source_faces_sha256, excluded=()):
    expected = {face['face_id'] for face in cache['faces']}
    if not set(excluded) <= expected:
        raise ValueError('Excluded faces must belong to the extraction')
    return dict(schema=1, extraction=extraction_binding(cache), source_faces_sha256=source_faces_sha256,
                identities={}, links={}, excluded_face_ids=sorted(excluded), revision=0, events=[])


def validate_store(store, cache):
    if store.get('schema') != 1 or store.get('extraction') != extraction_binding(cache):
        raise ValueError('Identity store extraction/model/feature mismatch')
    expected = {face['face_id'] for face in cache['faces']}
    if not set(store['links']) <= expected or not set(store['excluded_face_ids']) <= expected:
        raise ValueError('Identity store has unknown face IDs')
    if set(store['links']) & set(store['excluded_face_ids']):
        raise ValueError('Excluded faces cannot have identity links')
    if any(identity not in store['identities'] for identity in store['links'].values()):
        raise ValueError('Identity link points to an unknown identity')
    return store


def register_identity(store, confirmed_alias):
    """Called only for an explicit user identity, never an automatic cluster name."""
    if not isinstance(confirmed_alias, str) or not confirmed_alias.strip():
        raise ValueError('An explicit identity alias is required')
    namespace = store['extraction']['context_sha256'] + ':' + confirmed_alias
    identity = 'identity_' + hashlib.sha256(namespace.encode('utf-8')).hexdigest()[:24]
    if identity not in store['identities']:
        store['identities'][identity] = dict(display_name=confirmed_alias)
    elif store['identities'][identity]['display_name'] != confirmed_alias:
        raise ValueError('Identity alias collision')
    return identity


def apply_identity_changes(store, cache, changes, reason, actor='user', undo_of=None):
    """Validate atomically, return a new revision, and retain the complete prior history."""
    validate_store(store, cache)
    if not isinstance(changes, dict) or not changes or not isinstance(reason, str) or not reason.strip():
        raise ValueError('Explicit face-ID changes and a reason are required')
    expected = {face['face_id'] for face in cache['faces']}
    for fid, target in changes.items():
        if fid not in expected:
            raise ValueError('Unknown face ID')
        if fid in store['excluded_face_ids']:
            raise ValueError('Cannot link a confirmed excluded face')
        if target is not None and target not in store['identities']:
            raise ValueError('Unknown identity ID')
    result = deepcopy(store)
    before = {fid: result['links'].get(fid) for fid in sorted(changes)}
    for fid, target in changes.items():
        if target is None:
            result['links'].pop(fid, None)
        else:
            result['links'][fid] = target
    result['revision'] += 1
    result['events'].append(dict(revision=result['revision'],
                                  timestamp=datetime.now(timezone.utc).isoformat(), actor=actor,
                                  reason=reason, before=before, after=dict(sorted(changes.items())),
                                  undo_of=undo_of))
    return validate_store(result, cache)


def undo_last_change(store, cache):
    if not store['events']:
        raise ValueError('There is no identity change to undo')
    event = store['events'][-1]
    return apply_identity_changes(store, cache, event['before'], 'Undo revision ' + str(event['revision']),
                                  undo_of=event['revision'])


def identity_collections(faces, store):
    """Derive search results without modifying a face's automatic group membership."""
    by_id = {face['face_id']: face for face in faces}
    if len(by_id) != len(faces) or set(store['links']) - set(by_id):
        raise ValueError('Identity links and source face mapping differ')
    collections = defaultdict(lambda: dict(face_ids=[], photo_ids=set(), automatic_groups=set(), unknown_faces=[]))
    for fid, identity in sorted(store['links'].items()):
        face = by_id[fid]
        row = collections[identity]
        row['face_ids'].append(fid)
        row['photo_ids'].add(face['photo_id'])
        if face['group_id']:
            row['automatic_groups'].add(face['group_id'])
        else:
            row['unknown_faces'].append(fid)
    return {identity: dict(display_name=store['identities'][identity]['display_name'],
                           face_ids=row['face_ids'], photo_ids=sorted(row['photo_ids']),
                           automatic_groups=sorted(row['automatic_groups']), unknown_faces=row['unknown_faces'])
            for identity, row in sorted(collections.items())}
