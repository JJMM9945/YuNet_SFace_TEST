"""Optional local parity check against the preserved selected .75 extraction."""
import json
from pathlib import Path
import numpy as np
from face_lab.engine import FaceEngine


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'tests/private/results/여러크기 + 0.75/run_20261103T064516_657319Z_ab5259'
    cache = json.loads((source / 'cache.json').read_text(encoding='utf-8'))
    with np.load(source / 'features.npz', allow_pickle=False) as data:
        expected_vectors = dict(zip(data['face_ids'].tolist(), data['vectors']))
    engine = FaceEngine(root / 'models')
    count, max_error = 0, 0.
    for photo in cache['photos']:
        faces, vectors = engine.extract_photo((root / 'ljm_dataset' / photo['relative_path']).read_bytes(), photo['photo_id'])
        expected = [f for f in cache['faces'] if f['photo_id'] == photo['photo_id']]
        assert len(faces) == len(expected), 'Detection count differs'
        for current, old in zip(faces, expected):
            for key in ('face_id', 'quality_reason', 'x', 'y', 'width', 'height', 'landmarks', 'detection_score'):
                assert current[key] == old[key], f'Detection/quality differs: {key}'
        assert set(vectors) == {f['face_id'] for f in expected if f['face_id'] in expected_vectors}
        for fid, vector in vectors.items():
            error = float(np.max(np.abs(vector - expected_vectors[fid])))
            max_error = max(max_error, error)
            np.testing.assert_allclose(vector, expected_vectors[fid], rtol=0, atol=1e-5)
        count += len(faces)
    print(f'PASS: {len(cache["photos"])} photos, {count} faces; max vector component error={max_error}')


if __name__ == '__main__':
    main()
