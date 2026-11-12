"""Selected .75 reference CLI for device checks and arbitrary photo folders."""
import argparse
import json
from pathlib import Path
import numpy as np

from .engine import FaceEngine


def main():
    parser = argparse.ArgumentParser(description='YuNet multiscale .75 + SFace device-port reference')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare-models', 'check', 'extract', 'albums'):
        item = sub.add_parser(command)
        item.add_argument('--models', default='models')
        if command == 'albums':
            item.add_argument('--input', required=True, help='Your photo folder (recursive)')
            item.add_argument('--output', default='results/photo_albums', help='Result parent folder outside input')
        if command == 'extract':
            item.add_argument('--input', required=True, help='One image file')
            item.add_argument('--photo-id', required=True, help='Stable app photo ID')
            item.add_argument('--output', required=True, help='New result directory')
    args = parser.parse_args()
    if args.command == 'albums':
        from .photo_albums import run
        try:
            run(args.input, args.output, args.models)
        except (ValueError, OSError) as error:
            parser.exit(2, f'ERROR: {error}\n')
        return
    if args.command == 'prepare-models':
        from .models import prepare
        prepare(args.models)
        return
    engine = FaceEngine(args.models)
    if args.command == 'check':
        engine.detector.setInputSize((320, 320))
        engine.detector.detect(np.zeros((320, 320, 3), np.uint8))
        feature = engine.recognizer.feature(np.zeros((112, 112, 3), np.uint8))
        if feature.size != 128 or not np.isfinite(feature).all():
            raise ValueError('Invalid SFace forward pass')
        print('OK: YuNet + SFace CPU forward passes')
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    faces, vectors = engine.extract_photo(Path(args.input).read_bytes(), args.photo_id)
    (output / 'faces.json').write_text(json.dumps(faces, ensure_ascii=False, indent=2), encoding='utf-8')
    ids = sorted(vectors)
    np.savez_compressed(output / 'features.npz', face_ids=np.array(ids, dtype=str),
                        vectors=np.stack([vectors[k] for k in ids]) if ids else np.empty((0, 128), np.float32))
    print(f'{len(faces)} faces; {len(vectors)} vectors -> {output}')


if __name__ == '__main__':
    main()
