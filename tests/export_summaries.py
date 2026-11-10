"""Print aggregate-only experiment evidence for Git, excluding face IDs and paths."""
import json
from pathlib import Path


def aggregate(value):
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        if isinstance(item, (int, float, bool)) or item is None:
            result[key] = item
        elif key in {'linkage', 'mode', 'status'} and isinstance(item, str):
            result[key] = item
        elif isinstance(item, dict):
            child = aggregate(item)
            if child:
                result[key] = child
    return result


def main():
    root = Path(__file__).resolve().parent / 'private' / 'results'
    rows = []
    for path in sorted(root.rglob('*.json')):
        if path.name not in {'summary.json', 'comparison.json', 'experiment.json', '비교결과.json', '진단.json', 'identity_summary.json'}:
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        rows.append(dict(experiment=path.relative_to(root).as_posix(), aggregates=aggregate(data)))
    print(json.dumps(dict(note='Aggregate evidence only; no images, vectors, face IDs or absolute source paths.',
                         experiments=rows), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
