"""Pin scientific source and artifacts before subsequent reporting-only edits."""
from pathlib import Path
import os
import shutil
import tempfile

from .io import atomic_json, read_json, safe_path, sha256_file, utc_now

INPUTS = ('poc', 'config', 'schemas', 'assets', 'tests', 'decisions', 'evidence',
          'data', 'locks', 'wheelhouse', 'docs', 'reports', 'README.md',
          'requirements.in', 'POC_EXECUTION_PLAN.md')
SHARED_IMMUTABLE_SUFFIXES = {'.npz', '.pt', '.whl'}


def create_snapshot(root: Path, destination: str) -> dict:
    root = root.resolve()
    target = safe_path(root, destination)
    if not target.is_relative_to(root / 'study') or target == root / 'study':
        raise ValueError('Study archive destination must be inside study/')
    if target.exists():
        raise ValueError('Study archive destination already exists')
    if any((root / 'locks/active').glob('*.json')):
        raise ValueError('Cannot freeze a study with an active job lock; finish or explicitly resume that job first')
    sources = []
    for name in INPUTS:
        item = root / name
        if item.is_symlink():
            raise ValueError(f'Symlink archive input: {name}')
        if not item.exists():
            continue
        for path in sorted(item.rglob('*')) if item.is_dir() else [item]:
            if path.is_symlink():
                raise ValueError(f'Symlink archive input: {path.relative_to(root)}')
            if '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            if path.is_file():
                sources.append(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.study-', dir=target.parent))
    try:
        entries = []
        for source in sources:
            relative = str(source.relative_to(root))
            expected = sha256_file(source)
            copied = temporary / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            shared = source.suffix in SHARED_IMMUTABLE_SUFFIXES
            if shared:
                os.link(source, copied)
            else:
                shutil.copyfile(source, copied)
            if sha256_file(copied) != expected or sha256_file(source) != expected:
                raise ValueError(f'Input changed during snapshot: {relative}')
            entries.append({'path': relative, 'sha256': expected, 'bytes': copied.stat().st_size,
                            'storage': 'shared_immutable' if shared else 'copied'})
        record = {'schema_version': 1, 'protocol_version': 'PTCO-1.0', 'created_utc': utc_now(),
                  'files': entries, 'binary_policy': 'Hardlinks only to immutable NPZ, tensor archives and wheels; never modify either name.'}
        atomic_json(temporary / 'STUDY_SNAPSHOT.json', record)
        os.rename(temporary, target)
        return record
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_snapshot(root: Path) -> dict:
    root = root.resolve()
    record = read_json(root / 'STUDY_SNAPSHOT.json')
    if record.get('schema_version') != 1 or record.get('protocol_version') != 'PTCO-1.0':
        raise ValueError('Unsupported study archive schema/protocol')
    seen = set()
    for item in record['files']:
        relative = item['path']
        if relative in seen:
            raise ValueError('Duplicate archive path')
        seen.add(relative)
        path = safe_path(root, relative)
        if any(part.is_symlink() for part in (root / relative, *(root / relative).parents) if part.is_relative_to(root)):
            raise ValueError(f'Symlink archive input: {relative}')
        if sha256_file(path) != item['sha256'] or path.stat().st_size != item['bytes']:
            raise ValueError(f'Study archive hash mismatch: {relative}')
    return record
