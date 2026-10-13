"""Fixed, lossless level-1 gzip encoding for large calibration JSONL streams."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import tempfile

from .io import canonical_json


def write_records(path: Path, rows) -> None:
    """Stream canonical records into an atomic gzip product, outside scorer clocks."""
    path = Path(path)
    if path.suffixes[-2:] != ['.jsonl', '.gz']:
        raise ValueError('Compressed calibration records require .jsonl.gz')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(filename='', mode='wb', compresslevel=1, mtime=0, fileobj=raw) as stream:
                for row in rows:
                    stream.write(canonical_json(row)+b'\n')
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_records(path: Path) -> list[dict]:
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]
