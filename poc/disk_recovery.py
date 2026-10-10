"""Bridge a closed disk-failed receipt to the existing exact-resume interface."""
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import uuid

from .contracts import verify_run
from .io import atomic_json, read_json, sha256_file, utc_now


def recover_disk_fit(root: Path, failed_run_id: str) -> Path:
    root = Path(root).resolve()
    if not failed_run_id or Path(failed_run_id).name != failed_run_id:
        raise ValueError('Exact storage-failed training run ID required')
    original = root / 'evidence/M30' / failed_run_id
    record = read_json(original / 'manifest.json')
    verify_run(root, record)
    failure = read_json(original / 'summary.json').get('failure', '')
    if (record['completion_status'] != 'failed' or record['milestone_id'] != 'M30'
            or failure != '[Errno 28] No space left on device'):
        raise ValueError('Only an identified storage failure can use this recovery bridge')
    marker = read_json(original / 'resume.complete.json')
    if (marker['checkpoint_sha256'] != sha256_file(original / 'resume.pt')
            or marker['rng_sha256'] != sha256_file(original / 'resume.rng.json')):
        raise ValueError('Storage-failed checkpoint hash closure does not verify')
    names = ('request.json', 'resume.pt', 'resume.rng.json', 'resume.complete.json',
             'weights.pt', 'validation_history.json')
    source_hashes = {name: sha256_file(original / name) for name in names}
    for path in (root / 'evidence/M30').glob('*/RECOVERED_FROM.json'):
        lineage = read_json(path)
        if lineage['failed_run_id'] == failed_run_id:
            if lineage['source_hashes'] != source_hashes:
                raise ValueError('Existing storage recovery conflicts with the original checkpoint')
            return path.parent
    identifier = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    target = root / 'evidence/M30' / identifier
    target.mkdir()
    for name in names:
        if name.endswith('.pt'):
            os.link(original / name, target / name)
        else:
            shutil.copyfile(original / name, target / name)
        if sha256_file(target / name) != source_hashes[name]:
            raise ValueError('Checkpoint changed while constructing the exact-resume bridge')
    atomic_json(target / 'RECOVERED_FROM.json', {
        'schema_version': 1, 'failed_run_id': failed_run_id,
        'failed_manifest_sha256': sha256_file(original / 'manifest.json'),
        'job_key': record['job_key'], 'source_hashes': source_hashes,
        'optimizer_update': marker['update'], 'created_utc': utc_now(),
        'classification': 'operational storage interruption; no scientific outcome',
        'purpose': 'Hash-identical unfinished input for the unchanged training_job --resume implementation'})
    return target
