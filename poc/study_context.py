"""Keep reporting changes separate from the scientific source that produced evidence."""
from pathlib import Path
import os
import subprocess
import sys

from .io import read_json, safe_path, sha256_file


def source_context(root):
    root = Path(root).resolve()
    selector = root / 'config/report_source.json'
    if not selector.exists():
        return None
    value = read_json(selector)
    if set(value) != {'schema_version', 'protocol_version', 'study_relative', 'snapshot_sha256'}:
        raise ValueError('Unexpected report source selector fields')
    if value['schema_version'] != 1 or value['protocol_version'] != 'PTCO-1.0':
        raise ValueError('Unsupported report source selector')
    study = safe_path(root, value['study_relative'])
    if not study.is_relative_to(root / 'study') or study == root / 'study':
        raise ValueError('Scientific source must be in the project archive')
    if sha256_file(study / 'STUDY_SNAPSHOT.json') != value['snapshot_sha256']:
        raise ValueError('Selected scientific snapshot hash changed')
    from .study_archive import verify_snapshot
    verify_snapshot(study)
    return study


def call_source(root, command):
    study = source_context(root)
    if study is None:
        raise ValueError('No completed scientific source has been selected')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[name] = '1'
    from .study_reproduce import command_argv
    return subprocess.run(command_argv(sys.executable, study, command), cwd=study,
                          env=env, capture_output=True, text=True)


def render_report(root):
    root = Path(root).resolve()
    study = source_context(root)
    from .gates import current_decision
    decision = current_decision(study) if study is not None else None
    if decision is None:
        raise ValueError('Selected scientific source lacks a completed gate decision')
    from .study_report import write_scientific_report
    from .study_reproduce import write_reproduction
    from .study_release import validate_report_tables
    state = write_scientific_report(study, [decision], output_root=root)
    write_reproduction(root, study, decision['gate_id'])
    validate_report_tables(root, study, [decision])
    return state
