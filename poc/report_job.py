"""Immutable MF receipts and independent release checks for a pinned study."""
from pathlib import Path
import json
import shutil

from .contracts import verify_run
from .execution import run_job
from .gates import current_decision, decide, load_inputs
from .io import atomic_json, read_json, sha256_file
from .study_context import source_context, render_report, call_source
from .study_release import validate_report_tables


CHECKS = ('source_closure', 'claim_tables', 'report_artifacts', 'reproduction_contract')
ARTIFACTS = ('final_report.md', 'claims.csv', 'run_index.csv', 'state.json',
             'independent_primary_audit.json', 'paired_primary_results.json',
             'reproduce.sh', 'reproduction_request.json')
FILES = ('poc/__init__.py', 'poc/__main__.py', 'poc/contracts.py', 'poc/gates.py',
         'poc/io.py', 'poc/execution.py', 'poc/report_job.py', 'poc/study_context.py',
         'poc/study_archive.py', 'poc/study_report.py', 'poc/finalization.py',
         'poc/criteria.py', 'poc/report.py', 'poc/study_audit.py',
         'poc/study_release.py', 'poc/study_reproduce.py')


def _selected(root):
    study = source_context(root)
    decision = current_decision(study) if study is not None else None
    if decision is None:
        raise ValueError('Selected scientific source lacks a completed gate decision')
    return study, decision


def validate_release(root):
    root = Path(root).resolve()
    study, decision = _selected(root)
    result = validate_report_tables(root, study, [decision])
    request = read_json(root / 'reports/reproduction_request.json')
    if (request['decision_id'] != decision['decision_id'] or request['outcome'] != decision['outcome']
            or request['study_relative'] != str(study.relative_to(root))
            or request['snapshot_sha256'] != sha256_file(study / 'STUDY_SNAPSHOT.json')
            or request['lock_sha256'] != sha256_file(study / 'locks/requirements.lock')):
        raise ValueError('Reproduction contract differs from the selected scientific evidence')
    for name in ARTIFACTS:
        if not (root / 'reports' / name).is_file():
            raise ValueError('Missing release artifact: ' + name)
    process = call_source(root, ['validate', '--scope', 'evidence'])
    if process.returncode:
        raise ValueError('Pinned scientific evidence validation failed: ' + process.stderr)
    result['scientific_evidence'] = json.loads(process.stdout)
    return result


def audit_receipt(root, record):
    root = Path(root).resolve()
    verify_run(root, record)
    if record['milestone_id'] != 'MF' or record['phase'] != 'report':
        raise ValueError('GF requires completed MF report evidence')
    folder = root / 'evidence/MF' / record['run_id']
    # Independently check table semantics before checking its immutable byte receipt.
    result = validate_release(root)
    for name in ARTIFACTS:
        if sha256_file(root / 'reports' / name) != sha256_file(folder / name):
            raise ValueError('Public report differs from its completed receipt: ' + name)
    if result['verdict'] != read_json(folder / 'summary.json')['report_status']:
        raise ValueError('Report verdict differs from its completed evidence')
    return result


def decide_report(root):
    records = load_inputs(Path(root), 'GF')
    if len(records) != 1:
        raise ValueError('GF requires exactly one selected complete report receipt')
    audit_receipt(root, records[0])
    return decide('GF', records, Path(root))


def run_report(root):
    root = Path(root).resolve()
    study, decision = _selected(root)
    scientific_path = study / 'decisions' / decision['gate_id'] / (decision['decision_id'] + '.json')
    dependencies = [*FILES, 'config/report_source.json',
                    str((study / 'STUDY_SNAPSHOT.json').relative_to(root)),
                    str(scientific_path.relative_to(root))]

    def producer(folder):
        state = render_report(root)
        result = validate_release(root)
        for name in ARTIFACTS:
            shutil.copyfile(root / 'reports' / name, folder / name)
        summary = {'report_status': state['status'], 'terminal_verdict': state['terminal_verdict'],
                   'source_decision': str(scientific_path.relative_to(root)),
                   'source_decision_sha256': sha256_file(scientific_path),
                   'validation': result, 'missing_cells': []}
        checks = [{'check_id': name, 'status': 'pass', 'reason': 'Verified against pinned scientific inputs and the complete report tables',
                   'scope': 'report', 'tolerance': None} for name in CHECKS]
        return checks, summary

    record = run_job(root, 'MF', 'report', dependencies, CHECKS, [], producer,
                     {'config/report_source.json': read_json(root / 'config/report_source.json')})
    if record['completion_status'] != 'complete':
        raise ValueError('Report producer failed; preserved receipt: ' + record['run_id'])
    folder = root / 'evidence/MF' / record['run_id']
    # A cached complete receipt can restore its public view without regenerating evidence.
    for name in ARTIFACTS:
        from .io import atomic_bytes
        atomic_bytes(root / 'reports' / name, (folder / name).read_bytes())
    return decide_report(root)
