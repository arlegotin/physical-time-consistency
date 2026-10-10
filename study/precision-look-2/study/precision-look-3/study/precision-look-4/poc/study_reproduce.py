"""Restore/verify a pinned study and reaggregate its actual decisive gate."""
from pathlib import Path
import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys

from .io import atomic_bytes, atomic_json, read_json, sha256_file


def validate_runtime_versions(expected, actual):
    if any(actual.get(name) != version for name, version in expected.items()):
        raise ValueError('Replay package version differs from the pinned study environment')


def reproduction_plan(gate, rerun=False, authorization=None, locks=()):
    if gate not in {'G00', 'G10', 'G20', 'GR'}:
        raise ValueError('Unsupported or unexecuted reproduction gate')
    plan = [['validate', '--scope', 'protocol'], ['status', '--verify']]
    if rerun:
        if not authorization or authorization.get('local_compute_authorized') is not True or authorization.get('paid_services_authorized') is not False:
            raise ValueError('Experiment execution requires current local authorization and no paid services')
        if locks:
            raise ValueError('Experiment execution cannot duplicate an active job lock')
        plan.append(['run', {'G00': 'M00', 'G10': 'M10', 'G20': 'M20', 'GR': 'MR'}[gate], '--resume'])
    return [*plan, ['decide', gate]]


def write_reproduction(output: Path, study: Path, gate):
    output, study = Path(output).resolve(), Path(study).resolve()
    if not study.is_relative_to(output):
        raise ValueError('Pinned study must be inside the report project')
    from .gates import current_decision
    decision = current_decision(study)
    if not decision or decision['gate_id'] != gate:
        raise ValueError('Reproduction must name the actually executed decisive gate')
    reproduction_plan(gate)
    request = {'protocol_version': 'PTCO-1.0', 'study_relative': str(study.relative_to(output)),
        'gate_id': gate, 'decision_id': decision['decision_id'], 'outcome': decision['outcome'],
        'lock_sha256': sha256_file(study / 'locks/requirements.lock'),
        'snapshot_sha256': sha256_file(study / 'STUDY_SNAPSHOT.json'),
        'execution_behavior': 'Default verifies and reaggregates. Explicit execution resumes frozen jobs; complete compatible jobs remain cached. No fresh-training replication is claimed.'}
    atomic_json(output / 'reports/reproduction_request.json', request)
    script = '''#!/usr/bin/env bash
set -euo pipefail
PTCO_PROJECT=$(cd -- "$(dirname -- "$0")/.." && pwd)
PTCO_PYTHON=${PTCO_PYTHON:-"$PTCO_PROJECT/.reproduction-venv/bin/python"}
if [[ ! -x "$PTCO_PYTHON" ]]; then
  python3.11 -m venv "$PTCO_PROJECT/.reproduction-venv"
  PTCO_PYTHON="$PTCO_PROJECT/.reproduction-venv/bin/python"
fi
cd -- "$PTCO_PROJECT"
"$PTCO_PYTHON" -m poc.study_reproduce --project "$PTCO_PROJECT" "$@"
'''
    atomic_bytes(output / 'reports/reproduce.sh', script.encode('utf-8'))
    return request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--rerun-experiments', action='store_true')
    args = parser.parse_args(argv)
    project = args.project.resolve()
    from .io import safe_path
    from .study_archive import verify_snapshot
    request = read_json(project / 'reports/reproduction_request.json')
    study = safe_path(project, request['study_relative'])
    if sys.version_info[:2] != (3, 11):
        raise ValueError('Reproduction requires the authorized CPython 3.11 interpreter')
    if sha256_file(study / 'STUDY_SNAPSHOT.json') != request['snapshot_sha256'] or sha256_file(study / 'locks/requirements.lock') != request['lock_sha256']:
        raise ValueError('Changed archived input/lock hash')
    verify_snapshot(study)
    plan = reproduction_plan(request['gate_id'], args.rerun_experiments,
        read_json(project / 'config/authorization.json'), list((project / 'locks/active').glob('*.json')))
    # A fresh interpreter is restored from the actual complete, hash-pinned wheel set.
    environment = read_json(study / 'locks/environment.json')
    imports_available = all(importlib.util.find_spec(name) is not None for name in ('numpy', 'scipy', 'torch', 'mujoco'))
    if not imports_available:
        if sys.prefix == sys.base_prefix:
            raise ValueError('Offline installation requires a project virtual environment; global software is never changed')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', str(study / 'wheelhouse'),
            '--require-hashes', '-r', str(study / 'locks/requirements.lock')], check=True)
    runtime_versions = {name: importlib.metadata.version(name) for name in environment['versions']}
    validate_runtime_versions(environment['versions'], runtime_versions)
    results = []
    env = dict(os.environ)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[name] = '1'
    env.pop('PYTHONPATH', None)
    for command in plan:
        process = subprocess.run([sys.executable, '-m', 'poc', '--root', str(study), *command],
                                 cwd=study, env=env, capture_output=True, text=True)
        try:
            value = json.loads(process.stdout)
        except ValueError:
            raise RuntimeError(f'Reproduction command failed: {command}; {process.stderr}')
        accepted_negative = (command[:1] in [['decide'], ['run']] and value.get('outcome') == 'NO-GO'
            and value.get('validity_status') == 'valid' and value.get('decision_id') == request['decision_id'])
        if process.returncode != 0 and not accepted_negative:
            raise RuntimeError(f'Reproduction command failed: {command}; {process.stderr or process.stdout}')
        results.append({'argv': command, 'returncode': process.returncode, 'result': value, 'stderr': process.stderr})
    final = results[-1]['result']
    if final.get('decision_id') != request['decision_id'] or final.get('outcome') != request['outcome']:
        raise ValueError('Fresh-process reaggregation changed the decisive decision')
    atomic_json(project / 'reports/reproduction_verification.json', {'status': 'verified',
        'decision_id': final['decision_id'], 'fresh_training': False, 'commands': results,
        'python': sys.version, 'executable': sys.executable, 'versions': runtime_versions})
    print(json.dumps({'status': 'verified', 'decision_id': final['decision_id'], 'fresh_training': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
