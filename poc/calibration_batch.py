"""Execute only the frozen 96 model-independent robot calibration episodes."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

from .io import atomic_json, read_json, safe_path, sha256_file, utc_now


def validate_registry(root, registry):
    from .control_jobs import validate_request
    if (registry['schema_version'] != 1 or registry['phase'] != 'calibration'
            or sha256_file(root / 'config/control_calibration.json') != registry['calibration_config_sha256']
            or sha256_file(root / 'reports/robot_calibration_representative.json') != registry['measurement_report_sha256']):
        raise ValueError('Changed prospective calibration or representative resource measurement')
    expected = {(a, e, m) for a in ('direct', 'lagged') for e in range(16)
                for m in ('zero_command', 'physics_fit', 'sim_oracle')}
    identities = [(j['actuator'], j['episode_id'], j['model_id']) for j in registry['jobs']]
    if len(identities) != 96 or set(identities) != expected:
        raise ValueError('Incomplete or duplicated prospective calibration jobs')
    for job in registry['jobs']:
        path = safe_path(root, job['request'])
        if sha256_file(path) != job['sha256']:
            raise ValueError('Prospective control request hash changed')
        request = validate_request(read_json(path))
        if (any(request[k] != job[k] for k in ('actuator', 'episode_id', 'model_id'))
                or request['phase'] != 'calibration' or request['seed'] != 9000):
            raise ValueError('Registry and control task identity disagree')
    return registry['jobs']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    registry_path = root / 'config/control_calibration_jobs.json'
    registry_hash = sha256_file(registry_path)
    jobs = validate_registry(root, read_json(registry_path))
    if args.dry_run:
        print({'execute': False, 'fixed_episodes': len(jobs), 'reference_bank_audits': 32,
               'existing_episodes_reused_by_exact_job_key': True,
               'resource_estimate': read_json(root / 'reports/robot_calibration_representative.json')})
        return 0
    from .contracts import verify_run
    from .robot_calibration import summarize
    rows, episodes, audits = [], [], []
    receipt = root / 'reports/robot_calibration_batch.json'

    def invoke(module, argument):
        output = subprocess.run([sys.executable, '-m', module, argument], cwd=root,
                                capture_output=True, text=True)
        if output.returncode:
            raise RuntimeError(f'{module} failed: {output.stdout}\n{output.stderr}')
        manifest = Path(output.stdout.strip().splitlines()[-1])
        record = read_json(manifest); verify_run(root, record)
        return manifest, read_json(manifest.parent / 'summary.json')

    def checkpoint(state, **extra):
        atomic_json(receipt, {'schema_version': 1, 'state': state, 'updated_utc': utc_now(),
            'registry_sha256': registry_hash, 'episode_manifests': episodes, 'bank_audits': audits,
            'completed_episodes': len(episodes), 'expected_episodes': 96, 'expected_bank_audits': 32, **extra})

    try:
        for index, job in enumerate(jobs):
            if sha256_file(registry_path) != registry_hash:
                raise ValueError('Prospective registry changed during execution')
            print(f"Robot calibration {index + 1}/96: {job['actuator']} episode{job['episode_id']} {job['model_id']}", flush=True)
            manifest, summary = invoke('poc.control_jobs', job['request'])
            rows.append(summary)
            episodes.append({'path': str(manifest.relative_to(root)), 'sha256': sha256_file(manifest)})
            checkpoint('running_episodes')
            if job['model_id'] == 'physics_fit':
                audit, result = invoke('poc.reference_bank', str(manifest.relative_to(root)))
                audits.append({'path': str(audit.relative_to(root)), 'sha256': sha256_file(audit), **result})
                checkpoint('running_episodes')
        headroom = summarize(rows)
        qualified = headroom['qualified'] and len(audits) == 32 and all(a['ranking']['qualified'] for a in audits)
        checkpoint('complete', headroom=headroom, qualified=qualified,
                   scope='Model-independent qualification only; learned control remains untested')
        print(receipt, flush=True)
        return 0
    except Exception as error:
        checkpoint('interrupted', reason=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
