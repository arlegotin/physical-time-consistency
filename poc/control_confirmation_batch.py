"""Execute the fixed control registry serially; never select or drop outcomes."""
from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

from .contracts import verify_run
from .io import append_event, atomic_json, read_json, safe_path, sha256_file, utc_now


def episode_receipt(root: Path, job: dict, manifest: Path) -> dict:
    manifest = Path(manifest).resolve()
    if not manifest.is_relative_to(root.resolve()):
        raise ValueError('Episode manifest escapes the project')
    record = read_json(manifest)
    verify_run(root, record)
    if record['milestone_id'] != 'M70' or record['phase'] != 'confirmatory' or record['completion_status'] != 'complete':
        raise ValueError('Only completed confirmatory control episodes can enter this batch')
    request = read_json(safe_path(root, job['path']))
    if read_json(manifest.parent/'request.json') != request:
        raise ValueError('Completed episode request differs from its frozen cell')
    summary = read_json(manifest.parent/'summary.json')
    for key in ('phase', 'condition', 'model_id', 'actuator', 'seed', 'episode_id', 'power_mode',
                'model_partition_ns', 'actuator_interval_ns', 'physics_step_ns'):
        if summary[key] != request[key]:
            raise ValueError('Completed episode identity differs: '+key)
    if (summary.get('complete') is not True or summary['checkpoint_hash'] != request['checkpoint_sha256']
            or not math.isfinite(summary['realized_cost']) or summary['realized_cost'] < 0
            or type(summary['fallback_count']) is not int or summary['fallback_count'] < 0):
        raise ValueError('Invalid complete episode cost, checkpoint or failure accounting')
    return dict(cell_id=job['cell_id'], request_sha256=job['sha256'],
        manifest=str(manifest.relative_to(root)), manifest_sha256=sha256_file(manifest),
        summary=summary, resources=read_json(manifest.parent/'resources.json'))


def run_jobs(root: Path, jobs: list[dict], registry_hash: str, *, resume: bool,
             frozen_sources: dict | None = None, batch_kind='primary') -> dict:
    """Registry validation and the batch lock belong to the caller; child jobs keep their own locks."""
    root = Path(root).resolve()
    if batch_kind not in ('primary','reference'):
        raise ValueError('Unknown fixed control batch')
    prefix='control_confirmation' if batch_kind=='primary' else 'control_reference'
    receipt_path = root/f'reports/{prefix}_batch.json'
    if len({j['cell_id'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicated registered episode cell')
    for job in jobs:
        if sha256_file(safe_path(root, job['path'])) != job['sha256']:
            raise ValueError('Changed frozen request hash before execution')
    prior = read_json(receipt_path) if receipt_path.exists() else None
    if prior is not None and not resume:
        raise ValueError('Existing batch receipt requires --resume')
    reused = {}
    if prior is not None:
        if prior['registry_sha256'] != registry_hash:
            raise ValueError('Changed confirmation registry is not a resumption')
        by_cell = {j['cell_id']: j for j in jobs}
        for previous in prior['episodes']:
            cell = previous['cell_id']
            if cell in reused or cell not in by_cell:
                raise ValueError('Unexpected or duplicated completed receipt cell')
            path = safe_path(root, previous['manifest'])
            if sha256_file(path) != previous['manifest_sha256']:
                raise ValueError('Changed completed episode manifest')
            checked = episode_receipt(root, by_cell[cell], path)
            if any(previous.get(k) != v for k, v in checked.items()):
                raise ValueError('Changed completed receipt observation')
            reused[cell] = checked
        if prior.get('state') == 'complete':
            if (prior.get('expected_episodes') != len(jobs) or prior.get('completed_episodes') != len(jobs)
                    or len(reused) != len(jobs)):
                raise ValueError('Completed batch receipt has incomplete cell coverage')
            for name, expected in (frozen_sources or {}).items():
                if sha256_file(safe_path(root, name)) != expected:
                    raise ValueError('Changed frozen batch evaluator: '+name)
            # Reverification is not new statistical evidence: preserve the exact receipt.
            return prior
    episodes = []

    def save(state, **extra):
        result = dict(schema_version=1, state=state, registry_sha256=registry_hash,
            updated_utc=utc_now(), expected_episodes=len(jobs), completed_episodes=len(episodes),
            episodes=episodes, scientific_decision='not_evaluated', **extra)
        atomic_json(receipt_path, result)
        return result

    try:
        for index, job in enumerate(jobs):
            for name, expected in (frozen_sources or {}).items():
                if sha256_file(safe_path(root, name)) != expected:
                    raise ValueError('Changed frozen batch evaluator: '+name)
            if sha256_file(safe_path(root, job['path'])) != job['sha256']:
                raise ValueError('Changed frozen request hash during execution')
            print(f"Control {batch_kind} {index+1}/{len(jobs)}: {job['cell_id']}", flush=True)
            if job['cell_id'] in reused:
                episode = dict(reused[job['cell_id']], reused=True)
            else:
                run = subprocess.run([sys.executable, '-m', 'poc.control_jobs', job['path']],
                                     cwd=root, capture_output=True, text=True)
                append_event(root/f'reports/{prefix}_attempts.jsonl',
                    dict(cell_id=job['cell_id'], request_sha256=job['sha256'], utc=utc_now(),
                         returncode=run.returncode, stdout=run.stdout, stderr=run.stderr))
                if run.returncode:
                    raise RuntimeError('Control episode interrupted: '+run.stdout+'\n'+run.stderr)
                path = Path(run.stdout.strip().splitlines()[-1])
                episode = dict(episode_receipt(root, job, path), reused=False)
            episodes.append(episode)
            save('running')
        return save('complete')
    except BaseException as error:
        save('interrupted', reason=f'{type(error).__name__}: {error}')
        raise
