"""Nominal-only solver fit audit and verified closed-loop evidence consumption."""
from __future__ import annotations

import math
import os
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import torch

from .io import read_json, sha256_file, safe_path

REGISTRY = 'config/control_solver_validation_jobs.json'
SOURCES = ('poc/control_solver_validation.py', 'poc/dataset.py',
           'reports/audit_nominal_selections.py', 'reports/audit_control_nominal_selections.py')


class ValidationResolution(torch.nn.Module):
    def __init__(self, model, substeps):
        super().__init__()
        self.model, self.substeps, self.actuator = model, substeps, model.actuator

    def predict(self, state, command, h_s, history=None, power_mode=None):
        return self.model.predict(state, command, h_s, history=history, power_mode=power_mode,
                                  solver_substeps=self.substeps)


def audit_validation(model, arrays, scales, selected):
    from .models.ode import NeuralODE
    from .train import canonical_tensor_hash, validate_nominal
    if not isinstance(model, NeuralODE) or model.structured:
        raise ValueError('Nominal audit requires the declared generic NODE')
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError('Nominal solver audit uses original float32 training precision')
    before = canonical_tensor_hash(model.state_dict())
    rows, material = [], False
    for substeps in (1, 2, 4):
        observed = validate_nominal(ValidationResolution(model, substeps), arrays, scales, 'shared')
        if substeps == 1:
            for key in ('objective', 'one_step', 'four_step'):
                if (type(selected.get(key)) not in (int, float) or not math.isfinite(selected[key])
                        or not math.isclose(observed[key], selected[key], rel_tol=1e-10, abs_tol=1e-12)):
                    raise ValueError('Original selected validation score differs: ' + key)
            for key in ('one_count', 'four_count'):
                if observed[key] != selected.get(key):
                    raise ValueError('Original selected validation target count differs: ' + key)
            initial = observed
        elif any(observed[key] != initial[key] for key in ('one_count', 'four_count')):
            raise ValueError('Solver audit changed validation target exposure')
        absolute = abs(observed['objective'] - initial['objective'])
        relative = absolute / initial['objective'] if initial['objective'] > 0 else (0. if absolute == 0 else None)
        if substeps > 1 and (relative is None or relative > .005):
            material = True
        rows.append(dict(substeps=substeps, observed=observed, absolute_objective_change=absolute,
            relative_objective_change=relative))
    if canonical_tensor_hash(model.state_dict()) != before:
        raise ValueError('Nominal solver audit changed selected tensors')
    return dict(schema_version=1, state='complete', resolutions=rows, selected=selected,
        canonical_tensor_sha256=before, checkpoint_changed=False, training=False, reselection=False,
        material_nominal_fit_change=material, continuous_field_claim_requires_interpretation=material,
        interpretation='Original one-substep-trained discrete model retained; refined evaluation is diagnostic',
        scientific_decision='not_evaluated')


def validation_jobs(freeze):
    from reports.audit_control_nominal_selections import audit_jobs
    return [job for job in audit_jobs(freeze) if job['model_id'] == 'node_generic']


def _verify_registry(root, registry):
    from .control_confirmation import verify_freeze
    root = Path(root)
    for name, digest in registry['input_hashes'].items():
        if sha256_file(safe_path(root, name)) != digest:
            raise ValueError('Changed nominal solver validation input: ' + name)
    freeze = read_json(root / 'config/control_confirmation_cells.json')
    verify_freeze(root, read_json(root / 'config/control_confirmation_design.json'), freeze)
    expected = validation_jobs(freeze)
    if registry.get('state') != 'complete' or len(registry.get('jobs', [])) != 10:
        raise ValueError('Complete ten-fit solver validation registry required')
    for row, job in zip(registry['jobs'], expected):
        path = safe_path(root, row['path'])
        if row['cell_id'] != job['cell_id'] or sha256_file(path) != row['sha256'] or read_json(path) != job:
            raise ValueError('Changed frozen nominal solver validation fit')
    return registry


def prepare(root):
    from .io import atomic_json, utc_now
    from .control_jobs import SOURCES as CORE
    root = Path(root).resolve()
    target = root / REGISTRY
    if target.exists():
        return _verify_registry(root, read_json(target))
    freeze = read_json(root / 'config/control_confirmation_cells.json')
    definitions = validation_jobs(freeze)
    names = set(CORE) | set(SOURCES) | {'config/control_confirmation_cells.json',
        'config/control_confirmation_design.json', 'poc/control_confirmation.py',
        'poc/control_confirmation_batch.py', 'poc/control_utility.py', 'poc/statistics.py'}
    jobs = []
    for definition in definitions:
        fit = safe_path(root, definition['manifest'])
        record = read_json(fit)
        source = fit.parents[3]
        validation = safe_path(source, record['job_inputs']['parameters']['validation_manifest'])
        if read_json(validation)['role'] != 'validation':
            raise ValueError('Nominal solver audit permits validation labels only')
        path = root / 'config/control_solver_validation_jobs' / (definition['cell_id'] + '.json')
        if path.exists() and read_json(path) != definition:
            raise ValueError('Conflicting nominal solver validation request')
        atomic_json(path, definition)
        jobs.append(dict(cell_id=definition['cell_id'], path=str(path.relative_to(root)), sha256=sha256_file(path)))
        names.add(definition['manifest'])
        for name in ('weights.pt', 'normalization.json', 'selection.json', 'summary.json'):
            names.add(str((fit.parent / name).relative_to(root)))
        names.update(str(p.relative_to(root)) for p in (validation, validation.parent / 'trajectories.npz'))
    registry = dict(schema_version=1, state='complete', jobs=jobs, created_utc=utc_now(),
        input_hashes={name: sha256_file(root / name) for name in sorted(names)},
        expected_fits=10, scientific_decision='not_evaluated')
    _verify_registry(root, registry)
    atomic_json(target, registry)
    return registry


def _observe_fit(root, job, path):
    from .contracts import verify_run
    root, path = Path(root), Path(path)
    request_path = safe_path(root, job['path'])
    if sha256_file(request_path) != job['sha256']:
        raise ValueError('Changed nominal solver validation request')
    manifest = read_json(path)
    verify_run(root, manifest)
    result = read_json(path.parent / 'summary.json')
    if (manifest['milestone_id'] != 'M70' or manifest['phase'] != 'confirmatory'
            or result.get('state') != 'complete' or result.get('job') != read_json(request_path)
            or result.get('training') is not False or result.get('reselection') is not False
            or result.get('checkpoint_changed') is not False
            or [row['substeps'] for row in result.get('resolutions', [])] != [1, 2, 4]):
        raise ValueError('Incomplete or changed nominal solver validation observation')
    return dict(cell_id=job['cell_id'], manifest=str(path.relative_to(root)), manifest_sha256=sha256_file(path),
        summary=result, resources=read_json(path.parent / 'resources.json'))


def audit_fit(root, request_path):
    from .control_jobs import SOURCES as CORE
    from .contracts import verify_run
    from .execution import cpu_defaults, run_job
    from .control_models import load_selected
    from .models.common import Scales
    from . import dataset
    from .io import atomic_json
    from reports.audit_nominal_selections import load_validation_for_audit
    from reports.audit_control_nominal_selections import check_fit_identity
    root, request_path = Path(root).resolve(), Path(request_path).resolve()
    job = read_json(request_path)
    path = safe_path(root, job['manifest'])
    source, record = path.parents[3], read_json(path)
    verify_run(source, record)
    check_fit_identity(record, job)
    validation = safe_path(source, record['job_inputs']['parameters']['validation_manifest'])
    if read_json(validation)['role'] != 'validation':
        raise ValueError('Only nominal validation labels may be opened')
    cpu_defaults(job['seed'])
    key_path = str(request_path.relative_to(root))
    files = set(CORE) | set(SOURCES) | {key_path, job['manifest'], str(validation.relative_to(root)),
        str((validation.parent / 'trajectories.npz').relative_to(root))}
    files.update(str((path.parent / name).relative_to(root))
        for name in ('weights.pt', 'normalization.json', 'selection.json', 'summary.json'))
    checks = ('control.solver_validation_identity', 'control.solver_validation_targets', 'control.solver_validation_weights')
    def produce(output):
        model = load_selected(path, job['manifest_sha256'], job['checkpoint_sha256'],
            model_id='node_generic', actuator=job['actuator'], seed=job['seed']).float()
        norm = read_json(path.parent / 'normalization.json')
        scales = Scales(*(torch.tensor(norm[key], dtype=torch.float32)
            for key in ('scale_q', 'scale_v', 'velocity_input_rms')))
        arrays = load_validation_for_audit(dataset, validation, output / 'data_access.jsonl')
        result = audit_validation(model, arrays, scales, read_json(path.parent / 'selection.json'))
        if result['canonical_tensor_sha256'] != job['canonical_tensor_sha256']:
            raise ValueError('Solver validation changed the selected canonical tensors')
        result.update(job=job, validation_manifest_sha256=sha256_file(validation),
            validation_array_sha256=sha256_file(validation.parent / 'trajectories.npz'), missing_cells=[])
        atomic_json(output / 'request.json', job)
        checks_observed = [dict(check_id=key, status='pass', scope='nominal_validation_only', tolerance=None,
            reason='Same selected float32 tensors, nominal labels and target counts; no training/reselection') for key in checks]
        return checks_observed, result
    record = run_job(root, 'M70', 'confirmatory', sorted(files), checks, [], produce,
        config_keys={key_path: job}, argv=['-m', 'poc.control_solver_validation', '--fit', key_path])
    if record['completion_status'] != 'complete':
        raise RuntimeError('Nominal solver validation incomplete: ' + record['run_id'])
    manifest = root / 'evidence/M70' / record['run_id'] / 'manifest.json'
    return _observe_fit(root, dict(cell_id=job['cell_id'], path=key_path, sha256=sha256_file(request_path)), manifest)


def run_validation(root, registry, *, resume=False):
    import hashlib
    import subprocess
    import sys
    from .io import atomic_json, append_event, job_lock, utc_now
    root = Path(root).resolve()
    _verify_registry(root, registry)
    identity = sha256_file(root / REGISTRY)
    target = root / 'reports/control_solver_validation.json'
    jobs = {job['cell_id']: job for job in registry['jobs']}
    with job_lock(hashlib.sha256(b'registered nominal NODE solver validation').hexdigest(), root):
        reused = {}
        if target.exists():
            previous = read_json(target)
            if not resume or previous['registry_sha256'] != identity:
                raise ValueError('Nominal solver validation requires compatible --resume')
            for entry in previous['fits']:
                cell = entry['cell_id']
                if cell not in jobs or cell in reused:
                    raise ValueError('Unknown or duplicate saved nominal solver fit')
                checked = _observe_fit(root, jobs[cell], safe_path(root, entry['manifest']))
                if checked != entry:
                    raise ValueError('Saved nominal solver fit differs from completed evidence')
                reused[cell] = checked
            if previous['state'] == 'complete':
                if len(reused) != 10 or previous['completed_fits'] != 10:
                    raise ValueError('Incomplete completed nominal solver validation')
                return previous
        fits = []
        def save(state, **extra):
            report = dict(schema_version=1, state=state, registry_sha256=identity, updated_utc=utc_now(),
                expected_fits=10, completed_fits=len(fits), fits=fits, scientific_decision='not_evaluated', **extra)
            atomic_json(target, report)
            return report
        try:
            for index, job in enumerate(registry['jobs']):
                for name, digest in registry['input_hashes'].items():
                    if sha256_file(safe_path(root, name)) != digest:
                        raise ValueError('Changed nominal solver validation source: ' + name)
                if sha256_file(root / REGISTRY) != identity:
                    raise ValueError('Changed nominal solver validation registry')
                print(f"Nominal solver validation {index + 1}/10: {job['cell_id']}", flush=True)
                entry = reused.get(job['cell_id'])
                if entry is None:
                    process = subprocess.run([sys.executable, '-m', 'poc.control_solver_validation', '--fit', job['path']],
                        cwd=root, capture_output=True, text=True)
                    append_event(root / 'reports/control_solver_validation_attempts.jsonl',
                        dict(cell_id=job['cell_id'], utc=utc_now(), returncode=process.returncode,
                            stdout=process.stdout, stderr=process.stderr))
                    if process.returncode:
                        raise RuntimeError('Nominal solver validation interrupted: ' + process.stdout + '\n' + process.stderr)
                    entry = _observe_fit(root, job, Path(process.stdout.strip().splitlines()[-1]))
                fits.append(entry)
                save('running')
            material = any(entry['summary']['material_nominal_fit_change'] for entry in fits)
            return save('complete', n_fits=10, material_nominal_fit_change=material,
                continuous_field_claim_requires_interpretation=material)
        except BaseException as error:
            save('interrupted', reason=f'{type(error).__name__}: {error}')
            raise


def load_closed_loop_evidence(root, evidence):
    """Attach only fully reverified actual solver and nominal-fit observations."""
    import copy
    from .control_solver_continuation import REGISTRY as SOLVER_REGISTRY, verify_registry, observe_job, summarize_solver
    root = Path(root).resolve()
    evidence = copy.deepcopy(evidence)
    required = sorted(evidence['solver_cells_required'])
    if not required:
        return evidence
    paths = [SOLVER_REGISTRY, REGISTRY, 'reports/control_solver_continuation_batch.json',
        'reports/control_solver_continuation_result.json', 'reports/control_solver_validation.json']
    if not all((root / name).exists() for name in paths):
        return evidence
    registry = verify_registry(root, read_json(root / SOLVER_REGISTRY))
    if registry['parent_cells'] != required:
        raise ValueError('Closed-loop solver evidence differs from the actual required cells')
    batch = read_json(root / 'reports/control_solver_continuation_batch.json')
    if (batch.get('state') != 'complete' or batch.get('expected_episodes') != 100
            or batch.get('completed_episodes') != 100 or len(batch.get('episodes', [])) != 100
            or batch['registry_sha256'] != sha256_file(root / SOLVER_REGISTRY)):
        return evidence
    jobs = {job['cell_id']: job for job in registry['jobs']}
    checked, seen = [], set()
    for entry in batch['episodes']:
        cell = entry['cell_id']
        if cell not in jobs or cell in seen:
            raise ValueError('Missing or duplicated closed-loop solver observation')
        actual = observe_job(root, jobs[cell], safe_path(root, entry['manifest']))
        if actual != entry:
            raise ValueError('Saved solver receipt differs from its complete raw observation')
        checked.append(actual)
        seen.add(cell)
    if seen != set(jobs):
        raise ValueError('Incomplete solver resolution coverage')
    summary = summarize_solver(read_json(root / 'reports/control_confirmation_batch.json'),
        [entry['summary'] for entry in checked], required, evidence['design'])
    summary.update(registry_sha256=sha256_file(root / SOLVER_REGISTRY),
        batch_sha256=sha256_file(root / 'reports/control_solver_continuation_batch.json'))
    if summary != read_json(root / 'reports/control_solver_continuation_result.json'):
        raise ValueError('Solver convergence summary differs from raw observed costs')
    validation_registry = _verify_registry(root, read_json(root / REGISTRY))
    validation = read_json(root / 'reports/control_solver_validation.json')
    if (validation.get('state') != 'complete' or validation.get('expected_fits') != 10
            or validation.get('completed_fits') != 10 or len(validation.get('fits', [])) != 10
            or validation['registry_sha256'] != sha256_file(root / REGISTRY)):
        return evidence
    validation_jobs_by_id = {job['cell_id']: job for job in validation_registry['jobs']}
    fit_ids = set()
    for entry in validation['fits']:
        cell = entry['cell_id']
        if cell not in validation_jobs_by_id or cell in fit_ids:
            raise ValueError('Duplicated or unknown nominal solver fit')
        if _observe_fit(root, validation_jobs_by_id[cell], safe_path(root, entry['manifest'])) != entry:
            raise ValueError('Changed completed nominal solver validation')
        fit_ids.add(cell)
    if fit_ids != set(validation_jobs_by_id):
        raise ValueError('Incomplete nominal solver validation coverage')
    material = any(entry['summary']['material_nominal_fit_change'] for entry in validation['fits'])
    if (validation.get('n_fits') != 10 or validation.get('material_nominal_fit_change') != material
            or validation.get('continuous_field_claim_requires_interpretation') != material):
        raise ValueError('Nominal solver interpretation differs from observed validation scores')
    summary.update(evidence_verified=True, validation=dict(state='complete', n_fits=10,
        material_nominal_fit_change=material, continuous_field_claim_requires_interpretation=material))
    evidence['solver_resolution'] = summary
    names = paths + ['poc/control_solver_continuation.py', 'poc/control_solver_validation.py']
    evidence['input_hashes'].update({name: sha256_file(root / name) for name in names})
    evidence['resources']['solver_closed_loop'] = [entry['resources'] for entry in checked]
    evidence['resources']['solver_validation'] = [entry['resources'] for entry in validation['fits']]
    return evidence


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--fit', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.dry_run:
        print(dict(execute=False, expected_fits=10, substeps=[1, 2, 4], labels='nominal validation only'))
        return 0
    if args.fit:
        path = args.fit if args.fit.is_absolute() else root / args.fit
        result = audit_fit(root, path)
        print(root / result['manifest'], flush=True)
        return 0
    registry = prepare(root)
    if args.prepare:
        print(root / REGISTRY)
        return 0
    result = run_validation(root, registry, resume=args.resume)
    print({key: result[key] for key in ('state', 'completed_fits', 'material_nominal_fit_change')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
