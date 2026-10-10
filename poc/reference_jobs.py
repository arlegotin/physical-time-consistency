"""Recorded physical-reference fits; archived robot labels remain immutable."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

from .io import atomic_bytes, atomic_json, canonical_json, read_json, sha256_file, utc_now


def tensor_hash(weights):
    digest = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(canonical_json({'name': name, 'dtype': str(value.dtype), 'shape': list(value.shape)}))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def save_tensors(path, values):
    import torch
    buffer = io.BytesIO()
    torch.save(values, buffer)
    atomic_bytes(path, buffer.getvalue())


def _csv(path, rows):
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, text.getvalue().encode())


def produce_reference(folder, train_manifest, validation_manifest, normalization_path,
                      *, model_id, epochs=100, batch_size=256, phase='exploratory', resume_folder=None):
    import numpy as np
    import torch
    from .dataset import load_corpus
    from .models.common import Scales
    from .physics_fit import PhysicsFit, AccelRidge, fit_ridge
    from .physics_training import endpoint_loss, optimize_endpoints

    train_meta, valid_meta = read_json(train_manifest), read_json(validation_manifest)
    if train_meta['role'] != 'train' or valid_meta['role'] != 'validation':
        raise ValueError('Physical fitting requires distinct train/validation roles')
    if any(train_meta[key] != valid_meta[key] for key in ('actuator', 'condition', 'physics_step_ns')):
        raise ValueError('Incompatible physical-reference corpora')
    normalization = read_json(normalization_path)
    if normalization['training_manifest_sha256'] != sha256_file(train_manifest):
        raise ValueError('Physical-reference normalization must derive from these training labels')
    if model_id not in {'physics_fit', 'accel_ridge'}:
        raise ValueError('Unknown physical reference')
    scales = Scales(*(torch.tensor(normalization[key], dtype=torch.float32)
                     for key in ('scale_q', 'scale_v', 'velocity_input_rms')))
    access = folder / 'data_access.jsonl'
    def flattened(manifest, role, purpose):
        arrays = load_corpus(manifest, phase=phase, role=role, purpose=purpose, audit_path=access)
        dimension = arrays['state'].shape[-1]
        return {'state': arrays['state'][:, :-1].reshape(-1, dimension),
                'target': arrays['state'][:, 1:].reshape(-1, dimension),
                'command': arrays['command'].reshape(-1, 2),
                'h_s': arrays['duration_ns'].reshape(-1, 1) / 1e9}
    actuator = train_meta['actuator']
    if model_id == 'physics_fit':
        train = flattened(train_manifest, 'train', 'training')
        valid = flattened(validation_manifest, 'validation', 'selection')
        model = PhysicsFit(torch.tensor([.10, .02, .02, .05, .05]), actuator, scales)
        identity = {'train_sha256': sha256_file(train_manifest),
                    'validation_sha256': sha256_file(validation_manifest),
                    'normalization_sha256': sha256_file(normalization_path),
                    'epochs': epochs, 'batch_size': batch_size, 'seed': 8201}
        resume = None
        if resume_folder is not None:
            marker = read_json(resume_folder / 'resume.complete.json')
            if marker['inputs'] != identity or marker['checkpoint_sha256'] != sha256_file(resume_folder / 'resume.pt'):
                raise ValueError('Incompatible or corrupted physical-reference resume checkpoint')
            resume = torch.load(resume_folder / 'resume.pt', weights_only=True, map_location='cpu')
        def checkpoint(state):
            save_tensors(folder / 'resume.pt', state)
            atomic_json(folder / 'resume.complete.json', {'inputs': identity,
                        'checkpoint_sha256': sha256_file(folder / 'resume.pt'), 'epoch': state['epoch'], 'utc': utc_now()})
        fitted = optimize_endpoints(model, train, valid, epochs=epochs, batch_size=batch_size,
                                   resume=resume, on_epoch=checkpoint)
        _csv(folder / 'training.csv', fitted['training_history'])
        _csv(folder / 'validation.csv', fitted['validation_history'])
        selection = {key: value for key, value in fitted.items()
                     if key not in {'best_state', 'last_state', 'optimizer_state', 'rng_state',
                                    'training_history', 'validation_history'}}
        selection['physical_parameters'] = model.physical_parameters().detach().tolist()
        updates = fitted['optimizer_updates']
    else:
        fitted = fit_ridge(train_manifest, audit_path=access)
        model = AccelRidge(np.array(fitted['coefficients']), actuator, scales)
        valid = flattened(validation_manifest, 'validation', 'selection')
        with torch.no_grad():
            score = float(endpoint_loss(model, valid, .00125))
        selection = {**fitted, 'best_validation_loss': score, 'evaluation_step_s': .00125,
                     'selection': 'single prespecified ridge solve; no fine-time selection'}
        updates = 0
    selection['target_components'] = ['q', 'v']
    selection['training_manifest_sha256'] = sha256_file(train_manifest)
    selection['validation_manifest_sha256'] = sha256_file(validation_manifest)
    atomic_json(folder / 'selection.json', selection)
    save_tensors(folder / 'weights.pt', model.state_dict())
    atomic_json(folder / 'normalization.json', normalization)
    return {'model_id': model_id, 'actuator': actuator, 'condition': train_meta['condition'],
            'selection': selection, 'training_updates': updates,
            'model_hashes': {'canonical_tensor_sha256': tensor_hash(model.state_dict()),
                             'archive_sha256': sha256_file(folder / 'weights.pt')},
            'missing_cells': [], 'scope': 'reference endpoint fit; robot MPC qualification not run'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    root, request = args.root.resolve(), read_json(args.request)
    from .execution import cpu_defaults, run_job
    from .io import safe_path
    cpu_defaults(8201)
    required = {'model_id', 'train_manifest', 'validation_manifest', 'normalization', 'epochs',
                'batch_size', 'resource_projection'}
    if set(request) != required or request['epochs'] != 100 or request['batch_size'] != 256:
        raise ValueError('Use the exact preregistered physical-reference request and initial training budget')
    paths = [safe_path(root, request[key]) for key in ('train_manifest', 'validation_manifest', 'normalization')]
    files = ['poc/__init__.py', 'poc/reference_jobs.py', 'poc/io.py', 'poc/contracts.py', 'poc/gates.py',
             'poc/execution.py', 'poc/dataset.py', 'poc/analytic.py', 'poc/physics_fit.py',
             'poc/physics_training.py', 'poc/models/__init__.py', 'poc/models/common.py',
             'poc/models/mlp.py', 'poc/models/power.py']
    files += [str(path.relative_to(root)) for path in paths]
    files += [str((path.parent / 'trajectories.npz').relative_to(root)) for path in paths[:2]]
    files += ['config/revisions/owner-objective-2026-10-01.json']
    key_path = str(args.request.resolve().relative_to(root))
    files += [key_path]
    expected = ('reference.endpoint_fit', 'reference.provenance')
    def producer(folder):
        partial = None
        # Resume only exactly matching request/source identities, never an arbitrary latest file.
        current_dependencies = {name: sha256_file(root / name) for name in files}
        matches = []
        for candidate in (root / 'evidence/M40').glob('*/resume.complete.json'):
            prior_inputs = candidate.parent / 'reference_inputs.json'
            if (not (candidate.parent / 'COMPLETED').exists() and prior_inputs.exists()
                    and not (candidate.parent / 'RESUMED_BY.json').exists()
                    and read_json(prior_inputs) == current_dependencies):
                matches.append(candidate.parent)
        if len(matches) > 1:
            raise ValueError('Conflicting physical-reference interrupted attempts')
        if matches and not args.resume:
            raise ValueError('Compatible interrupted physical fit exists; use --resume')
        if args.resume:
            partial = matches[0] if matches else None
        atomic_json(folder / 'reference_inputs.json', {name: sha256_file(root / name) for name in files})
        summary = produce_reference(folder, *paths, model_id=request['model_id'],
                                    epochs=request['epochs'], batch_size=request['batch_size'], resume_folder=partial)
        summary['resource_projection'] = request['resource_projection']
        summary['data_hashes'] = {str(path.relative_to(root)): sha256_file(path) for path in paths}
        # A metadata-only replacement may retire a stale receipt only with identical selected tensors.
        from .contracts import verify_run
        replacements = []
        for manifest_path in (root / 'evidence/M40').glob('*/manifest.json'):
            old = read_json(manifest_path)
            if (old['completion_status'] == 'complete' and old['phase'] == 'exploratory'
                    and old['expected_cells'] == list(expected)
                    and old['job_inputs'].get('configuration_keys') == {key_path: request}):
                differences = {name for name, digest in old['job_inputs']['dependencies'].items()
                               if sha256_file(root / name) != digest}
                if differences and differences <= {'poc/execution.py', 'poc/reference_jobs.py'}:
                    verify_run(root, old, check_dependencies=False)
                    if old['model_hashes']['canonical_tensor_sha256'] != summary['model_hashes']['canonical_tensor_sha256']:
                        raise ValueError('Metadata-only physical refit changed selected tensors; diagnose reproducibility')
                    replacements.append(old['run_id'])
        if replacements:
            summary['supersedes'] = replacements
            summary['supersession_reason'] = 'Metadata envelope repaired; selected canonical tensors identical and original fit artifacts preserved'
        imported = {str(Path(module.__file__).resolve().relative_to(root))
                    for module in list(sys.modules.values()) if getattr(module, '__file__', None)
                    and str(module.__file__).endswith('.py')
                    and Path(module.__file__).resolve().is_relative_to(root / 'poc')}
        if imported - set(files):
            raise ValueError('Undeclared physical-reference imports: ' + str(sorted(imported - set(files))))
        if any(sha256_file(root / path) != digest for path, digest in current_dependencies.items()):
            raise ValueError('Physical-reference inputs changed during fitting')
        checks = [{'check_id': name, 'status': 'pass', 'reason': 'Endpoint-only fit and qualified corpus hashes verified',
                   'scope': request['model_id'], 'tolerance': None} for name in expected]
        return checks, summary
    record = run_job(root, 'M40', 'exploratory', files, expected, [], producer,
                     config_keys={key_path: request},
                     argv=[sys.executable, '-m', 'poc.reference_jobs', key_path, '--root', str(root)]
                          + (['--resume'] if args.resume else []))
    print(str(root / 'evidence/M40' / record['run_id'] / 'manifest.json'), flush=True)
    return 0 if record['completion_status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
