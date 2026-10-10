"""Recompute frozen nominal scores after the identified recovery-write defect."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import sys


def load_validation_for_audit(dataset_module, manifest_path, access_journal):
    """Keep frozen validation logic, redirect only its access-event destination."""
    original_sink = dataset_module.append_event
    def audit_sink(_frozen_path, event):
        return original_sink(access_journal, event)
    dataset_module.append_event = audit_sink
    try:
        return dataset_module.load_corpus(manifest_path, phase='confirmatory',
                                          role='validation', purpose='selection')
    finally:
        dataset_module.append_event = original_sink


def assert_scores(observed, selected):
    for key in ('objective', 'one_step', 'four_step'):
        if (key not in observed or key not in selected
                or not math.isfinite(observed[key]) or not math.isfinite(selected[key])
                or not math.isclose(observed[key], selected[key], abs_tol=1e-12, rel_tol=1e-10)):
            raise ValueError('Selected tensors do not match their recorded nominal score: ' + key)
    for key in ('one_count', 'four_count'):
        if key not in observed or observed[key] != selected.get(key):
            raise ValueError('Selected nominal validation coverage differs: ' + key)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv); root = args.root.resolve()
    output = args.output.resolve()
    if output.is_relative_to(root):
        raise ValueError('Audit output and access journal must stay outside the frozen source context')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    import poc
    if Path(poc.__file__).resolve().parent.parent != root:
        raise ValueError('Nominal audit must import the exact frozen study source')
    from poc.io import read_json, sha256_file, atomic_json, utc_now
    jobs, missing = [], []
    for seed in range(40):
        for model in ('node_generic','power_correct'):
            cell = f'direct__A__shared__{model}__seed{seed}'
            path = root / 'config/rejection_checkpoint_freezes' / (cell + '.json')
            if not path.exists():
                missing.append(cell)
                continue
            freeze = read_json(path)
            if (freeze['phase'] != 'confirmatory' or len(freeze['training_manifests']) != 1
                    or len(freeze['checkpoint_hashes']) != 1):
                raise ValueError('Require one explicitly frozen confirmatory checkpoint per cell')
            jobs.append(dict(cell_id=cell,seed=seed,model=model,freeze=path,
                             manifest=root / freeze['training_manifests'][0],checkpoint=freeze['checkpoint_hashes'][0]))
    if args.dry_run:
        import json
        print(json.dumps(dict(execute=False,jobs=80,available_frozen_cells=len(jobs),
            missing_frozen_cells=missing,labels='validation only',training=False,
            fine_time_outcomes_opened=False,output=str(args.output))))
        return 0
    if missing:
        raise ValueError('Frozen cohort is incomplete; missing cells: ' + ', '.join(missing))
    import torch
    from poc.contracts import verify_run
    from poc.execution import cpu_defaults
    from poc import dataset
    from poc.models.common import Scales
    from poc.models.factory import build_model
    from poc.train import validate_nominal, canonical_tensor_hash
    cpu_defaults(0)
    checked, rows = {}, []

    def checkpoint(state, **extra):
        atomic_json(output,dict(schema_version=1,state=state,updated_utc=utc_now(),
            source_context=str(root),script_sha256=sha256_file(Path(__file__)),expected_cells=80,
            completed_cells=len(rows),cells=rows,labels='Nominal validation only',
            scope='Frozen selection-integrity check; no retraining, reselection or additional statistical look',**extra))
    try:
        for job in jobs:
            manifest = read_json(job['manifest']); folder = job['manifest'].parent
            verify_run(root,manifest,check_dependencies=False)
            for path,h in manifest['job_inputs'].get('dependencies',{}).items():
                if path not in checked:
                    checked[path] = sha256_file(root/path)
                if checked[path] != h:
                    raise ValueError('Changed nominal-fit dependency: ' + path)
            for name,h in manifest['job_inputs'].get('wheel_hashes',{}).items():
                path = 'wheelhouse/' + name
                if path not in checked: checked[path] = sha256_file(root/path)
                if checked[path] != h: raise ValueError('Changed nominal-fit wheel: ' + name)
            parameters = manifest['job_inputs']['parameters']
            if (manifest['phase'] != 'confirmatory' or any(parameters[k] != v for k,v in
                    [('model_id',job['model']),('seed',job['seed']),('actuator','direct'),
                     ('condition','A'),('loss_mode','shared')])):
                raise ValueError('Frozen nominal cell identity differs')
            validation_manifest = root / parameters['validation_manifest']
            if read_json(validation_manifest)['role'] != 'validation':
                raise ValueError('Nominal audit forbids non-validation labels')
            data = load_validation_for_audit(dataset, validation_manifest,
                                            output.with_suffix('.data_access.jsonl'))
            norm = read_json(folder/'normalization.json')
            scales = Scales(*(torch.tensor(norm[key],dtype=torch.float32)
                             for key in ('scale_q','scale_v','velocity_input_rms')))
            model = build_model(job['model'],'direct',scales,job['seed'])
            weights_path = folder/'weights.pt'
            if sha256_file(weights_path) != job['checkpoint']:
                raise ValueError('Selected archive differs from the prospective checkpoint freeze')
            weights = torch.load(weights_path,weights_only=True,map_location='cpu')
            before = canonical_tensor_hash(weights)
            if before != manifest['model_hashes']['canonical_tensor_sha256']:
                raise ValueError('Selected canonical tensors differ from their training receipt')
            model.load_state_dict(weights)
            selected = read_json(folder/'selection.json')
            observed = validate_nominal(model,data,scales,'shared')
            assert_scores(observed,selected)
            assert_scores(observed,read_json(folder/'summary.json')['validation'])
            if canonical_tensor_hash(model.state_dict()) != before:
                raise ValueError('Nominal validation mutated selected tensors')
            rows.append(dict(cell_id=job['cell_id'],manifest=str(job['manifest'].relative_to(root)),
                manifest_sha256=sha256_file(job['manifest']),freeze_sha256=sha256_file(job['freeze']),
                canonical_tensor_sha256=before,selected=selected,observed=observed,status='pass'))
            checkpoint('running');print(f"Selection integrity {len(rows)}/80: {job['cell_id']}",flush=True)
        checkpoint('complete',passed=True)
        return 0
    except Exception as error:
        checkpoint('invalid',passed=False,reason=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
