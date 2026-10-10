"""Run one frozen pilot episode using the validated original control machinery."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'


def run_episode(root: Path, request_path: Path):
    from .control_jobs import SOURCES
    from .control_models import SimOracle, load_selected
    from .control_episode import execute_episode
    from .control_artifacts import save_episode
    from .efficient_power import EfficientPowerHead
    from .efficiency_pilot import FREEZE, validate_request, verify_hashes
    from .envs import Plant
    from .execution import cpu_defaults, run_job
    from .io import atomic_json, read_json, safe_path, sha256_file
    from .planner_qualification import verify_frozen

    root, request_path = root.resolve(), request_path.resolve()
    freeze = read_json(root/FREEZE)
    request = validate_request(read_json(request_path), freeze)
    if request['freeze_sha256'] != sha256_file(root/FREEZE):
        raise ValueError('Episode checkpoint/source freeze changed')
    verify_hashes(root, {**freeze['source_hashes'], **freeze['input_hashes']})
    _, qualification_files = verify_frozen(root)
    cpu_defaults(request['seed'])
    relative = str(request_path.relative_to(root))
    files = list(dict.fromkeys([*SOURCES, 'poc/efficient_power.py', 'poc/efficiency_episode.py',
        'poc/efficiency_pilot.py', FREEZE, relative, *freeze['input_hashes'], *qualification_files]))
    if request['model_manifest']:
        manifest = safe_path(root, request['model_manifest'])
        files += [request['model_manifest'], str((manifest.parent/'weights.pt').relative_to(root)),
                  str((manifest.parent/'normalization.json').relative_to(root))]
    hashes = {name: sha256_file(root/name) for name in files}
    expected = ('control.episode', 'control.clocks', 'control.provenance')

    def producer(folder):
        kind = request['model_id']
        if kind == 'sim_oracle':
            model = SimOracle(request['actuator'], request['physics_step_ns'], root/'assets')
        else:
            model = load_selected(safe_path(root, request['model_manifest']), request['manifest_sha256'],
                                  request['checkpoint_sha256'], model_id=kind,
                                  actuator=request['actuator'], seed=request['seed'])
        if request['inference_variant'] == 'efficient':
            model = EfficientPowerHead.from_head(model)
        plant = Plant(request['actuator'], request['physics_step_ns'], root/'assets')
        atomic_json(folder/'request.json', request)
        result = execute_episode(plant, model, request['planner_settings'], seed=request['seed'],
             episode_id=request['episode_id'], phase='exploratory', condition='clean',
             model_partition=tuple(request['model_partition_ns']), actuator_interval_ns=request['actuator_interval_ns'])
        summary = save_episode(folder, result, model_id=kind,
                               checkpoint_hash=request['checkpoint_sha256'], run_id=folder.name)
        # Existing scalar_candidate_evaluations counts transition queries. It
        # already multiplies by batch size; do not multiply it by 128 again.
        transitions = summary['work']['scalar_candidate_evaluations']
        neural_rows = (transitions*(2 if request['actuator'] == 'lagged' else 1) if kind == 'power_correct'
                       else transitions*4 if kind.startswith('node_') else transitions if kind == 'mlp_h' else 0)
        summary.update(protocol_id=request['protocol_id'], inference_variant=request['inference_variant'],
                       neural_rows=neural_rows, oracle_physics_steps=getattr(model, 'physics_steps', 0),
                       missing_cells=[], resource_projection=request['resource_projection'])
        atomic_json(folder/'efficiency.json', dict(protocol_id=request['protocol_id'], neural_rows=neural_rows,
                                                 inference_variant=request['inference_variant']))
        imported = {str(Path(m.__file__).resolve().relative_to(root)) for m in list(sys.modules.values())
                    if getattr(m, '__file__', None) and str(m.__file__).endswith('.py')
                    and Path(m.__file__).resolve().is_relative_to(root/'poc')}
        if imported-set(files):
            raise ValueError('Undeclared local pilot imports: '+str(sorted(imported-set(files))))
        verify_hashes(root, hashes)
        checks = [dict(check_id=key, status='pass', scope=kind,
                       reason='Complete physical episode with frozen exploratory inputs', tolerance=None) for key in expected]
        return checks, summary

    return run_job(root, 'M40', 'exploratory', files, expected, [], producer,
                   config_keys={relative: request}, argv=[sys.executable, '-m', 'poc.efficiency_episode', relative])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    record = run_episode(args.root, args.request)
    print(args.root.resolve()/'evidence/M40'/record['run_id']/'manifest.json', flush=True)
    return 0 if record['completion_status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
