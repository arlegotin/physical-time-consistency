"""Replay a prospectively selected first-decision bank with both physical references."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import torch
from .io import atomic_json, read_json, safe_path, sha256_file
from .robot_calibration import cost_envelope, ranking_check


def score_bank(oracle_models, physics_models, state, target, knots, recorded_costs):
    from .mpc import rollout_candidates
    if len(oracle_models) != 3 or len(physics_models) != 3:
        raise ValueError('Both references require three solver resolutions')
    arrays = {'knots': np.asarray(knots), 'recorded_costs': np.asarray(recorded_costs)}
    for label, models in (('oracle', oracle_models), ('physics', physics_models)):
        packets = [rollout_candidates(model, state, target, None, knots, (40_000_000,), 40_000_000)
                   for model in models]
        if any(any(status != 'ok' for status in p['statuses']) for p in packets):
            raise ValueError('Physical-reference bank has an invalid prediction')
        arrays[label + '_costs'] = np.stack([p['costs'] for p in packets])
        arrays[label + '_anchors'] = np.stack([p['anchors'] for p in packets])
    if not np.allclose(arrays['physics_costs'][0], recorded_costs, atol=1e-12, rtol=1e-12):
        raise ValueError('Replay disagrees with recorded first-decision physics scores')
    oe, pe = cost_envelope(arrays['oracle_costs']), cost_envelope(arrays['physics_costs'])
    arrays.update(oracle_envelope=oe, physics_envelope=pe)
    return arrays, {'ranking': ranking_check(arrays['oracle_costs'][0], arrays['physics_costs'][0], oe, pe),
        'maximum_q_disagreement_rad': float(abs(arrays['oracle_anchors'][-1, :, :, :2]
                                                   - arrays['physics_anchors'][-1, :, :, :2]).max()),
        'maximum_v_disagreement_rad_s': float(abs(arrays['oracle_anchors'][-1, :, :, 2:4]
                                                    - arrays['physics_anchors'][-1, :, :, 2:4]).max()),
        'maximum_oracle_cost_envelope': float(oe.max()), 'maximum_physics_cost_envelope': float(pe.max()),
        'envelope_interpretation': 'Empirical two-successive-refinement discrepancy and roundoff; not certified',
        'scope': 'Identical saved first-decision bank; no learned-controller performance claim'}


class RefinedPhysics:
    def __init__(self, model, step):
        self.model, self.step = model, step

    def predict(self, state, command, h_s, history=None):
        return self.model.predict(state, command, h_s, history=history, integration_step_s=self.step)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    from .contracts import verify_run
    from .control_jobs import SOURCES
    from .control_models import load_selected, SimOracle
    from .control_artifacts import _npz
    from .execution import cpu_defaults, run_job
    cpu_defaults(9000)
    manifest = args.manifest.resolve()
    record = read_json(manifest); verify_run(root, record)
    summary = read_json(manifest.parent / 'summary.json')
    if (summary['model_id'] != 'physics_fit' or summary['phase'] != 'calibration'
            or summary['episode_id'] not in range(16) or summary['seed'] != 9000
            or summary['model_partition_ns'] != [40_000_000] or summary['condition'] != 'clean'):
        raise ValueError('Audit only the fixed calibration reference workload')
    request = read_json(manifest.parent / 'request.json')
    source_manifest = safe_path(root, request['model_manifest'])
    files = list(SOURCES) + ['poc/robot_calibration.py', 'poc/reference_bank.py',
        str(manifest.relative_to(root)), str((manifest.parent / 'candidates.npz').relative_to(root)),
        str((manifest.parent / 'summary.json').relative_to(root)), str((manifest.parent / 'request.json').relative_to(root)),
        request['model_manifest'], str((source_manifest.parent / 'weights.pt').relative_to(root)),
        str((source_manifest.parent / 'normalization.json').relative_to(root)),
        f'assets/two_link_{request["actuator"]}.xml']
    dependencies = {name: sha256_file(root / name) for name in files}
    expected = ('control.reference_bank',)

    def producer(folder):
        physics = load_selected(source_manifest, request['manifest_sha256'], request['checkpoint_sha256'],
            model_id='physics_fit', actuator=request['actuator'], seed=9000)
        bank = np.load(manifest.parent / 'candidates.npz', allow_pickle=False)
        if bank['knots'].shape != (40, 512, 6, 2):
            raise ValueError('Unexpected fixed-budget calibration bank coverage')
        oracle = [SimOracle(request['actuator'], tick, root / 'assets') for tick in (250_000, 125_000, 62_500)]
        arrays, result = score_bank(oracle, [RefinedPhysics(physics, h) for h in (.00125, .000625, .0003125)],
            torch.tensor(summary['initial_state'], dtype=torch.float64), summary['target'],
            bank['knots'][0], bank['costs'][0])
        _npz(folder / 'reference_bank.npz', **arrays)
        atomic_json(folder / 'source_episode.json', {'manifest': str(manifest.relative_to(root)),
            'sha256': sha256_file(manifest), 'decision_index': 0, 'candidate_ids': bank['global_ids'][0].tolist()})
        if any(sha256_file(root / name) != digest for name, digest in dependencies.items()):
            raise ValueError('Reference-bank provenance changed during replay')
        result.update(actuator=request['actuator'], episode_id=request['episode_id'], missing_cells=[],
                      oracle_physics_steps=sum(m.physics_steps for m in oracle))
        checks = [{'check_id': expected[0], 'status': 'pass', 'scope': 'calibration',
                   'reason': 'Complete identical bank replay, preserved scores and refinement arrays', 'tolerance': None}]
        return checks, result

    result = run_job(root, 'M40', 'calibration', files, expected, [], producer,
        config_keys={str((manifest.parent / 'request.json').relative_to(root)): request},
        argv=[sys.executable, '-m', 'poc.reference_bank', str(manifest.relative_to(root))])
    print(root / 'evidence/M40' / result['run_id'] / 'manifest.json', flush=True)
    return 0 if result['completion_status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
