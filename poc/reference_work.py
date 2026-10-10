"""Reconstruct omitted bank work counts without rewriting experiment artifacts."""
from __future__ import annotations
import argparse
from pathlib import Path

from .io import atomic_json, read_json, safe_path, sha256_file, utc_now


def reconstruct_work(candidate_count, observed_oracle_steps):
    if candidate_count != 512 or type(candidate_count) is not int:
        raise ValueError('The complete fixed 512-candidate bank is required')
    queries = candidate_count * 6
    oracle = [dict(physics_step_ns=tick, scalar_transition_queries=queries,
                   physics_steps=queries * (40_000_000 // tick))
              for tick in (250_000, 125_000, 62_500)]
    total = sum(r['physics_steps'] for r in oracle)
    if type(observed_oracle_steps) is not int or observed_oracle_steps != total:
        raise ValueError('Reconstructed work disagrees with the observed simulator step counter')
    physics = []
    for step in (1_250_000, 625_000, 312_500):
        field_calls = 6 * 4 * (40_000_000 // step)
        physics.append(dict(integration_step_ns=step, network_calls=0,
            batched_vector_field_calls=field_calls,
            scalar_vector_field_evaluations=field_calls * candidate_count,
            scalar_transition_queries=queries, history_tokens_processed=0))
    return dict(oracle=oracle, physics=physics, oracle_physics_steps_total=total,
        count_provenance='Derived from complete successful bank coverage and fixed solver loops',
        timing_provenance='Use original resources.json; no new runtime or memory measurement',
        limitation='Does not count internal MuJoCo operations or turn simulator steps into neural NFE')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    import numpy as np
    from .planner_qualification import verify_frozen
    verify_frozen(root)
    batch_path = root / 'reports/robot_calibration_batch.json'
    batch = read_json(batch_path)
    rows = []
    for entry in batch['bank_audits']:
        path = safe_path(root, entry['path'])
        summary = read_json(path.parent / 'summary.json')
        with np.load(path.parent / 'reference_bank.npz', allow_pickle=False) as raw:
            if (raw['knots'].shape != (512, 6, 2)
                    or raw['oracle_costs'].shape != (3, 512)
                    or raw['physics_costs'].shape != (3, 512)
                    or not np.isfinite(raw['oracle_costs']).all()
                    or not np.isfinite(raw['physics_costs']).all()):
                raise ValueError('Resource reconstruction requires every successful saved bank score')
        rows.append(dict(actuator=summary['actuator'], episode_id=summary['episode_id'],
            source_manifest=entry['path'], source_manifest_sha256=entry['sha256'],
            **reconstruct_work(summary['ranking']['candidate_count'], summary['oracle_physics_steps'])))
    ids = [(r['actuator'], r['episode_id']) for r in rows]
    if len(ids) != 32 or set(ids) != {(a, e) for a in ('direct', 'lagged') for e in range(16)}:
        raise ValueError('Complete bank accounting requires all 32 distinct audits')
    result = dict(schema_version=1, recorded_utc=utc_now(), rows=rows,
        batch_sha256=sha256_file(batch_path),
        source_hashes={name: sha256_file(root / name) for name in
                      ('poc/reference_work.py', 'poc/reference_bank.py', 'poc/physics_fit.py',
                       'poc/mpc.py', 'poc/models/common.py')},
        oracle_physics_steps_total=sum(r['oracle_physics_steps_total'] for r in rows),
        scope='Derived operation ledger only; original raw observations and timing remain immutable')
    destination = root / 'reports/reference_bank_work.json'
    atomic_json(destination, result)
    print(destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
