from pathlib import Path

CHECKS = ('analytic', 'decimal', 'jacobians', 'composition', 'identity', 'weight_hash',
          'fixed_activation', 'units', 'zero_time', 'clocks', 'mutations', 'slide_fixtures', 'tests')


def run(root: Path, resume: bool = False):
    from poc.mechanics import run_m10
    return run_m10(root)


def evaluate(root: Path, evidence: list[dict], seen: dict):
    import numpy as np
    from poc.io import read_json, run_directory
    if len(evidence) != 1 or evidence[0]['phase'] != 'unit' or set(seen) != set(CHECKS):
        raise ValueError('G10 requires complete deterministic unit evidence')
    if any(seen[key]['status'] != 'pass' for key in CHECKS):
        raise ValueError('A mechanics invariant is invalid or unresolved')
    folder = run_directory(root, evidence[0])
    summary = read_json(folder / 'summary.json')['measurements']
    limits = {'analytic_max_abs_error': 1e-12, 'decimal_max_relative_error': 1e-12,
              'jacobian_max_relative_error': 1e-10, 'composition_max_abs_error': 1e-12,
              'identity_max_abs_error': 1e-12, 'slide_accepted_max_abs_error': 1e-12}
    for key, limit in limits.items():
        value = summary.get(key)
        if value is None or not np.isfinite(value) or value < 0 or value > limit:
            raise ValueError(f'G10 numerical budget not established: {key}')
    with np.load(folder / 'analytic.npz', allow_pickle=False) as arrays:
        if arrays['states_direct'].shape != (1000, 4) or arrays['states_lagged'].shape != (1000, 6):
            raise ValueError('G10 exact-case coverage is incomplete')
        for actuator in ('direct', 'lagged'):
            np.testing.assert_allclose(arrays['predicted_'+actuator], arrays['reference_'+actuator], atol=1e-12, rtol=1e-12)
    with np.load(folder / 'power_identity.npz', allow_pickle=False) as arrays:
        for actuator in ('direct', 'lagged'):
            for field in ('outputs', 'state_jac', 'command_jac', 'weight_grad', 'four_step_loss', 'four_step_grad'):
                values = arrays[actuator+'_'+field]
                for mode in (1, 2):
                    np.testing.assert_allclose(values[:, 0], values[:, mode], atol=1e-12, rtol=1e-12)
        if not np.all(arrays['checkpoint_hashes'] == arrays['checkpoint_hashes'][:, :1]):
            raise ValueError('Power modes used different checkpoints')
    with np.load(folder / 'slide_refinement.npz', allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays['endpoints'][-1], arrays['reference'], atol=1e-12, rtol=1e-12)
    if not all(read_json(folder / 'mutations.json')['detected'].values()):
        raise ValueError('A required mechanics mutation was not detected')
    clocks = read_json(folder / 'clock_traces.json')
    if any(sum(row['durations_ns']) != 40_000_000 for row in clocks['partitions']):
        raise ValueError('Clock partitions changed physical duration')
    evaluations = [{'criterion_id': ('K01.identity.' if key in {'identity','weight_hash'} else
                    'K01.clocks.' if key == 'clocks' else 'K01.mutations.' if key == 'mutations' else 'K01.oracle.')+key,
                    'status': 'pass', 'estimate': 1., 'bounds': None, 'threshold': None,
                    'operator': 'deterministic', 'reason': seen[key]['reason']} for key in CHECKS]
    return 'GO', 'valid', evaluations
