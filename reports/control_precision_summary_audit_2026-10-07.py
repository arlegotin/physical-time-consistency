"""Independent, read-only reconciliation of the already recorded second look."""
from pathlib import Path
import hashlib
import json
import math
import statistics

import numpy as np
from scipy.stats import t

from poc.io import atomic_json, read_json, sha256_file, utc_now


root = Path.cwd()
output = root / 'reports/control_precision_summary_audit_2026-10-07.json'
assert not output.exists(), 'Completed audit receipts are immutable'
result = read_json(root / 'reports/control_precision_result.json')
numeric = read_json(root / 'reports/control_precision_numerical.json')
assert result['look_index'] == 2 and result['family_size'] == 8
assert result['tail_alpha'] == .00078125 and numeric['state'] == 'qualified'
assert numeric['required_actions'] == []
source_names = [
    'reports/control_precision_result.json',
    'reports/control_precision_numerical.json',
    'reports/control_confirmation_batch.json',
    'reports/control_precision_primary.json',
    'reports/control_confirmation_result.json',
    'reports/control_precision_primary_completion_2026-10-07.json',
    'reports/control_precision_finer_completion_2026-10-07.json',
    'decisions/control_utility_looks/look-00002.json',
]
hashes = {name: sha256_file(root / name) for name in source_names}
episodes = {}
for batch_name in source_names[2:4]:
    batch = read_json(root / batch_name)
    assert batch['state'] == 'complete'
    for entry in batch['episodes']:
        cached = entry['summary']
        if cached['model_id'] not in ('mlp_dt', 'power_correct'):
            continue
        if cached['model_partition_ns'] != [5_000_000] * 8:
            continue
        manifest_path = root / entry['manifest']
        assert sha256_file(manifest_path) == entry['manifest_sha256']
        manifest = read_json(manifest_path)
        summaries = [f for f in manifest['produced_files'] if f['path'].endswith('/summary.json')]
        assert len(summaries) == 1
        artifact = summaries[0]
        assert sha256_file(root / artifact['path']) == artifact['sha256']
        actual = read_json(root / artifact['path'])
        assert actual == cached and actual['complete'] and actual['missing_cells'] == []
        key = (actual['actuator'], actual['model_id'], actual['seed'], actual['episode_id'])
        assert key not in episodes
        episodes[key] = actual
        hashes[entry['manifest']] = entry['manifest_sha256']
        hashes[artifact['path']] = artifact['sha256']
expected = {(a, m, s, e) for a in ('direct', 'lagged')
            for m in ('mlp_dt', 'power_correct') for s in range(10) for e in range(20)}
assert set(episodes) == expected
groups = []
for a in ('direct', 'lagged'):
    for m in ('mlp_dt', 'power_correct'):
        seeds = [[episodes[(a, m, s, e)] for e in range(20)] for s in range(10)]
        costs = [math.fsum(r['realized_cost'] for r in rows) / 20 for rows in seeds]
        success = [sum(r['success'] for r in rows) for rows in seeds]
        fallback = [sum(r['fallback_count'] for r in rows) for rows in seeds]
        row = next(r for r in result['seed_costs']['rows'] if r['actuator'] == a and r['model_id'] == m)
        assert costs == row['cost_by_seed']
        assert success == row['success_by_seed'] and fallback == row['fallback_by_seed']
        groups.append(dict(actuator=a, model_id=m, n_episodes=200,
                           cost_by_seed=costs, success_count=sum(success), fallback_count=sum(fallback)))
indices = np.random.Generator(np.random.PCG64(np.random.SeedSequence([9102, 2]))).integers(
    0, 10, size=(20_000, 10), dtype=np.int64)
index_hash = hashlib.sha256(indices.tobytes()).hexdigest()
checks = []
for criterion in result['criteria']:
    a = criterion['actuator']
    b = next(r['cost_by_seed'] for r in groups if r['actuator'] == a and r['model_id'] == 'mlp_dt')
    m = next(r['cost_by_seed'] for r in groups if r['actuator'] == a and r['model_id'] == 'power_correct')
    differences = [x - y for x, y in zip(b, m)]
    mean = statistics.mean(differences)
    halfwidth = float(t.ppf(1 - .00078125, 9)) * statistics.stdev(differences) / math.sqrt(10)
    envelope = numeric['envelopes'][criterion['criterion_id']]
    lower, upper = mean - halfwidth - envelope['lower'], mean + halfwidth + envelope['upper']
    central = np.quantile(np.asarray(differences)[indices].mean(axis=1), [.025, .975], method='linear')
    bounds = criterion['bounds']
    for actual, saved in ((mean, bounds['mean']), (lower, bounds['lower']), (upper, bounds['upper'])):
        assert math.isclose(actual, saved, rel_tol=1e-12, abs_tol=1e-14)
    assert np.allclose(central, bounds['central_bootstrap'], rtol=1e-12, atol=1e-14)
    assert bounds['bootstrap_indices_sha256'] == index_hash
    assert differences == criterion['difference_by_seed']
    assert lower > 0 and central[0] > envelope['lower'] and all(x > 0 for x in differences)
    assert not bounds['influence']['unstable'] and criterion['status'] == 'lower_cost'
    for omitted in range(10):
        remaining = differences[:omitted] + differences[omitted + 1:]
        loo_lower = statistics.mean(remaining) - float(t.ppf(1 - .00078125, 8)) * statistics.stdev(remaining) / 3 - envelope['lower']
        assert loo_lower > 0
    checks.append(dict(criterion_id=criterion['criterion_id'], status='lower_cost',
                       relative_reduction=1 - statistics.mean(m) / statistics.mean(b),
                       difference_bounds=[lower, upper], all_ten_seed_differences_positive=True,
                       all_leave_one_out_lower_bounds_positive=True))
parent = read_json(root / 'reports/control_confirmation_result.json')
updated = {r['criterion_id'] for r in checks}
expected_cached = [r for r in parent['criteria'] if r['criterion_id'] not in updated]
assert len(expected_cached) == 6
assert result['cached_criteria'] == [dict(r, source_look_index=1) for r in expected_cached]
ledger = read_json(root / 'decisions/control_utility_looks/look-00002.json')
assert ledger['evidence_identity'] == result['evidence_identity']
assert result['fallback_intervals'] == sum(r['fallback_count'] for r in groups) == 0
hashes[str(Path(__file__).relative_to(root))] = sha256_file(Path(__file__))
receipt = dict(schema_version=1, recorded_utc=utc_now(), state='independently_reconciled_recorded_look',
               n_episodes=800, n_training_seeds=10, n_paired_tasks=400,
               groups=groups, criteria=checks, input_hashes=hashes,
               raw_trajectory_verification='Original and new raw audits remain referenced through the registered consumer and completion proofs; this check independently reopens actual episode summaries.',
               new_statistical_look=False, additional_alpha_spent=0,
               reproduction_status='New twenty-coordinate replay remains pending; no full PTCO verdict is assigned.')
atomic_json(output, receipt)
print(json.dumps(dict(state=receipt['state'], n_episodes=800, criteria=checks), sort_keys=True))
