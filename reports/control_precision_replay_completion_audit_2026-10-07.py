"""Verify all twenty actual precision replays without launching experiments."""
from pathlib import Path
import json
import math

from poc.control_precision_replay import _observe, verify_registry
from poc.io import atomic_json, read_json, safe_path, sha256_file, utc_now


root = Path.cwd()
output = root / 'reports/control_precision_replay_completion_2026-10-07.json'
assert not output.exists(), 'Completed audit receipts are immutable'
registry_name = 'config/control_precision_replay_jobs.json'
batch_name = 'reports/control_precision_replay.json'
registry, batch = read_json(root / registry_name), read_json(root / batch_name)
assert batch['state'] == 'complete' and batch['diagnosis_required'] is False
assert batch['completed_episodes'] == batch['expected_episodes'] == 20
assert batch['registry_sha256'] == sha256_file(root / registry_name)
assert batch['fresh_training'] is False
assert batch['new_independent_training_clusters'] == batch['alpha_spent'] == 0
verify_registry(root, registry)
jobs = {j['cell_id']: j for j in registry['jobs']}
assert len(jobs) == 20 and len(batch['episodes']) == 20
hashes = dict(registry['input_hashes'])
hashes.update({name: sha256_file(root / name) for name in (registry_name, batch_name)})
coordinates, comparisons, resources = set(), [], []
for entry in batch['episodes']:
    job = jobs[entry['cell_id']]
    path = safe_path(root, entry['manifest'])
    assert _observe(root, job, path) == entry
    comparison = entry['comparison']
    assert comparison['status'] == 'pass' and comparison['diagnosis_required'] is False
    key = (comparison['actuator'], comparison['model_id'], comparison['seed'], comparison['episode_id'])
    assert key not in coordinates
    coordinates.add(key); comparisons.append(comparison); resources.append(entry['resources'])
    hashes[job['path']] = job['sha256']
    for name in (entry['manifest'], job['parent_manifest']):
        manifest_path = safe_path(root, name)
        hashes[name] = sha256_file(manifest_path)
        marker = manifest_path.parent / 'COMPLETED'
        hashes[str(marker.relative_to(root))] = sha256_file(marker)
        manifest = read_json(manifest_path)
        for artifact in manifest['produced_files']:
            digest = sha256_file(safe_path(root, artifact['path']))
            assert digest == artifact['sha256']
            assert artifact['path'] not in hashes or hashes[artifact['path']] == digest
            hashes[artifact['path']] = digest
        for dependency, digest in manifest['job_inputs'].get('dependencies', {}).items():
            assert sha256_file(safe_path(root, dependency)) == digest
            assert dependency not in hashes or hashes[dependency] == digest
            hashes[dependency] = digest
expected = {(a, m, s, 0) for a in ('direct', 'lagged')
            for m in ('mlp_dt', 'power_correct') for s in range(5, 10)}
assert coordinates == expected
for name, digest in hashes.items():
    assert sha256_file(safe_path(root, name)) == digest, name
hashes[str(Path(__file__).relative_to(root))] = sha256_file(Path(__file__))
maximum_differences = {key: max(c[key] for c in comparisons) for key in (
    'max_q_difference_rad', 'max_v_difference_rad_s', 'max_command_difference_nm',
    'absolute_cost_difference', 'relative_cost_difference')}
activations = [c['max_activation_difference_nm'] for c in comparisons if c['max_activation_difference_nm'] is not None]
maximum_differences['max_activation_difference_nm'] = max(activations)
receipt = dict(schema_version=1, recorded_utc=utc_now(), state='complete_verified_precision_replay',
               n_episodes=20, scope='Both plants, MLP/head, new seeds 5–9, first task each; fresh processes with the exact selected weights',
               registry_path=registry_name, registry_sha256=hashes[registry_name],
               batch_path=batch_name, batch_sha256=hashes[batch_name],
               maximum_differences=maximum_differences, comparisons=comparisons,
               actual_resources=dict(cpu_seconds=math.fsum(r['cpu_seconds'] for r in resources),
                                     producer_wall_seconds=math.fsum(r['wall_seconds'] for r in resources),
                                     logical_artifact_bytes=sum(r['disk_bytes'] for r in resources),
                                     peak_rss_bytes=max(r['peak_rss_bytes'] for r in resources)),
               resource_scope='Only completed replay episode producers; startup, preparation/root audits, original runs, training and interruptions are separate',
               file_hashes=hashes, unique_declared_files_rehashed=len(hashes),
               actual_verification_command='PYTHONPATH=. .venv/bin/python reports/control_precision_replay_completion_audit_2026-10-07.py > .superpowers/sdd/2026-10-06-control-precision/audit-precision-replay.log 2>&1',
               fresh_training=False, new_independent_training_clusters=0, alpha_spent=0,
               full_ptco_terminal_verdict=None)
atomic_json(output, receipt)
print(json.dumps(dict(state=receipt['state'], n_episodes=20, maximum_differences=maximum_differences), sort_keys=True))
