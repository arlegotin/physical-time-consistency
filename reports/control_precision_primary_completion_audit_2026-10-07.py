"""Read-only operational verification of the closed 400-episode extension."""
from collections import defaultdict
from pathlib import Path
import math
import shutil

from poc.contracts import validate_record
from poc.control_precision import verify_episode_registry
from poc.control_utility import audit_raw_episode
from poc.io import atomic_json, read_json, safe_path, sha256_file, utc_now


root = Path.cwd().resolve()
registry_path = 'config/control_precision_control_jobs.json'
batch_path = 'reports/control_precision_primary.json'
registry = verify_episode_registry(root, read_json(root / registry_path))
batch = read_json(root / batch_path)
assert batch['state'] == 'complete'
assert batch['expected_episodes'] == batch['completed_episodes'] == 400
assert batch['registry_sha256'] == sha256_file(root / registry_path)
jobs = {r['cell_id']: r for r in registry['jobs']}
entries = {r['cell_id']: r for r in batch['episodes']}
assert len(jobs) == len(entries) == len(batch['episodes']) == 400
assert set(jobs) == set(entries)
hashes = dict(registry['input_hashes'])
hashes[registry_path] = sha256_file(root / registry_path)
hashes[batch_path] = sha256_file(root / batch_path)
groups = defaultdict(list)
paired = defaultdict(dict)
verified = []


def declare(name, digest):
    assert name not in hashes or hashes[name] == digest, ('conflicting hash', name)
    hashes[name] = digest


for cell, job in jobs.items():
    entry = entries[cell]
    declare(job['path'], job['sha256'])
    assert entry['request_sha256'] == job['sha256']
    request = read_json(safe_path(root, job['path']))
    path = safe_path(root, entry['manifest'])
    declare(entry['manifest'], entry['manifest_sha256'])
    manifest = read_json(path)
    validate_record('manifest', manifest, root)
    assert manifest['completion_status'] == 'complete'
    assert manifest['phase'] == 'confirmatory' and manifest['milestone_id'] == 'M70'
    marker = path.with_name('COMPLETED')
    assert read_json(marker)['manifest_sha256'] == entry['manifest_sha256']
    declare(str(marker.relative_to(root)), sha256_file(marker))
    for artifact in manifest['produced_files']:
        declare(artifact['path'], artifact['sha256'])
    for name, digest in manifest['job_inputs'].get('dependencies', {}).items():
        declare(name, digest)
    for name, digest in manifest['job_inputs'].get('wheel_hashes', {}).items():
        declare('wheelhouse/' + name, digest)
    for name, keys in manifest['job_inputs'].get('configuration_keys', {}).items():
        current = read_json(safe_path(root, name))
        assert all(current.get(k) == v for k, v in keys.items()), name
    assert read_json(path.with_name('request.json')) == request
    summary = read_json(path.with_name('summary.json'))
    resources = read_json(path.with_name('resources.json'))
    assert summary == entry['summary'] and resources == entry['resources']
    for key in ('phase', 'condition', 'model_id', 'actuator', 'seed', 'episode_id',
                'power_mode', 'model_partition_ns', 'actuator_interval_ns', 'physics_step_ns'):
        assert summary[key] == request[key], (cell, key)
    assert summary['complete'] is True and summary['checkpoint_hash'] == request['checkpoint_sha256']
    assert summary['model_partition_ns'] == [5_000_000] * 8
    assert summary['actuator_interval_ns'] == 40_000_000 and summary['physics_step_ns'] == 250_000
    assert math.isfinite(summary['realized_cost']) and summary['realized_cost'] >= 0
    raw = audit_raw_episode(path.parent, summary, expected_tick_ns=250_000)
    assert raw == entry['raw_audit'] and raw['n_observations'] == raw['n_real_anchors'] == 40
    groups[(summary['actuator'], summary['model_id'])].append(resources)
    key = (summary['actuator'], summary['seed'], summary['episode_id'])
    paired[key][summary['model_id']] = summary
    verified.append(dict(cell_id=cell, manifest=entry['manifest'],
                         manifest_sha256=entry['manifest_sha256'], request=job['path'],
                         request_sha256=job['sha256'], fallback_count=summary['fallback_count']))
    if len(verified) % 50 == 0:
        print(f'Verified raw episode {len(verified)}/400', flush=True)

assert len(paired) == 200
for key, value in paired.items():
    assert set(value) == {'mlp_dt', 'power_correct'}, key
    assert all(value['mlp_dt'][k] == value['power_correct'][k]
               for k in ('initial_state', 'target', 'seed_namespaces')), key
for name, digest in hashes.items():
    assert sha256_file(safe_path(root, name)) == digest, name

resource_fields = ('cpu_seconds', 'wall_seconds', 'disk_bytes', 'peak_rss_bytes')
for resources in (e['resources'] for e in entries.values()):
    assert all(type(resources[k]) in (float, int) and math.isfinite(resources[k])
               and resources[k] >= 0 for k in resource_fields)
by_cell = [dict(actuator=a, model_id=m, n_episodes=len(rows),
                cpu_seconds=math.fsum(r['cpu_seconds'] for r in rows),
                producer_wall_seconds=math.fsum(r['wall_seconds'] for r in rows),
                logical_artifact_bytes=sum(r['disk_bytes'] for r in rows),
                peak_rss_bytes=max(r['peak_rss_bytes'] for r in rows))
           for (a, m), rows in groups.items()]
assert len(by_cell) == 4 and all(r['n_episodes'] == 100 for r in by_cell)
result = dict(schema_version=1, state='complete_verified_primary_batch',
    recorded_utc=utc_now(), operational_only=True, new_statistical_look=False,
    alpha_spent=0, scientific_verdict=None, n_episodes=400, n_paired_tasks=200,
    scope='New seeds 5–9, twenty tasks each, both plants and MLP/head at the frozen 5 ms model clock; numerical qualification and inference remain separate',
    registry_path=registry_path, registry_sha256=hashes[registry_path],
    batch_path=batch_path, batch_sha256=hashes[batch_path],
    unique_declared_files_rehashed=len(hashes),
    resource_scope='Completed producer records only; startup, coordinator/root verification, training, interrupted attempts, finer physics and replay excluded',
    actual_resources=dict(cpu_seconds=math.fsum(r['cpu_seconds'] for r in by_cell),
        producer_wall_seconds=math.fsum(r['producer_wall_seconds'] for r in by_cell),
        logical_artifact_bytes=sum(r['logical_artifact_bytes'] for r in by_cell),
        peak_rss_bytes=max(r['peak_rss_bytes'] for r in by_cell),
        fallback_count=sum(r['fallback_count'] for r in verified), rss_convention='macOS bytes'),
    actual_free_bytes=shutil.disk_usage(root).free, by_cell=by_cell,
    verified=verified, file_hashes=hashes)
target = root / 'reports/control_precision_primary_completion_2026-10-07.json'
assert not target.exists(), 'Preserve existing completion evidence'
atomic_json(target, result)
print({k: result[k] for k in ('state', 'n_episodes', 'n_paired_tasks',
      'unique_declared_files_rehashed', 'actual_resources')}, flush=True)
