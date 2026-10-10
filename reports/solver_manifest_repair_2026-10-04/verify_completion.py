from pathlib import Path
import math
import shutil

from poc.contracts import verify_run
from poc.io import atomic_json, read_json, sha256_file, utc_now

root = Path.cwd()
folder = root / 'reports/solver_manifest_repair_2026-10-04'
original_path = folder / 'original/reports/control_confirmation_solver.json'
replacement_path = root / 'reports/control_confirmation_solver.json'
original = read_json(original_path)
replacement = read_json(replacement_path)
for report in (original, replacement):
    assert report['state'] == 'complete'
    assert report['expected_episodes'] == report['completed_episodes'] == len(report['episodes']) == 100
old = {entry['parent_cell_id']: entry for entry in original['episodes']}
new = {entry['parent_cell_id']: entry for entry in replacement['episodes']}
assert len(old) == len(new) == 100 and old.keys() == new.keys()
assert original['closed_loop_solver_cells_required'] == replacement['closed_loop_solver_cells_required']
rows = []
resources = []
for cell in sorted(old):
    before, after = old[cell], new[cell]
    before_path, after_path = root / before['manifest'], root / after['manifest']
    assert sha256_file(before_path) == before['manifest_sha256']
    assert sha256_file(after_path) == after['manifest_sha256']
    old_manifest, new_manifest = read_json(before_path), read_json(after_path)
    assert old_manifest['completion_status'] == new_manifest['completion_status'] == 'complete'
    assert old_manifest['job_key'] != new_manifest['job_key']
    verify_run(root, new_manifest)
    assert new_manifest['job_inputs']['configuration_keys'] == {
        'config/control_confirmation_solver_audit.json':
        old_manifest['job_inputs']['configuration_keys']['config/control_confirmation_solver_audit.json']
    }
    assert old_manifest['job_inputs']['configuration_keys']['parent_cell_id'] == cell
    # Historical dependencies are intentionally stale after the repair; verify
    # their immutable outputs rather than pretending they pass current closure.
    for artifact in old_manifest['produced_files']:
        assert sha256_file(root / artifact['path']) == artifact['sha256']
    assert before['summary'] == after['summary']
    observed = {}
    for name in ('summary.json', 'local_probes.json', 'candidate_banks.json'):
        digest = sha256_file(before_path.parent / name)
        assert digest == sha256_file(after_path.parent / name)
        observed[name] = digest
    resources.append(after['resources'])
    rows.append(dict(cell_id=cell, original_manifest=before['manifest'],
                     original_manifest_sha256=before['manifest_sha256'],
                     replacement_manifest=after['manifest'],
                     replacement_manifest_sha256=after['manifest_sha256'],
                     unchanged_scientific_artifact_hashes=observed))
receipt = dict(schema_version=1, recorded_utc=utc_now(),
    status='complete_verified_repair', scope='Solver audit manifest metadata only',
    scientific_decision='not_evaluated', statistical_look_reserved=False,
    original_report_sha256=sha256_file(original_path),
    replacement_report_sha256=sha256_file(replacement_path),
    request_sha256=sha256_file(folder / 'request.json'),
    diagnosis_sha256=sha256_file(folder / 'diagnosis.json'),
    replacement_input_registry_sha256=sha256_file(root / 'config/control_confirmation_solver_inputs.json'),
    verified_replacement_manifests=len(rows), preserved_original_manifests=len(rows),
    unchanged_summaries=100, unchanged_local_probe_artifacts=100,
    unchanged_candidate_bank_artifacts=100,
    required_closed_loop_cells=replacement['closed_loop_solver_cells_required'],
    completed_producer_resources=dict(
        cpu_seconds=math.fsum(row['cpu_seconds'] for row in resources),
        wall_seconds=math.fsum(row['wall_seconds'] for row in resources),
        disk_bytes=sum(row['disk_bytes'] for row in resources),
        peak_rss_bytes=max(row['peak_rss_bytes'] for row in resources)),
    actual_free_bytes=shutil.disk_usage(root).free, episodes=rows)
atomic_json(folder / 'completion.json', receipt)
print({key: value for key, value in receipt.items() if key not in ('episodes', 'required_closed_loop_cells')})
print('closed_loop_cells', len(receipt['required_closed_loop_cells']))
