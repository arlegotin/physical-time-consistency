"""Raw qualification and paired inference for the registered control extension."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from .control_precision import ACTUATORS, CRITERIA, MODELS, PARAMETERS
from .statistics import contrast_bounds, look_alpha


def _indexed(rows, seeds, *, tick=250_000):
    from .control_episode import episode_conditions
    expected = {(a, m, s, e) for a in ACTUATORS for m in MODELS for s in seeds for e in range(20)}
    indexed, checkpoints = {}, {}
    try:
        for row in rows:
            a, m, s, e = key = tuple(row[k] for k in ('actuator', 'model_id', 'seed', 'episode_id'))
            value = row['realized_cost']
            if (key not in expected or key in indexed or type(s) is not int or type(e) is not int
                    or row['phase'] != 'confirmatory' or row['complete'] is not True
                    or row['condition'] != 'clean' or row['power_mode'] is not None
                    or row['physics_step_ns'] != tick or row['actuator_interval_ns'] != 40_000_000
                    or row['model_partition_ns'] != [5_000_000]*8
                    or type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    or type(row['success']) is not bool or type(row['fallback_count']) is not int
                    or not 0 <= row['fallback_count'] <= 40
                    or not isinstance(row['checkpoint_hash'], str)
                    or not re.fullmatch('[0-9a-f]{64}', row['checkpoint_hash'])):
                raise ValueError('Incomplete or changed registered precision episode')
            task = episode_conditions(s, e, a, 'confirmatory')
            if (row['initial_state'] != task['initial_state'].tolist() or row['target'] != task['target'].tolist()
                    or row['seed_namespaces'] != {'cem': task['cem_seed_entropy'],
                                                'disturbance': task['disturbance_seed_entropy']}):
                raise ValueError('Changed paired task or random-number namespace')
            coordinate = (a, m, s)
            if coordinate in checkpoints and checkpoints[coordinate] != row['checkpoint_hash']:
                raise ValueError('Selected checkpoint changed during evaluation')
            checkpoints[coordinate] = row['checkpoint_hash']
            indexed[key] = row
    except (KeyError, TypeError) as error:
        raise ValueError('Incomplete precision episode fields') from error
    return indexed


def aggregate_pairs(original: list[dict], appended: list[dict], request: dict) -> dict:
    """Twenty paired tasks are nested within each of ten independent fits."""
    if request.get('action_parameters') != PARAMETERS:
        raise ValueError('Changed prospective precision design')
    old, new = _indexed(original, range(5)), _indexed(appended, range(5, 10))
    if len(old) != 400 or len(new) != 400:
        raise ValueError('Complete retained and appended paired cohorts required')
    indexed = old | new
    rows = []
    for a in ACTUATORS:
        for m in MODELS:
            clusters = [[indexed[(a, m, s, e)] for e in range(20)] for s in range(10)]
            rows.append(dict(actuator=a, model_id=m, model_step_ns=5_000_000,
                cost_by_seed=[math.fsum(r['realized_cost'] for r in c)/20 for c in clusters],
                fallback_by_seed=[sum(r['fallback_count'] for r in c) for c in clusters],
                success_by_seed=[sum(r['success'] for r in c) for c in clusters]))
    return dict(seed_ids=list(range(10)), episodes_per_seed=20, n_episodes=800, rows=rows)


def _envelopes(values):
    if (set(values) != set(CRITERIA) or any(set(v) != {'lower', 'upper'} or any(
            type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in v.values())
            for v in values.values())):
        raise ValueError('Both finite measured outward numerical envelopes required')
    return values


def evaluate_precision(costs: dict, *, look_index: int, envelopes: dict) -> dict:
    """Use the original eight-family, two-direction control spending rule."""
    if (type(look_index) is not int or look_index != 2 or costs.get('seed_ids') != list(range(10))
            or costs.get('episodes_per_seed') != 20 or costs.get('n_episodes') != 800):
        raise ValueError('Changed registered precision look or independent clusters')
    _envelopes(envelopes)
    indexed = {(r['actuator'], r['model_id']): r for r in costs['rows']}
    if len(costs['rows']) != 4 or set(indexed) != {(a, m) for a in ACTUATORS for m in MODELS}:
        raise ValueError('Four complete paired seed cost groups required')
    for row in costs['rows']:
        x = np.asarray(row['cost_by_seed'])
        if x.shape != (10,) or not np.isfinite(x).all() or (x < 0).any():
            raise ValueError('Invalid independent seed-level costs')
    alpha = look_alpha(look_index)
    tail = alpha/16
    indices = np.random.Generator(np.random.PCG64(np.random.SeedSequence([9102, look_index]))).integers(
        0, 10, size=(20_000, 10), dtype=np.int64)
    criteria = []
    for a, criterion in zip(ACTUATORS, CRITERIA):
        baseline = np.asarray(indexed[(a, 'mlp_dt')]['cost_by_seed'], dtype=np.float64)
        method = np.asarray(indexed[(a, 'power_correct')]['cost_by_seed'], dtype=np.float64)
        difference = baseline-method
        envelope = envelopes[criterion]
        bounds = contrast_bounds(difference, tail, indices, envelope)
        status = 'unresolved'
        if bounds['status'] == 'unresolved_variance':
            status = 'unresolved_variance'
        elif bounds['influence']['unstable']:
            status = 'unresolved_influence'
        elif bounds['lower'] > 0 and bounds['central_bootstrap'][0]-envelope['lower'] > 0:
            status = 'lower_cost'
        elif bounds['upper'] < 0 and bounds['central_bootstrap'][1]+envelope['upper'] < 0:
            status = 'higher_cost'
        bmean, mmean = float(baseline.mean()), float(method.mean())
        criteria.append(dict(criterion_id=criterion, actuator=a, competitor='mlp_dt', model_step_ns=5_000_000,
            n_training_seeds=10, episodes_per_seed=20, status=status, baseline_mean_cost=bmean,
            method_mean_cost=mmean, point_difference=float(difference.mean()),
            relative_reduction=1-mmean/bmean if bmean > 0 else None,
            difference_by_seed=difference.tolist(), bounds=bounds,
            fallback_by_seed=indexed[(a, 'power_correct')]['fallback_by_seed']))
    return dict(look_index=look_index, look_alpha=alpha, tail_alpha=tail, family_size=8,
                bootstrap_seed_entropy=[9102, look_index], bootstrap_resamples=20_000, criteria=criteria)


def compare_precision_refinement(primary, finer, old_envelopes, *, look_index):
    """Measure the same frozen paths at the next finer physics tick."""
    if look_index != 2:
        raise ValueError('Changed prospective numerical look')
    _envelopes(old_envelopes)
    p, f = _indexed(primary, range(5, 10)), _indexed(finer, range(5, 10), tick=125_000)
    if len(p) != 400:
        raise ValueError('All new primary episodes required before numerical qualification')
    scopes = {}
    for a in ACTUATORS:
        actual = {key for key in f if key[0] == a}
        subset = {key for key in p if key[0] == a and key[3] < 5}
        full = {key for key in p if key[0] == a}
        if actual not in (subset, full):
            raise ValueError('Complete first-five or full finer episode coverage required for each plant')
        scopes[a] = 'full' if actual == full else 'subset'
    discrepancies, maxima, affected = [], {}, set()
    for key in sorted(f):
        a, m, s, e = key
        x, y = p[key], f[key]
        if x['checkpoint_hash'] != y['checkpoint_hash']:
            raise ValueError('Finer physics changed the checkpoint')
        delta = abs(x['realized_cost']-y['realized_cost'])
        relative = delta/x['realized_cost'] if x['realized_cost'] else (0. if delta == 0 else None)
        success_changed = x['success'] != y['success']
        fallback_changed = x['fallback_count'] != y['fallback_count']
        if relative is None or relative > .02 or success_changed or fallback_changed:
            affected.add(a)
        maxima[(a, m)] = max(maxima.get((a, m), 0.), delta)
        discrepancies.append(dict(actuator=a, model_id=m, seed=s, episode_id=e,
            absolute_cost_difference=delta, relative_cost_difference=relative,
            success_changed=success_changed, fallback_changed=fallback_changed))
    envelopes, effects = {}, []
    for a, criterion in zip(ACTUATORS, CRITERIA):
        def improvement(index):
            means = [math.fsum(index[k]['realized_cost'] for k in f if k[0] == a and k[1] == m)
                     /sum(k[0] == a and k[1] == m for k in f) for m in MODELS]
            return 1-means[1]/means[0] if means[0] > 0 else None
        before, after = improvement(p), improvement(f)
        movement = abs(after-before) if before is not None and after is not None else None
        if movement is None or movement > .02:
            affected.add(a)
        eta = maxima[(a, 'mlp_dt')]+maxima[(a, 'power_correct')]
        envelopes[criterion] = {side: max(eta, old_envelopes[criterion][side]) for side in ('lower', 'upper')}
        effects.append(dict(criterion_id=criterion, primary_reduction=before, finer_reduction=after,
                            absolute_reduction_change=movement))
    return dict(audited_episode_pairs=len(f), scope_by_actuator=scopes, discrepancies=discrepancies,
                effect_changes=effects, envelopes=envelopes, full_cells_required=sorted(affected),
                envelope_interpretation='Maximum old/new empirical discrepancy, not a certified error bound')


def qualify_numerics(root: Path, request: dict, primary: dict, finer: dict) -> dict:
    """Full old coverage cannot replace the new paths' own numerical audit."""
    from .io import read_json
    parent = read_json(Path(root)/'reports/control_confirmation_numerical.json')
    if (parent.get('state') != 'qualified' or parent.get('scope') != 'full'
            or parent.get('required_actions') != []):
        raise ValueError('The original full numerical qualification remains mandatory')
    old = {key: parent['envelopes'][key] for key in CRITERIA}
    comparison = compare_precision_refinement(primary['appended'], finer['episodes'], old, look_index=2)
    costs = aggregate_pairs(primary['original'], primary['appended'], request)
    provisional = evaluate_precision(costs, look_index=2, envelopes=comparison['envelopes'])
    near = sorted(r['actuator'] for r in provisional['criteria'] if r['status'] not in ('lower_cost', 'higher_cost'))
    affected = set(comparison['full_cells_required'])
    full = sorted(a for a in ACTUATORS if comparison['scope_by_actuator'][a] == 'subset'
                  and (a in affected or a in near))
    further = sorted(a for a in affected if comparison['scope_by_actuator'][a] == 'full')
    actions = []
    if full: actions.append(dict(branch='physics_full', actuators=full))
    if further: actions.append(dict(branch='physics_further_refinement', actuators=further))
    return dict(schema_version=1, state='additional_numerical_evidence_required' if actions else 'qualified',
        proposed_look_index=2, comparison=comparison, near_boundary_actuators=near,
        envelopes=comparison['envelopes'], required_actions=actions, statistical_look_reserved=False,
        scientific_decision='not_evaluated',
        interpretation='Old/new empirical plant discrepancies; the unchanged NODE comparisons retain their original envelope')


def _verify_hashes(root, hashes):
    from .io import safe_path, sha256_file
    if not hashes:
        raise ValueError('Explicit evidence hash closure required')
    for name, digest in hashes.items():
        if sha256_file(safe_path(root, name)) != digest:
            raise ValueError('Changed closed precision evidence: '+name)


def _load_evidence(root, request, selection):
    """Open exact complete manifests, independently reconstruct cost, then close hashes."""
    from .control_precision import REQUEST_PATH, _observe_episode, verify_episode_registry
    from .control_audit_closure import load_evidence as load_original
    from .io import read_json, safe_path, sha256_file
    if set(selection)-{'finer_registry_paths'}:
        raise ValueError('Only explicit finer registry selections are allowed')
    original = load_original(root, scope='full')
    hashes = dict(request['input_hashes'])
    names = [REQUEST_PATH, 'poc/control_precision.py', 'poc/control_precision_result.py',
             'config/control_precision_nominal_jobs.json', 'reports/control_precision_nominal.json']
    for name in names: hashes[name] = sha256_file(safe_path(root, name))

    def capture(entry):
        path = safe_path(root, entry['manifest'])
        hashes[entry['manifest']] = entry['manifest_sha256']
        marker = path.parent/'COMPLETED'
        hashes[str(marker.relative_to(root))] = sha256_file(marker)
        record = read_json(path)
        for artifact in record['produced_files']:
            name, digest = artifact['path'], artifact['sha256']
            if name in hashes and hashes[name] != digest:
                raise ValueError('Conflicting captured precision artifact')
            hashes[name] = digest
        for name, digest in record['job_inputs'].get('dependencies', {}).items():
            if name in hashes and hashes[name] != digest:
                raise ValueError('Conflicting captured precision dependency')
            hashes[name] = digest

    old_rows = []
    for entry in read_json(root/'reports/control_confirmation_batch.json')['episodes']:
        row = entry['summary']
        if row['model_id'] in MODELS and row['model_partition_ns'] == [5_000_000]*8:
            capture(entry); old_rows.append(row)

    def batch(registry_name, batch_name, kind):
        registry_path, batch_path = safe_path(root, registry_name), safe_path(root, batch_name)
        registry, receipt = read_json(registry_path), read_json(batch_path)
        verify_episode_registry(root, registry)
        if registry['kind'] != kind or registry['registry_path'] != registry_name:
            raise ValueError('Changed explicit precision registry identity')
        jobs = {r['cell_id']: r for r in registry['jobs']}
        entries = receipt.get('episodes', [])
        if (receipt.get('state') != 'complete' or receipt.get('kind') != kind
                or receipt.get('registry_sha256') != sha256_file(registry_path)
                or receipt.get('expected_episodes') != len(jobs) or receipt.get('completed_episodes') != len(jobs)
                or len(entries) != len(jobs) or len({r['cell_id'] for r in entries}) != len(jobs)
                or {r['cell_id'] for r in entries} != set(jobs)):
            raise ValueError('Complete hash-compatible precision batch required')
        checked = []
        for entry in entries:
            path = safe_path(root, entry['manifest'])
            if sha256_file(path) != entry['manifest_sha256']:
                raise ValueError('Changed precision episode manifest')
            observed = _observe_episode(root, jobs[entry['cell_id']], path)
            if any(entry.get(k) != value for k, value in observed.items()):
                raise ValueError('Saved precision receipt differs from verified raw evidence')
            capture(observed); checked.append(observed)
        hashes[registry_name], hashes[batch_name] = sha256_file(registry_path), sha256_file(batch_path)
        hashes.update(registry['input_hashes'])
        for job in registry['jobs']: hashes[job['path']] = job['sha256']
        return checked

    primary_entries = batch('config/control_precision_control_jobs.json', 'reports/control_precision_primary.json', 'primary')
    fine_entries = batch('config/control_precision_finer_jobs.json', 'reports/control_precision_finer.json', 'finer')
    extra_names = selection.get('finer_registry_paths', [])
    if not isinstance(extra_names, list) or len(set(extra_names)) != len(extra_names):
        raise ValueError('Explicit unique finer registry paths required')
    indexed = {entry['cell_id']: entry for entry in fine_entries}
    for name in extra_names:
        if name == 'config/control_precision_finer_jobs.json':
            raise ValueError('The initial finer registry is already included')
        registry = read_json(safe_path(root, name))
        if registry.get('scope') != 'full':
            raise ValueError('Only explicit full affected-cell expansions may augment the initial audit')
        suffix = Path(name).stem.removeprefix('control_precision_finer_jobs')
        for entry in batch(name, f'reports/control_precision_finer{suffix}.json', 'finer'):
            cell = entry['cell_id']
            if cell in indexed and any(entry[k] != indexed[cell][k] for k in (
                    'manifest', 'manifest_sha256', 'summary', 'raw_audit', 'resources', 'request_sha256')):
                raise ValueError('Conflicting finer episode completions')
            indexed[cell] = entry
    fine_entries = list(indexed.values())
    _verify_hashes(root, hashes)
    return dict(design=original['design'], original_result=read_json(root/'reports/control_confirmation_result.json'),
        primary=dict(original=old_rows, appended=[r['summary'] for r in primary_entries]),
        finer=dict(episodes=[r['summary'] for r in fine_entries]), input_hashes=hashes,
        resources=dict(primary=[r['resources'] for r in primary_entries], finer=[r['resources'] for r in fine_entries]))


def _render(result):
    lines = ['# Control precision: corrected head versus MLP at 5 ms', '',
        'The two updated comparisons use ten independent training seeds with twenty paired tasks each. '
        'Six other comparisons retain five seeds and their original look-1 evidence.', '',
        '| Plant | Evidence | MLP cost | Head cost | MLP minus head | Lower bound | Upper bound |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    def number(x): return 'unresolved' if x is None else f'{x:.9g}'
    for row in result['criteria']:
        lines.append('| '+' | '.join([row['actuator'], row['status'], number(row['baseline_mean_cost']),
            number(row['method_mean_cost']), number(row['point_difference']), number(row['bounds']['lower']),
            number(row['bounds']['upper'])])+' |')
    lines.extend(['', '| Cached criterion | Evidence | Training seeds | Source look |',
                  '| --- | --- | --- | --- |'])
    for row in result['cached_criteria']:
        lines.append(f"| {row['criterion_id']} | {row['status']} | {row['n_training_seeds']} | {row['source_look_index']} |")
    lines.extend(['', 'Positive differences favor the head. Ratios are descriptive; signed bounds include '
        'paired training-seed uncertainty and measured numerical envelopes. Unresolved or influential '
        'clusters do not establish a win.', '',
        'Control look 2 spends alpha 0.0125, with eight comparisons and two directional tails '
        '(0.00078125 per tail); common 20,000 training-seed bootstrap draws use [9102,2].', '',
        'This extension does not establish nominal noninferiority, NODE equivalence, full-suite superiority, '
        'actuator-rate or noise robustness, or a physical-robot result. The original report and NODE solver '
        'limitations remain attached to the six cached comparisons.', '',
        f"All 800 paired primary episodes remain included, with {result['fallback_intervals']} fallback intervals.",
        'Resources below cover completed new primary/finer episode producers; training, prior experiments '
        'and failed/interrupted attempts are separate.', ''])
    return '\n'.join(lines)


def write_report(root: Path, request: dict, evidence: dict) -> dict:
    """Validate actual files before any sign decision or statistical look."""
    import copy
    from .control_precision import verify_request
    from .control_confirmation import validate_design
    from .control_utility import reserve_control_look
    from .io import atomic_bytes, atomic_json, job_key, job_lock, read_json, safe_path, sha256_file
    root = Path(root).resolve()
    verify_request(root, request)
    with job_lock(job_key({'control_precision_report': request['request_id']}), root):
        closed = _load_evidence(root, request, evidence)
        hashes = closed['input_hashes']; _verify_hashes(root, hashes)
        design, parent = closed['design'], closed['original_result']
        validate_design(design)
        expected = {r['criterion_id'] for r in design['criteria']}
        if (parent.get('look_index') != 1 or parent.get('family_size') != 8
                or len(parent.get('criteria', [])) != 8 or {r['criterion_id'] for r in parent['criteria']} != expected
                or any(r['n_training_seeds'] != 5 for r in parent['criteria'])):
            raise ValueError('The complete original eight comparisons must remain at look 1')
        costs = aggregate_pairs(**closed['primary'], request=request)
        qualification = qualify_numerics(root, request, closed['primary'], closed['finer'])
        qualification.update(input_hashes=hashes, evaluator_sha256=sha256_file(Path(__file__)))
        numerical_path = root/'reports/control_precision_numerical.json'
        if qualification['state'] != 'qualified':
            atomic_json(numerical_path, qualification)
            return qualification
        result = evaluate_precision(costs, look_index=2, envelopes=qualification['envelopes'])
        cached = [dict(copy.deepcopy(r), source_look_index=1) for r in parent['criteria'] if r['criterion_id'] not in CRITERIA]
        if len(cached) != 6:
            raise ValueError('Exactly six unchanged look-1 comparisons required')
        resources = {}
        expected_counts = {'primary': 400, 'finer': qualification['comparison']['audited_episode_pairs']}
        for group, count in expected_counts.items():
            records = closed['resources'][group]
            if len(records) != count or any(type(r[k]) not in (int, float) or not math.isfinite(r[k]) or r[k] < 0
                    for r in records for k in ('wall_seconds', 'cpu_seconds', 'peak_rss_bytes', 'disk_bytes')):
                raise ValueError('Complete finite measured precision resources required')
            resources[group] = dict(n_episodes=count, wall_seconds=math.fsum(r['wall_seconds'] for r in records),
                cpu_seconds=math.fsum(r['cpu_seconds'] for r in records),
                peak_rss_bytes=max(r['peak_rss_bytes'] for r in records), artifact_bytes=sum(r['disk_bytes'] for r in records))
        result.update(schema_version=1, protocol_id=design['protocol_id'], status='limited_precision_results',
            cached_criteria=cached, seed_costs=costs, resources=resources, input_hashes=hashes,
            numerical_qualification=qualification, original_ptco_terminal_verdict=None,
            fallback_intervals=sum(sum(r['fallback_by_seed']) for r in costs['rows']))
        text = _render(result)
        _verify_hashes(root, hashes)
        atomic_json(numerical_path, qualification)
        identity = dict(registry_sha256=sha256_file(root/'config/control_precision_control_jobs.json'),
            batch_sha256=sha256_file(root/'reports/control_precision_primary.json'),
            numerical_sha256=sha256_file(numerical_path))
        result['evidence_identity'] = identity
        ledgers = [read_json(p) for p in sorted((root/'decisions/control_utility_looks').glob('look-*.json'))]
        if (len(ledgers) not in (1, 2) or ledgers[0].get('look_index') != 1
                or len(ledgers) == 2 and (ledgers[1].get('look_index') != 2
                    or ledgers[1].get('evidence_identity') != identity)):
            raise ValueError('Prospective look 2 conflicts with the existing control look ledger')
        path = root/'reports/control_precision_result.json'
        if path.exists() and read_json(path) != result:
            raise ValueError('Changed evidence requires an explicitly versioned precision request, never an overwritten look')
        _verify_hashes(root, hashes)
        archive = root/f'decisions/control_utility_numerics/{identity["numerical_sha256"]}.json'
        if archive.exists() and sha256_file(archive) != identity['numerical_sha256']:
            raise ValueError('Changed immutable precision numerical evidence')
        if not archive.exists(): atomic_bytes(archive, numerical_path.read_bytes())
        look = reserve_control_look(root, design, identity)
        if look['look_index'] != 2:
            raise ValueError('Prospective precision must use the existing ledger look 2')
        atomic_json(path, result)
        atomic_bytes(root/'reports/control_precision_result.md', text.encode('utf-8'))
        return result


def main(argv=None):
    import argparse
    from .control_precision import REQUEST_PATH
    from .io import read_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--report', action='store_true', required=True)
    parser.add_argument('--finer-registry', action='append', default=[], help='Exact full affected-cell registry; may repeat')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    result = write_report(root, read_json(root/REQUEST_PATH), dict(finer_registry_paths=args.finer_registry))
    print(result.get('status', result.get('state')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
