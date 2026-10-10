"""Ten-cluster inference retains pairing, numerical uncertainty and old looks."""
import copy
from pathlib import Path

import numpy as np
import pytest

from poc.io import read_json


def precision_request():
    return read_json(Path('config/revisions/MX-control-precision-0001.json'))


def episodes(seeds):
    from poc.control_episode import episode_conditions
    rows = []
    for actuator in ['direct', 'lagged']:
        for model in ['mlp_dt', 'power_correct']:
            for seed in seeds:
                for episode in range(20):
                    task = episode_conditions(seed, episode, actuator, 'confirmatory')
                    rows.append(dict(phase='confirmatory', complete=True, actuator=actuator, model_id=model,
                        seed=seed, episode_id=episode, condition='clean', model_partition_ns=[5_000_000]*8,
                        actuator_interval_ns=40_000_000, physics_step_ns=250_000, power_mode=None,
                        checkpoint_hash='a'*64, initial_state=task['initial_state'].tolist(), target=task['target'].tolist(),
                        seed_namespaces={'cem': task['cem_seed_entropy'], 'disturbance': task['disturbance_seed_entropy']},
                        realized_cost=3.+seed*.1+episode*.01-(1.+seed*.003 if model == 'power_correct' else 0.),
                        fallback_count=1 if episode == 3 else 0, success=episode == 4))
    return rows


def envelopes(value=0.):
    return {f'utility_control/{a}/mlp_dt/h5000000': dict(lower=value, upper=value) for a in ['direct', 'lagged']}


def test_paired_costs_keep_ten_training_clusters_and_all_fallbacks():
    from poc.control_precision_result import aggregate_pairs
    costs = aggregate_pairs(episodes(range(5)), episodes(range(5, 10)), precision_request())
    assert costs['seed_ids'] == list(range(10)) and costs['episodes_per_seed'] == 20
    assert costs['n_episodes'] == 800 and len(costs['rows']) == 4
    head = next(r for r in costs['rows'] if r['actuator'] == 'direct' and r['model_id'] == 'power_correct')
    assert head['cost_by_seed'] == pytest.approx([2.095+s*.097 for s in range(10)])
    assert head['fallback_by_seed'] == [1]*10 and head['success_by_seed'] == [1]*10


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'fallback', 'namespace', 'target', 'clock',
                                     'seed_replacement', 'checkpoint', 'nonfinite', 'partial'])
def test_no_partial_or_stale_ten_seed_sign(mutation):
    from poc.control_precision_result import aggregate_pairs
    original, appended = episodes(range(5)), episodes(range(5, 10))
    if mutation == 'missing': appended.pop()
    elif mutation == 'duplicate': appended[-1] = copy.deepcopy(appended[0])
    elif mutation == 'fallback': appended[0].pop('fallback_count')
    elif mutation == 'namespace': appended[0]['seed_namespaces']['cem'][0] = 7101
    elif mutation == 'target': appended[0]['target'][0] += .001
    elif mutation == 'clock': appended[0]['model_partition_ns'] = [5_000_000]*7
    elif mutation == 'seed_replacement': appended[0]['seed'] = 9000
    elif mutation == 'checkpoint': appended[0]['checkpoint_hash'] = None
    elif mutation == 'nonfinite': appended[0]['realized_cost'] = float('nan')
    else: appended[0]['complete'] = False
    with pytest.raises(ValueError): aggregate_pairs(original, appended, precision_request())


def test_fixed_tail_cluster_bootstrap_and_outward_envelope():
    from poc.control_precision_result import aggregate_pairs, evaluate_precision
    costs = aggregate_pairs(episodes(range(5)), episodes(range(5, 10)), precision_request())
    first = evaluate_precision(costs, look_index=2, envelopes=envelopes())
    assert first['look_alpha'] == .0125 and first['tail_alpha'] == .00078125
    assert first['bootstrap_seed_entropy'] == [9102, 2] and first['family_size'] == 8
    assert len(first['criteria']) == 2
    assert all(r['status'] == 'lower_cost' and r['n_training_seeds'] == 10 for r in first['criteria'])
    assert len({r['bounds']['bootstrap_indices_sha256'] for r in first['criteria']}) == 1
    assert all(r['bounds']['rule'] == 't_rare_tail_with_central_bootstrap' for r in first['criteria'])
    wider = evaluate_precision(costs, look_index=2, envelopes=envelopes(2.))
    assert all(r['status'] == 'unresolved' for r in wider['criteria'])
    assert wider['criteria'][0]['bounds']['lower'] == pytest.approx(first['criteria'][0]['bounds']['lower']-2.)
    with pytest.raises(ValueError): evaluate_precision(costs, look_index=2, envelopes={})


def test_constant_or_influential_clusters_cannot_manufacture_certainty():
    from poc.control_precision_result import aggregate_pairs, evaluate_precision
    original, appended = episodes(range(5)), episodes(range(5, 10))
    for row in original+appended: row['realized_cost'] = 1. if row['model_id'] == 'power_correct' else 2.
    result = evaluate_precision(aggregate_pairs(original, appended, precision_request()), look_index=2, envelopes=envelopes())
    assert all(r['status'] == 'unresolved_variance' for r in result['criteria'])
    for row in original+appended:
        row['realized_cost'] = 1. if row['model_id'] == 'power_correct' else 2.+row['seed']*.003
        if row['seed'] == 9 and row['model_id'] == 'mlp_dt': row['realized_cost'] = 100.
    result = evaluate_precision(aggregate_pairs(original, appended, precision_request()), look_index=2, envelopes=envelopes())
    assert all(r['status'] == 'unresolved_influence' for r in result['criteria'])


def test_new_refinement_measures_discrepancy_and_expands_changed_classification():
    from poc.control_precision_result import compare_precision_refinement
    primary = episodes(range(5, 10))
    finer = [dict(r, physics_step_ns=125_000) for r in primary if r['episode_id'] < 5]
    result = compare_precision_refinement(primary, finer, envelopes(), look_index=2)
    assert result['audited_episode_pairs'] == 100
    assert result['envelopes'] == envelopes()
    # This uses the same raw cost twice only as an exact numeric fixture; the
    # actual consumer verifies the two distinct complete physics manifests.
    finer[0]['realized_cost'] += .001
    result = compare_precision_refinement(primary, finer, envelopes(), look_index=2)
    assert result['envelopes']['utility_control/direct/mlp_dt/h5000000']['upper'] == pytest.approx(.001)
    finer[0]['success'] = not finer[0]['success']
    result = compare_precision_refinement(primary, finer, envelopes(), look_index=2)
    assert 'direct' in result['full_cells_required']
    with pytest.raises(ValueError): compare_precision_refinement(primary, finer[:-1], envelopes(), look_index=2)


def stage_numerics(root):
    from poc.io import atomic_json
    atomic_json(root/'reports/control_confirmation_numerical.json', dict(state='qualified', scope='full',
        required_actions=[], envelopes=envelopes(.00001)))


def test_outward_envelope_and_near_boundary_expansion(tmp_path):
    from poc.control_precision_result import qualify_numerics
    stage_numerics(tmp_path)
    original, appended = episodes(range(5)), episodes(range(5, 10))
    primary = dict(original=original, appended=appended)
    finer = dict(episodes=[dict(r, physics_step_ns=125_000) for r in appended if r['episode_id'] < 5])
    result = qualify_numerics(tmp_path, precision_request(), primary, finer)
    assert result['state'] == 'qualified' and result['statistical_look_reserved'] is False
    assert result['envelopes'] == envelopes(.00001)
    # Weak but variable evidence triggers full new-cell convergence, even when
    # the already-completed old cohort had full stable physics coverage.
    for row in original+appended:
        row['realized_cost'] = 1.+(.01*row['seed'] if row['model_id'] == 'mlp_dt' else 0.)
    for row in finer['episodes']:
        row['realized_cost'] = 1.+(.01*row['seed'] if row['model_id'] == 'mlp_dt' else 0.)
    result = qualify_numerics(tmp_path, precision_request(), primary, finer)
    assert result['state'] == 'additional_numerical_evidence_required'
    assert result['near_boundary_actuators'] == ['direct', 'lagged']
    assert result['required_actions'] == [dict(branch='physics_full', actuators=['direct', 'lagged'])]
    finer['episodes'] = [dict(r, physics_step_ns=125_000) for r in appended]
    result = qualify_numerics(tmp_path, precision_request(), primary, finer)
    assert result['state'] == 'qualified'
    finer['episodes'][0]['success'] = not finer['episodes'][0]['success']
    result = qualify_numerics(tmp_path, precision_request(), primary, finer)
    assert result['required_actions'] == [dict(branch='physics_further_refinement', actuators=['direct'])]


def staged_closed_report(root, monkeypatch):
    from poc.io import atomic_json, sha256_file
    from poc.control_utility import reserve_control_look
    import poc.control_precision_result as module
    d = read_json(Path('config/control_confirmation_design.json'))
    old = read_json(Path('reports/control_confirmation_result.json'))
    for row in old['criteria']:
        row['look_index'] = 1
    stage_numerics(root)
    for name, value in [('config/control_confirmation_design.json', d),
                        ('reports/control_confirmation_result.json', old),
                        ('config/control_precision_control_jobs.json', dict(fixture='registry')),
                        ('reports/control_precision_primary.json', dict(fixture='batch')),
                        ('reports/utility_followup.json', dict(state='original-ready'))]:
        atomic_json(root/name, value)
    reserve_control_look(root, d, dict(registry_sha256='a'*64, batch_sha256='b'*64, numerical_sha256='c'*64))
    old_primary, appended = episodes(range(5)), episodes(range(5, 10))
    evidence = dict(design=d, original_result=old, primary=dict(original=old_primary, appended=appended),
        finer=dict(episodes=[dict(r, physics_step_ns=125_000) for r in appended if r['episode_id'] < 5]),
        resources={group: [dict(wall_seconds=1., cpu_seconds=.9, peak_rss_bytes=100, disk_bytes=50)
                          for _ in range(count)] for group, count in [('primary', 400), ('finer', 100)]},
        input_hashes={name: sha256_file(root/name) for name in (
            'config/control_confirmation_design.json', 'reports/control_confirmation_result.json',
            'config/control_precision_control_jobs.json', 'reports/control_precision_primary.json',
            'reports/control_confirmation_numerical.json', 'reports/utility_followup.json')})
    # The file-backed loader's heavy raw-manifest boundary is independently
    # exercised by the producer integration fixture. These report tests inject
    # its closed data and exercise actual aggregation, hashing and look ledger.
    monkeypatch.setattr(module, '_load_evidence', lambda *args, **kwargs: evidence)
    monkeypatch.setattr('poc.control_precision.verify_request', lambda root, request: request)
    return evidence


def test_report_is_idempotent_and_six_old_cells_stay_at_look1(tmp_path, monkeypatch):
    from poc.control_precision_result import write_report
    from poc.io import sha256_file
    evidence = staged_closed_report(tmp_path, monkeypatch)
    originals = {name: sha256_file(tmp_path/name) for name in (
        'reports/control_confirmation_result.json', 'reports/utility_followup.json',
        'decisions/control_utility_looks/look-00001.json')}
    first = write_report(tmp_path, precision_request(), {})
    assert first['look_index'] == 2 and first['family_size'] == 8
    assert len(first['criteria']) == 2 and len(first['cached_criteria']) == 6
    assert all(r['n_training_seeds'] == 5 and r['source_look_index'] == 1 for r in first['cached_criteria'])
    assert all(r['n_training_seeds'] == 10 for r in first['criteria'])
    cached = {r['criterion_id']: r for r in first['cached_criteria']}
    for old in evidence['original_result']['criteria']:
        if old['criterion_id'] in cached:
            assert {k: v for k, v in cached[old['criterion_id']].items() if k != 'source_look_index'} == old
    look2 = read_json(tmp_path/'decisions/control_utility_looks/look-00002.json')
    assert look2['cumulative_alpha'] == .0375 and look2['tail_alpha'] == .00078125
    second = write_report(tmp_path, precision_request(), {})
    assert second == first
    assert len(list((tmp_path/'decisions/control_utility_looks').glob('look-*.json'))) == 2
    assert all(sha256_file(tmp_path/name) == digest for name, digest in originals.items())


@pytest.mark.parametrize('mutation', ['hash', 'coverage', 'numerical', 'resource_count', 'nonfinite_resource'])
def test_report_cannot_spend_a_look_from_unverified_or_partial_evidence(tmp_path, monkeypatch, mutation):
    from poc.control_precision_result import write_report
    from poc.io import atomic_json
    evidence = staged_closed_report(tmp_path, monkeypatch)
    if mutation == 'hash': atomic_json(tmp_path/'reports/control_precision_primary.json', dict(changed=True))
    elif mutation == 'coverage': evidence['primary']['appended'].pop()
    elif mutation == 'numerical': evidence['finer']['episodes'].pop()
    elif mutation == 'resource_count': evidence['resources']['primary'].pop()
    else: evidence['resources']['primary'][0]['cpu_seconds'] = float('nan')
    with pytest.raises(ValueError): write_report(tmp_path, precision_request(), {})
    assert not (tmp_path/'decisions/control_utility_looks/look-00002.json').exists()
    assert not (tmp_path/'reports/control_precision_result.json').exists()


def test_changed_existing_look_cannot_spend_a_third_look_before_rejection(tmp_path, monkeypatch):
    from poc.control_precision_result import write_report
    from poc.control_utility import reserve_control_look
    evidence = staged_closed_report(tmp_path, monkeypatch)
    reserve_control_look(tmp_path, evidence['design'], dict(registry_sha256='d'*64,
        batch_sha256='e'*64, numerical_sha256='f'*64))
    with pytest.raises(ValueError, match='look'):
        write_report(tmp_path, precision_request(), {})
    assert not (tmp_path/'decisions/control_utility_looks/look-00003.json').exists()


def test_report_renders_scope_seed_counts_and_failures(tmp_path, monkeypatch):
    from poc.control_precision_result import write_report
    staged_closed_report(tmp_path, monkeypatch)
    write_report(tmp_path, precision_request(), {})
    text = (tmp_path/'reports/control_precision_result.md').read_text()
    assert 'ten independent training seeds' in text and 'Six other comparisons retain five seeds' in text
    assert '800 paired primary episodes' in text and '40 fallback intervals.' in text
    assert 'nominal noninferiority, NODE equivalence' in text
    assert 'utility_control/direct/node_generic/h5000000' in text
    assert '| unresolved | 5 | 1 |' in text


def test_file_loader_rejects_partial_primary_before_any_look(tmp_path):
    from poc.control_precision_result import main
    from poc.io import atomic_json
    atomic_json(tmp_path/'config/revisions/MX-control-precision-0001.json', precision_request())
    with pytest.raises((ValueError, FileNotFoundError)):
        main(['--root', str(tmp_path), '--report'])
    assert not list((tmp_path/'decisions/control_utility_looks').glob('look-*.json'))
