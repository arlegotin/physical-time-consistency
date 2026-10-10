import copy
import pytest

from poc.planner_qualification import qualification, verify_planner_transition, verify_analytic_reports
from poc.io import atomic_json, sha256_file


def fixture():
    rows, audits = [], []
    for actuator in ('direct', 'lagged'):
        for episode in range(16):
            for model in ('zero_command', 'physics_fit', 'sim_oracle'):
                rows.append(dict(actuator=actuator, episode_id=episode, model_id=model,
                    phase='calibration', seed=9000, condition='clean', complete=True,
                    fallback_count=0, actuator_interval_ns=40_000_000,
                    model_partition_ns=[40_000_000], physics_step_ns=250_000,
                    realized_cost=1.0 if model == 'zero_command' else .2,
                    initial_state=[0.] * (4 if actuator == 'direct' else 6), target=[.5, .5]))
            audits.append(dict(actuator=actuator, episode_id=episode,
                ranking=dict(qualified=True, resolved_inversions=0, candidate_count=512)))
    planner = dict(settings=dict(candidate_count=128, iterations=4, elite_count=16,
        initial_std=1., std_floor=.1), analytic_qualification=dict(qualified=True),
        final_common_budget_frozen=False, robot_reference_qualification='not_run')
    return planner, rows, audits


def test_complete_independent_qualification_authorizes_only_learned_development():
    planner, rows, audits = fixture()
    result = qualification(planner, rows, audits)
    assert result['qualified']
    assert result['learned_model_control'] == 'not_run'
    assert result['terminal_verdict'] is None


@pytest.mark.parametrize('mutation', ['missing_episode', 'duplicate_audit', 'inversion',
    'bad_reference', 'task_changed', 'fallback', 'budget', 'analytic'])
def test_missing_conflicting_or_unqualified_evidence_cannot_freeze_planner(mutation):
    planner, rows, audits = fixture()
    if mutation == 'missing_episode': rows.pop()
    if mutation == 'duplicate_audit': audits[-1] = copy.deepcopy(audits[0])
    if mutation == 'inversion': audits[0]['ranking']['resolved_inversions'] = 1
    if mutation == 'bad_reference':
        for row in rows:
            if row['actuator'] == 'direct' and row['model_id'] == 'physics_fit':
                row['realized_cost'] = 2.
    if mutation == 'task_changed': rows[1]['target'][0] = .6
    if mutation == 'fallback': rows[1]['fallback_count'] = 1
    if mutation == 'budget': audits[0]['ranking']['candidate_count'] = 256
    if mutation == 'analytic': planner['analytic_qualification']['qualified'] = False
    with pytest.raises(ValueError): qualification(planner, rows, audits)


def test_only_evidenced_qualification_metadata_can_change():
    prior = dict(status='analytic_qualified_robot_pending', settings={'candidate_count': 128},
        analytic_qualification={'qualified': True}, final_common_budget_frozen=False,
        robot_reference_qualification='not_run', recorded_utc='old', qualification_evidence=[])
    entry = dict(path='reports/robot_planner_qualification.json', sha256='1' * 64)
    current = copy.deepcopy(prior)
    current.update(status='qualified', final_common_budget_frozen=True,
        robot_reference_qualification='qualified', recorded_utc='new', qualification_evidence=[entry])
    verify_planner_transition(prior, current, entry)
    current['settings']['candidate_count'] = 256
    with pytest.raises(ValueError): verify_planner_transition(prior, current, entry)


@pytest.mark.parametrize('mutation', [None, 'source', 'receipt', 'settings'])
def test_analytic_authorization_checks_reports_and_original_sources(tmp_path, mutation):
    from poc.mpc_calibration import qualification_summary
    rows = [dict(episode_id=e, stream_id=s, status='ok', achieved_headroom_fraction=.8,
                 zero_cost=10., optimal_cost=2., cem_cost=3.6) for s in range(5) for e in range(16)]
    source = tmp_path / 'source.py'
    source.write_text('unchanged\n')
    settings = dict(candidate_count=128, iterations=4, elite_count=16, initial_std=1., std_floor=.1)
    summary = qualification_summary(rows)
    report = dict(completion_status='complete', settings=settings, qualification=summary, rows=rows,
                  source_hashes={'source.py': sha256_file(source)})
    atomic_json(tmp_path / 'report.json', report)
    planner = dict(settings=copy.deepcopy(settings), analytic_qualification=summary,
                   qualification_evidence=[dict(path='report.json', sha256=sha256_file(tmp_path / 'report.json'))])
    if mutation == 'source': source.write_text('changed\n')
    if mutation == 'receipt': atomic_json(tmp_path / 'report.json', {'damaged': True})
    if mutation == 'settings': planner['settings']['iterations'] = 8
    if mutation is None: assert 'report.json' in verify_analytic_reports(tmp_path, planner)
    else:
        with pytest.raises(ValueError): verify_analytic_reports(tmp_path, planner)
