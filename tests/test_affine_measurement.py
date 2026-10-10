"""Complete paired measurements retain missingness, failures and physical costs."""
import numpy as np
import pytest


def test_prediction_summary_rejects_a_missing_probe_and_retains_signed_error():
    from poc.affine_measurement import summarize_prediction
    coordinates = dict(probe_ids=[0, 1], strata=['cold', 'warm'], horizons_ns=[5_000_000])
    truth = np.zeros((2, 1, 4))
    raw = dict(probe_ids=np.array([0,1]), strata=np.array(['cold','warm']), horizons_ns=np.array([5_000_000]),
               truth_zero=truth.copy(), truth_command=truth.copy(), model_zero=truth.copy(),
               model_command=truth.copy(), model_composed=truth.copy())
    raw['model_zero'][:,:,0] = .3
    raw['model_command'][:,:,0] = .1
    raw['model_composed'][:,:,0] = .2
    rows = summarize_prediction(raw, coordinates, 'direct')
    q = next(r for r in rows if r['component']=='q' and r['stratum']=='cold')
    assert q['drift_error_mean'] == pytest.approx(.3)
    assert q['command_error_mean'] == pytest.approx(.2)
    assert q['total_error_mean'] == pytest.approx(.1)
    assert q['composed_error_mean'] == pytest.approx(.2)
    bad = {k:v[:1] if k != 'horizons_ns' else v for k,v in raw.items()}
    with pytest.raises(ValueError, match='coordinate'):
        summarize_prediction(bad, coordinates, 'direct')


def rows_for_pair():
    return [dict(actuator='direct', model_id=model, actuator_interval_ns=40_000_000,
                 model_partition_ns=[5_000_000]*8, episode_id=0, complete=True, phase='exploratory', seed=9000,
                 realized_cost=cost, planner_latency_median_seconds=timer,
                 planner_latency_p95_seconds=timer*2, q_rmse_rad=.1, terminal_q_error_rad=.2,
                 neural_rows=100, fallback_count=fallback, success=False,
                 initial_state=[0.,0.,0.,0.], target=[.4,-.2])
            for model,cost,timer,fallback in (('affine_flow',2.,.3,1),('node_structured',4.,.6,0))]


def test_control_summary_retains_fallback_episode_and_pairs_costs_and_latency():
    from poc.affine_measurement import summarize_control
    rows = rows_for_pair()
    expected = [('direct',r['model_id'],40_000_000,5_000_000,0) for r in rows]
    result = summarize_control(rows, expected)
    comparison = result['comparisons'][0]
    assert comparison['cost_reduction'] == pytest.approx(.5)
    assert comparison['planning_time_reduction'] == pytest.approx(.5)
    assert comparison['method_fallback_count'] == 1
    assert comparison['paired_cost_differences'] == [2.]
    with pytest.raises(ValueError, match='coverage'):
        summarize_control(rows[1:], expected)
    with pytest.raises(ValueError, match='coverage'):
        summarize_control(rows+[rows[0]], expected)


def test_control_pair_refuses_different_task_or_incomplete_episode():
    from poc.affine_measurement import summarize_control
    rows = rows_for_pair()
    expected = [('direct',r['model_id'],40_000_000,5_000_000,0) for r in rows]
    rows[0]['target'] = [.6, -.2]
    with pytest.raises(ValueError, match='paired task'):
        summarize_control(rows, expected)
    rows = rows_for_pair()
    rows[0]['complete'] = False
    with pytest.raises(ValueError, match='incomplete'):
        summarize_control(rows, expected)


def test_raw_episode_cost_mismatch_rejected(tmp_path):
    from poc.affine_measurement import recompute_episode_cost
    from poc.io import atomic_json
    import json
    np.savez_compressed(tmp_path/'true_trajectories.npz', anchors=np.zeros((40,4)), target=np.array([0.,0.]))
    segments = [dict(start_ns=i*40_000_000, duration_ns=40_000_000, command=[0.,0.]) for i in range(40)]
    (tmp_path/'action_segments.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in segments))
    atomic_json(tmp_path/'summary.json', dict(realized_cost=0.))
    assert recompute_episode_cost(tmp_path) == 0.
    atomic_json(tmp_path/'summary.json', dict(realized_cost=1.))
    with pytest.raises(ValueError, match='raw truth'):
        recompute_episode_cost(tmp_path)


def test_prediction_reaggregation_detects_changed_report_and_missing_rows():
    from poc.affine_measurement import summarize_prediction, validate_prediction_result
    coordinates=dict(probe_ids=[0,1],strata=['cold','warm'],horizons_ns=[5_000_000])
    raw={}
    for actor,dimension in (('direct',4),('lagged',6)):
        raw[actor]=dict(probe_ids=np.array([0,1]),strata=np.array(['cold','warm']),horizons_ns=np.array([5_000_000]))
        raw[actor].update({k:np.zeros((2,1,dimension)) for k in
                          ('truth_zero','truth_command','model_zero','model_command','model_composed')})
    freeze=dict(coordinates={a:coordinates for a in raw},expected_new_prediction_rows=4)
    rows=sum((summarize_prediction(r,coordinates,a) for a,r in raw.items()),[])
    summary=dict(new_model_rows=4,rows=rows)
    validate_prediction_result(raw,summary,freeze)
    summary['rows'][0]['total_error_mean']=1.
    with pytest.raises(ValueError,match='reaggregation'):
        validate_prediction_result(raw,summary,freeze)
