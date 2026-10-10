import copy
import json

import numpy as np
import pytest

from poc.affine_presentation import prepare_clips, render_document, validate_replay_episode


def clips():
    return [dict(actuator=a, model=m, hold=r, partition=h, task=t,
                 time_s=np.arange(161).astype(float).tolist(),
                 q=np.zeros((161, 2)).tolist(), target=[.3, -.2],
                 cost=1., planner_seconds=.2)
            for a in ('direct', 'lagged')
            for m in ('affine_flow', 'power_correct', 'mlp_h', 'node_generic',
                      'node_structured', 'physics_fit')
            for r in (40, 10) for h in (40, 5) for t in range(10)]


def valid_clips():
    value = clips()
    for row in value:
        row['time_s'] = (np.arange(161) / 100).tolist()
    return value


def test_replay_requires_complete_unique_paired_coverage():
    value = valid_clips()
    assert len(prepare_clips(value)) == 480
    for damaged in (value[:-1], value + [copy.deepcopy(value[0])]):
        with pytest.raises(ValueError, match='coverage'):
            prepare_clips(damaged)


def test_replay_checks_physical_time_finite_states_and_same_task():
    value = valid_clips()
    for key, changed in (('time_s', np.arange(161).tolist()),
                         ('q', np.zeros((160, 2)).tolist()),
                         ('target', [.4, -.2]),
                         ('cost', float('nan'))):
        damaged = copy.deepcopy(value)
        damaged[0][key] = changed
        with pytest.raises(ValueError):
            prepare_clips(damaged)


def test_replay_payload_contains_measured_values_without_hidden_simulation():
    value = valid_clips()
    value[0]['cost'] = 1.2345
    document = render_document(value)
    start = document.index('<script type="application/json" id="clips">')
    start = document.index('>', start) + 1
    end = document.index('</script>', start)
    payload = json.loads(document[start:end])
    assert payload == prepare_clips(value)
    measured = next(c for c in payload if c['actuator'] == 'direct' and c['model'] == 'affine_flow'
                    and c['hold'] == 40 and c['partition'] == 40 and c['task'] == 0)
    assert measured['cost'] == 1.2345
    assert measured['time_s'][-1] == 1.6


def test_replay_does_not_trust_edited_report_latency_or_model_identity():
    saved=dict(model_id='affine_flow',actuator='lagged',seed=9000,episode_id=0,
               initial_state=[0.]*6,target=[.2,-.1],model_partition_ns=[5000000]*8,
               actuator_interval_ns=40000000,realized_cost=.8,planner_latency_median_seconds=.5)
    validate_replay_episode(saved,saved)
    for key,value in [('planner_latency_median_seconds',.1),('model_id','node_structured')]:
        changed={**saved,key:value}
        with pytest.raises(ValueError,match='source summary'):
            validate_replay_episode(changed,saved)
