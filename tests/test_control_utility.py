"""Control claims use paired training clusters, complete workloads and numeric bounds."""
import copy
from pathlib import Path

import numpy as np
import pytest
from poc.io import read_json


def registered_design():
    return read_json(Path('config/control_confirmation_design.json'))


def private_episodes():
    from poc.control_episode import episode_conditions
    rows=[]
    for a in ('direct','lagged'):
        for m in ('mlp_dt','node_generic','power_correct'):
            for s in range(5):
                for h in (40_000_000,5_000_000):
                    for e in range(20):
                        task=episode_conditions(s,e,a,'confirmatory')
                        effect=(1.+s*.003) if m=='power_correct' else 0.
                        rows.append(dict(phase='confirmatory',complete=True,actuator=a,model_id=m,
                            seed=s,episode_id=e,condition='clean',model_partition_ns=[h]*(40_000_000//h),
                            actuator_interval_ns=40_000_000,physics_step_ns=250_000,power_mode=None,
                            initial_state=task['initial_state'].tolist(),target=task['target'].tolist(),
                            seed_namespaces={'cem':task['cem_seed_entropy'],'disturbance':task['disturbance_seed_entropy']},
                            realized_cost=3.+s*.1+e*.01-effect,fallback_count=1 if e==3 else 0,success=e==4))
    return rows


def test_seed_costs_average_twenty_episodes_within_five_training_clusters():
    from poc.control_utility import seed_costs
    result=seed_costs(private_episodes(),registered_design())
    assert result['seed_ids']==list(range(5)) and result['episodes_per_seed']==20
    assert len(result['rows'])==12
    row=next(r for r in result['rows'] if r['actuator']=='direct' and r['model_id']=='power_correct'
             and r['model_step_ns']==5_000_000)
    assert row['cost_by_seed']==pytest.approx([2.095+s*.097 for s in range(5)])
    assert row['fallback_by_seed']==[1]*5 and row['success_by_seed']==[1]*5


@pytest.mark.parametrize('mutation',['missing','duplicate','target','initial','clock','nonfinite',
                                     'discovery','namespace','partial'])
def test_incomplete_unpaired_or_mislabelled_episodes_cannot_enter_statistics(mutation):
    from poc.control_utility import seed_costs
    rows=private_episodes()
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows[-1]=copy.deepcopy(rows[0])
    elif mutation=='target':rows[0]['target'][0]+=.001
    elif mutation=='initial':rows[0]['initial_state'][0]+=.001
    elif mutation=='clock':rows[0]['model_partition_ns']=[5_000_000]*7
    elif mutation=='nonfinite':rows[0]['realized_cost']=float('nan')
    elif mutation=='discovery':rows[0]['seed']=9000
    elif mutation=='namespace':rows[0]['seed_namespaces']['cem'][0]=7101
    else:rows[0]['complete']=False
    with pytest.raises(ValueError):seed_costs(rows,registered_design())


def envelopes(design,amount=0.):
    return {r['criterion_id']:dict(lower=amount,upper=amount) for r in design['criteria']}


def test_signed_bounds_use_new_control_family_and_common_seed_resamples():
    from poc.control_utility import seed_costs,utility_bounds
    design=registered_design();costs=seed_costs(private_episodes(),design)
    result=utility_bounds(costs,design,1,envelopes(design))
    assert result['look_alpha']==.025 and result['tail_alpha']==.025/16
    assert result['bootstrap_seed_entropy']==[9102,1]
    assert len(result['criteria'])==8
    assert all(r['n_training_seeds']==5 and r['status']=='lower_cost' for r in result['criteria'])
    assert len({r['bounds']['bootstrap_indices_sha256'] for r in result['criteria']})==1
    assert all(r['point_difference']==pytest.approx(1.006) for r in result['criteria'])


def test_numerical_envelope_crossing_zero_prevents_positive_claim():
    from poc.control_utility import seed_costs,utility_bounds
    design=registered_design();costs=seed_costs(private_episodes(),design)
    result=utility_bounds(costs,design,1,envelopes(design,2.))
    assert all(r['status']=='unresolved' for r in result['criteria'])


def test_constant_unbounded_cluster_difference_never_manufactures_certainty():
    from poc.control_utility import seed_costs,utility_bounds
    design=registered_design();rows=private_episodes()
    for row in rows:row['realized_cost']=1. if row['model_id']=='power_correct' else 2.
    result=utility_bounds(seed_costs(rows,design),design,1,envelopes(design))
    assert all(r['status']=='unresolved_variance' for r in result['criteria'])
    assert all(r['bounds']['lower'] is None for r in result['criteria'])


def test_missing_numerical_qualification_has_no_zero_default():
    from poc.control_utility import seed_costs,utility_bounds
    design=registered_design()
    with pytest.raises(ValueError,match='numerical'):
        utility_bounds(seed_costs(private_episodes(),design),design,1,{})


def test_negative_control_difference_is_reported_with_the_same_rule():
    from poc.control_utility import seed_costs,utility_bounds
    design=registered_design();rows=private_episodes()
    for row in rows:
        if row['model_id']=='power_correct':row['realized_cost']+=2.+row['seed']*.006
    result=utility_bounds(seed_costs(rows,design),design,1,envelopes(design))
    assert all(r['status']=='higher_cost' for r in result['criteria'])


def raw_cost_fixture(folder,tick=250_000):
    import json
    summary=private_episodes()[0]
    summary['physics_step_ns']=tick
    times=np.arange(0,1_600_000_001,tick,dtype=np.int64)
    states=np.zeros((len(times),4));states[0]=summary['initial_state']
    states[:,2]=.1
    states[0]=summary['initial_state']
    stride=40_000_000//tick
    anchors=states[stride::stride]
    target=np.asarray(summary['target'])
    np.savez(folder/'true_trajectories.npz',time_ns=times,state=states,anchors=anchors,target=target,
             anchors_with_initial=np.vstack([states[0],anchors]))
    obs_times=np.arange(0,1_600_000_000,40_000_000,dtype=np.int64)
    np.savez(folder/'observations.npz',time_ns=obs_times,state=states[::stride][:-1],
             packet_id=np.array([f'obs-{t}' for t in obs_times]),noise_time_ns=np.array([],dtype=np.int64))
    segments=[dict(start_ns=int(t),duration_ns=40_000_000,command=[.2,-.1]) for t in obs_times]
    (folder/'action_segments.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in segments))
    decisions=[dict(time_ns=int(t),observation_packet_id=f'obs-{t}',status='ok',fallback=None)
               for t in obs_times]
    (folder/'decisions.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in decisions))
    # Independently known constant anchor state and command: one terminal term.
    expected=6.6*float(target@target)+1.6*.05*.1**2+1.6*.001*(.2**2+.1**2)
    summary.update(realized_cost=expected,fallback_count=0,success=bool(target@target<.1**2))
    return summary,expected


def test_raw_episode_cost_is_independently_rebuilt_from_physical_anchors_and_effort(tmp_path):
    from poc.control_utility import audit_raw_episode
    summary,expected=raw_cost_fixture(tmp_path)
    result=audit_raw_episode(tmp_path,summary)
    assert result['raw_cost']==pytest.approx(expected,rel=1e-14)
    assert result['n_real_anchors']==40 and result['n_observations']==40


@pytest.mark.parametrize('mutation',['claimed_cost','dense_clock','truth_anchor','fresh_observation','fallback'])
def test_raw_audit_detects_cost_clock_observation_and_failure_corruption(tmp_path,mutation):
    from poc.control_utility import audit_raw_episode
    summary,_=raw_cost_fixture(tmp_path)
    if mutation=='claimed_cost':summary['realized_cost']+=1.
    elif mutation=='fallback':summary['fallback_count']=1
    else:
        filename='observations.npz' if mutation=='fresh_observation' else 'true_trajectories.npz'
        with np.load(tmp_path/filename,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
        if mutation=='dense_clock':arrays['time_ns'][1]+=1
        elif mutation=='truth_anchor':arrays['anchors'][0,0]+=.001
        else:arrays['state'][0,0]+=.001
        np.savez(tmp_path/filename,**arrays)
    with pytest.raises(ValueError):audit_raw_episode(tmp_path,summary)


def test_control_spending_ledger_is_separate_idempotent_and_preserves_prediction_ledger(tmp_path):
    from poc.control_utility import reserve_control_look
    from poc.io import atomic_json,sha256_file
    original=tmp_path/'decisions/statistical_looks/look-00001.json'
    atomic_json(original,{'historical_family_size':280,'untouched':True})
    original_hash=sha256_file(original)
    design=registered_design();identity={'registry_sha256':'a'*64,'batch_sha256':'b'*64,'numerical_sha256':'c'*64}
    first=reserve_control_look(tmp_path,design,identity)
    assert reserve_control_look(tmp_path,design,identity)==first
    assert first['look_index']==1 and first['tail_alpha']==.025/16
    assert first['family_size']==8 and first['bootstrap_seed_entropy']==[9102,1]
    second=reserve_control_look(tmp_path,design,{**identity,'batch_sha256':'d'*64})
    assert second['look_index']==2 and second['cumulative_alpha']==.0375
    assert sha256_file(original)==original_hash


def test_modified_control_look_cannot_reset_or_reallocate_spending(tmp_path):
    from poc.control_utility import reserve_control_look
    from poc.io import atomic_json
    design=registered_design();identity={'registry_sha256':'a'*64,'batch_sha256':'b'*64,'numerical_sha256':'c'*64}
    record=reserve_control_look(tmp_path,design,identity)
    record['tail_alpha']=.025
    atomic_json(tmp_path/'decisions/control_utility_looks/look-00001.json',record)
    with pytest.raises(ValueError,match='ledger'):
        reserve_control_look(tmp_path,design,identity)


def test_incomplete_primary_batch_cannot_generate_a_scientific_audit(tmp_path):
    from poc.control_utility import audit_batch
    from poc.io import atomic_json
    atomic_json(tmp_path/'reports/control_confirmation_batch.json',
                dict(state='interrupted',expected_episodes=1200,completed_episodes=1,episodes=[]))
    with pytest.raises(ValueError,match='Complete paired primary'):
        audit_batch(tmp_path)
    assert not (tmp_path/'reports/control_confirmation_raw_audit.json').exists()


def test_refined_raw_cost_requires_explicit_qualified_numerical_tick(tmp_path):
    from poc.control_utility import audit_raw_episode
    summary,expected=raw_cost_fixture(tmp_path,tick=125_000)
    with pytest.raises(ValueError,match='physics tick'):
        audit_raw_episode(tmp_path,summary)
    result=audit_raw_episode(tmp_path,summary,expected_tick_ns=125_000)
    assert result['raw_cost']==pytest.approx(expected,rel=1e-14)
    summary['physics_step_ns']=250_000
    with pytest.raises(ValueError):
        audit_raw_episode(tmp_path,summary,expected_tick_ns=125_000)
