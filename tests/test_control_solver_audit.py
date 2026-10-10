"""Exploratory solver probes reuse a fixed, complete development subset."""
import pytest


def complete_batch():
    episodes=[]
    for model in ('mlp_dt','node_generic','power_correct'):
        for h in (40_000_000,5_000_000):
            for episode in range(10):
                episodes.append(dict(manifest=f'evidence/M40/{model}_{h}_{episode}/manifest.json',
                    manifest_sha256='a'*64,summary=dict(model_id=model,actuator='direct',seed=9000,
                    phase='exploratory',complete=True,episode_id=episode,condition='clean',
                    physics_step_ns=250_000,actuator_interval_ns=40_000_000,
                    model_partition_ns=[h]*(40_000_000//h))))
    return dict(state='complete',expected_episodes=60,completed_episodes=60,episodes=episodes)


def test_solver_subset_is_first_five_tasks_at_both_registered_model_clocks():
    from poc.control_solver_audit import development_subset
    result=development_subset(complete_batch())
    assert len(result)==10
    assert {e['summary']['episode_id'] for e in result}==set(range(5))
    assert all(e['summary']['model_id']=='node_generic' for e in result)
    assert {e['summary']['model_partition_ns'][0] for e in result}=={40_000_000,5_000_000}


@pytest.mark.parametrize('mutation',['partial','missing','duplicate','phase','seed','clock'])
def test_solver_diagnostic_cannot_relabel_or_select_incomplete_development(mutation):
    from poc.control_solver_audit import development_subset
    batch=complete_batch()
    if mutation=='partial':batch['state']='running'
    elif mutation=='missing':batch['episodes'].pop()
    elif mutation=='duplicate':batch['episodes'][-1]=batch['episodes'][0]
    elif mutation=='phase':batch['episodes'][0]['summary']['phase']='confirmatory'
    elif mutation=='seed':batch['episodes'][0]['summary']['seed']=0
    else:batch['episodes'][0]['summary']['model_partition_ns']=[20_000_000]*2
    with pytest.raises(ValueError):development_subset(batch)


def test_changed_probe_namespace_fails_even_before_a_dry_run(tmp_path):
    from poc.control_solver_audit import main
    from poc.io import atomic_json,read_json
    from pathlib import Path
    design=read_json(Path('config/control_development_solver_audit.json'));design['command_namespace']=9102
    atomic_json(tmp_path/'config/control_development_solver_audit.json',design)
    atomic_json(tmp_path/'reports/control_development_direct.json',complete_batch())
    with pytest.raises(ValueError,match='solver audit design'):
        main(['--root',str(tmp_path),'--dry-run'])


def test_zero_solver_gain_stays_explicit_in_summary_instead_of_disappearing():
    from poc.control_solver_audit import summarize_probes
    difference=dict(maximum_state_difference=.001,jacobian_differences={
        'q':dict(relative_difference=None),'v':dict(relative_difference=.002)})
    raw=[dict(horizon_ns=h,differences=[difference,difference]) for h in (5_000_000,40_000_000)]
    rows=summarize_probes(raw)
    assert len(rows)==4
    assert all(r['n_probes']==1 and r['n_unresolved_q_norms']==1 and r['n_unresolved_v_norms']==0 for r in rows)
    assert all(r['maximum_relative_q_jacobian_difference'] is None for r in rows)
    assert all(r['maximum_relative_v_jacobian_difference']==.002 for r in rows)
