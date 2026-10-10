"""Reference tasks match confirmation; one physics fit is not five trained replications."""
from pathlib import Path
import copy
import pytest
from poc.io import read_json,atomic_json


def inputs():
    design=read_json(Path('config/control_confirmation_design.json'))
    settings=read_json(Path('config/planner.json'))['settings']
    selected=[dict(actuator=a,model_id='physics_fit',manifest=f'fits/{a}/manifest.json',
        manifest_sha256='a'*64,checkpoint_sha256='b'*64) for a in ('direct','lagged')]
    return design,settings,selected


def test_reference_requests_pair_all_tasks_and_limit_simulator_to_declared_subset():
    from poc.control_references import reference_requests
    design,settings,selected=inputs()
    rows=reference_requests(design,settings,selected)
    assert len(rows)==450
    assert len({r['cell_id'] for r in rows})==450
    requests=[r['request'] for r in rows]
    assert sum(r['model_id']=='physics_fit' for r in requests)==200
    assert sum(r['model_id']=='zero_command' for r in requests)==200
    assert sum(r['model_id']=='sim_oracle' for r in requests)==50
    assert {r['seed'] for r in requests}==set(range(5))
    assert all(r['phase']=='confirmatory' and r['condition']=='clean' and r['physics_step_ns']==250_000
               and r['model_partition_ns']==[5_000_000]*8 and r['actuator_interval_ns']==40_000_000 for r in requests)
    assert all(r['episode_id']<5 for r in requests if r['model_id']=='sim_oracle')
    assert len({r['checkpoint_sha256'] for r in requests if r['actuator']=='direct' and r['model_id']=='physics_fit'})==1
    assert all(r['model_manifest'] is None for r in requests if r['model_id']!='physics_fit')


@pytest.mark.parametrize('mutation',['missing','duplicate','kind'])
def test_reference_requests_require_one_exact_fitted_physics_reference_per_plant(mutation):
    from poc.control_references import reference_requests
    design,settings,selected=inputs()
    if mutation=='missing':selected.pop()
    elif mutation=='duplicate':selected[-1]=copy.deepcopy(selected[0])
    else:selected[0]['model_id']='node_structured'
    with pytest.raises(ValueError):reference_requests(design,settings,selected)


def test_reference_prepare_cannot_consume_unfinished_primary_control(tmp_path):
    from poc.control_references import prepare
    atomic_json(tmp_path/'reports/control_confirmation_batch.json',
                dict(state='running',expected_episodes=1200,completed_episodes=1,episodes=[]))
    with pytest.raises(ValueError,match='Complete primary'):
        prepare(tmp_path)
    assert not (tmp_path/'config/control_reference_jobs.json').exists()
