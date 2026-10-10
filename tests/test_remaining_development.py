"""Remaining control development uses the originally declared seven fits and ten tasks."""
import copy
import pytest


def requests():
    from poc.remaining_development import remaining_fit_keys
    result=[]
    for a,m in sorted(remaining_fit_keys()):
        for h in (40_000_000,5_000_000):
            for e in range(10):
                result.append(dict(schema_version=1,phase='exploratory',model_id=m,actuator=a,seed=9000,
                    episode_id=e,condition='clean',model_partition_ns=[h]*(40_000_000//h),
                    actuator_interval_ns=40_000_000,physics_step_ns=250_000,power_mode=None,
                    model_manifest=f'evidence/M30/{a}-{m}/manifest.json',manifest_sha256='a'*64,
                    checkpoint_sha256='b'*64,planner_settings=dict(candidate_count=128,iterations=4,
                    elite_count=16,initial_std=1.,std_floor=.1),resource_projection={'reason':'Private coverage test'}))
    return result


def test_remaining_development_has_exactly_original_one_hundred_forty_cells():
    from poc.remaining_development import validate_requests
    values=requests()
    assert len(values)==140 and validate_requests(values)==values


@pytest.mark.parametrize('mutation',['missing','duplicate','clock','seed','model','power','rate'])
def test_changed_remaining_workload_fails_before_control(mutation):
    from poc.remaining_development import validate_requests
    values=requests()
    if mutation=='missing':values.pop()
    elif mutation=='duplicate':values[-1]=copy.deepcopy(values[0])
    elif mutation=='clock':values[0]['model_partition_ns']=[10_000_000]*4
    elif mutation=='seed':values[0]['seed']=0
    elif mutation=='model':values[0]['model_id']='mlp_dt'
    elif mutation=='power':values[0]['power_mode']='low'
    else:values[0]['actuator_interval_ns']=20_000_000
    with pytest.raises(ValueError):validate_requests(values)


def test_remaining_registry_cannot_be_prepared_from_unfinished_nominal_fits(tmp_path):
    from poc.remaining_development import prepare_registry
    with pytest.raises(ValueError,match='Complete original nominal'):
        prepare_registry(tmp_path)
    assert not (tmp_path/'config/control_development_remaining_jobs.json').exists()
