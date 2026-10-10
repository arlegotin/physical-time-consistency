"""Fourth look retains the full twenty-task design and prior evidence."""
import copy
import pytest


def request():
    return dict(look=4, prior_seeds=list(range(2026100800, 2026100816)),
        new_seeds=list(range(2026100816, 2026100838)), episodes_per_seed=20,
        within_episode_fraction=.39176577617012026, variance=8.652425557330058e-6,
        mean_contrast=.0021911145320110937, planning_sd_margin=1.25,
        comparison_family_size=8, reason='Unstable strongest-NODE seed interpretation',
        prior_result_sha256='a'*64, allocation_sha256='b'*64,
        resource_projection=dict(cpu_hours=18., disk_gib=26., peak_ram_gib=.6))


def test_fourth_request_keeps_all_old_seeds_and_twenty_tasks():
    from poc.compute_study_seed_precision import validate_request, episode_ids
    r=request()
    assert validate_request(r)==list(range(2026100800,2026100838))
    assert episode_ids(r,'primary')==list(range(20))
    assert episode_ids(r,'replay')==[0,10] and episode_ids(r,'finer')==[0,10]
    for key,value in [('look',3),('episodes_per_seed',10),('comparison_family_size',7),
                      ('within_episode_fraction',.8),('new_seeds',list(range(2026100817,2026100839))),
                      ('new_seeds',[2026100801]*22),('variance',float('nan'))]:
        with pytest.raises(ValueError):validate_request({**r,key:value})
    with pytest.raises(ValueError):episode_ids(r,'other')


def test_external_receipts_keep_model_and_replica_paths_distinct(tmp_path):
    from poc.compute_study_seed_precision import receipt_path
    base=tmp_path/'authorized-store'
    assert receipt_path(base,'primary')==base/'receipts/primary.json'
    fit=receipt_path(base,'fit',model='affine_flow',seed=2026100816)
    replica=receipt_path(base,'fit',model='affine_flow',seed=2026100816,replica=True)
    assert fit!=replica and fit.parent==base/'receipts'
    with pytest.raises(ValueError):receipt_path(base,'fit',model='../escape',seed=1)
    with pytest.raises(ValueError):receipt_path(base,'../escape')


def test_fourth_design_does_not_change_previous_data_or_family():
    from poc.compute_study_seed_report import extend_design
    r=request();parent=dict(seeds=r['prior_seeds'],episodes_per_seed=20,comparison_family_size=8)
    before=copy.deepcopy(parent);expanded=extend_design(parent,r)
    assert parent==before and expanded['seeds']==list(range(2026100800,2026100838))
    assert expanded['episodes_per_seed']==20 and expanded['comparison_family_size']==8
    with pytest.raises(ValueError):extend_design({**parent,'seeds':[1,2]},r)
    with pytest.raises(ValueError):extend_design({**parent,'episodes_per_seed':10},r)


def test_fourth_join_retains_prior_rows_and_rejects_duplicate_observations():
    from poc.compute_study_seed_report import join_rows
    old=[dict(seed=2026100800,episode_id=19,model_id='affine_flow',step_ns=40000000)]
    new=[dict(seed=2026100816,episode_id=19,model_id='affine_flow',step_ns=40000000)]
    assert join_rows(old,new)==old+new
    with pytest.raises(ValueError):join_rows(old,old)


def test_fourth_reproduction_covers_both_tasks_of_the_last_new_seed():
    from poc.compute_study_seed_report import reproduction_coordinates
    coordinates=reproduction_coordinates(request()['new_seeds'])
    assert len(coordinates)==264
    assert ('node_quadrature',40000000,2026100837,0) in coordinates
    assert ('node_quadrature',40000000,2026100837,10) in coordinates
    assert ('node_quadrature',40000000,2026100837,19) not in coordinates
    assert ('node_quadrature',40000000,2026100800,0) not in coordinates


def test_reaggregation_uses_immutable_look_time_even_when_latest_is_older(tmp_path):
    from poc.compute_study_seed_report import creation_time
    from poc.io import atomic_json
    spending=dict(look=4,identity='a'*64,alpha=.003125)
    immutable=tmp_path/'reports/provenance/compute_study_look_004_result.json'
    atomic_json(immutable,dict(look=spending,created_utc='2026-10-10T01:00:00Z'))
    atomic_json(tmp_path/'reports/compute_study_result.json',dict(look=dict(look=3),created_utc='older'))
    assert creation_time(tmp_path,spending)=='2026-10-10T01:00:00Z'
    with pytest.raises(ValueError):creation_time(tmp_path,{**spending,'identity':'b'*64})


def test_reproduction_rejects_missing_and_duplicate_new_cells():
    from poc.compute_study_seed_report import validate_reproduction_cells, reproduction_coordinates
    seeds=request()['new_seeds']
    rows=[dict(model_id=n,step_ns=h,seed=s,episode_id=e) for n,h,s,e in reproduction_coordinates(seeds)]
    validate_reproduction_cells(rows,seeds)
    with pytest.raises(ValueError):validate_reproduction_cells(rows[:-1],seeds)
    with pytest.raises(ValueError):validate_reproduction_cells(rows[:-1]+[rows[0]],seeds)


def test_operational_progress_time_does_not_spend_another_scientific_look():
    from poc.compute_study_seed_report import cohort_identity
    a=dict(previous=dict(kind='primary',state='complete',completed=[dict(job_key='a',manifest_sha256='b')],updated_utc='first'))
    b=copy.deepcopy(a);b['previous']['updated_utc']='later'
    assert cohort_identity(a)==cohort_identity(b)
    b['previous']['completed'][0]['manifest_sha256']='changed'
    assert cohort_identity(a)!=cohort_identity(b)
