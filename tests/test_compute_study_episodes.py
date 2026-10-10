"""Episode precision adds paired tasks without changing trained models."""
import copy
import pytest


def request():
    return dict(look=3,seeds=list(range(2026100800,2026100816)),
        prior_episodes_per_seed=10,episodes_per_seed=20,new_episode_ids=list(range(10,20)),
        within_episode_fraction=.8441310986271785,variance=9.397944104935655e-6,
        mean_contrast=.00259866586880828,planning_sd_margin=1.25,
        reason='Unstable strongest-NODE contrast; uncertainty mostly within tasks',
        prior_result_sha256='a'*64,allocation_sha256='b'*64,
        resource_projection=dict(cpu_hours=3.13,disk_gib=8.5,peak_ram_gib=.52))


def test_episode_expansion_requires_fresh_complete_paired_block():
    from poc.compute_study_episodes import validate_request,episode_ids
    r=request();assert validate_request(r)==list(range(10,20))
    assert episode_ids(r,'primary')==list(range(10,20))
    assert episode_ids(r,'replay')==[10] and episode_ids(r,'finer')==[10]
    for key,value in [('new_episode_ids',[0]*10),('episodes_per_seed',19),
                      ('within_episode_fraction',.69),('look',2),('seeds',[1,1])]:
        with pytest.raises(ValueError):validate_request({**r,key:value})
    with pytest.raises(ValueError):episode_ids(r,'unknown')


def test_report_episode_design_preserves_all_seed_clusters_and_family():
    from poc.compute_study_report import extend_episode_design
    r=request();initial=dict(seeds=r['seeds'],episodes_per_seed=10,comparison_family_size=8)
    before=copy.deepcopy(initial);expanded=extend_episode_design(initial,r)
    assert initial==before and expanded['seeds']==before['seeds']
    assert expanded['episodes_per_seed']==20 and expanded['comparison_family_size']==8
    with pytest.raises(ValueError):extend_episode_design({**initial,'seeds':[1,2]},r)


def test_reproduction_ids_include_old_and_added_tasks_without_counting_new_seeds():
    from poc.compute_study_report import reproduction_ids
    assert reproduction_ids(1)==(0,) and reproduction_ids(2)==(0,)
    assert reproduction_ids(3)==(0,10)
    with pytest.raises(ValueError):reproduction_ids(4)
