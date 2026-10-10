"""Prospective continuation preserves earlier evidence and pairing."""
import copy
import numpy as np
import pytest


def request():
    return dict(look=2,prior_seeds=list(range(2026100800,2026100811)),
        new_seeds=list(range(2026100811,2026100816)),episodes_per_seed=10,
        within_episode_fraction=.6756924381326325,variance=1.161590332956962e-5,
        mean_contrast=.00270717879804495,planning_sd_margin=1.25,
        reason='Influential interpretation in coarse AffineFlow versus coarse RK4',
        prior_result_sha256='a'*64,allocation_sha256='b'*64,
        resource_projection=dict(cpu_hours=3.1,disk_gib=3.,peak_ram_gib=1.))


def test_precision_seeds_are_next_independent_block_not_replacements():
    from poc.compute_study_precision import validate_request
    r=request();assert validate_request(r)==list(range(2026100800,2026100816))
    for key,value in [('new_seeds',[2026100801]*5),('new_seeds',[2026100812,2026100813,2026100814,2026100815,2026100816]),
                      ('within_episode_fraction',.8),('look',1),('variance',float('nan'))]:
        bad={**r,key:value}
        with pytest.raises(ValueError):validate_request(bad)


def test_precision_factory_preserves_old_comparator_initialization():
    from poc.compute_study_precision import build
    from poc.affine_fresh import build_fresh_model
    from poc.compute_study import build as quad
    from poc.models.common import Scales
    from poc.train import canonical_tensor_hash
    import torch
    scales=Scales(torch.ones(2),torch.ones(2),torch.ones(2))
    for name in ('affine_flow','node_structured','node_quadrature'):
        model=build(name,scales,2026100811)
        expected=quad(scales,2026100811) if name=='node_quadrature' else build_fresh_model(name,scales,2026100811)
        assert canonical_tensor_hash(model.state_dict())==canonical_tensor_hash(expected.state_dict())


def test_expanded_replay_covers_every_seed_without_duplicate_tasks():
    from poc.compute_study_precision import joined_cells
    old=[dict(seed=s,episode_id=0,model_id='affine_flow',step_ns=40000000) for s in (1,2)]
    new=[dict(seed=3,episode_id=0,model_id='affine_flow',step_ns=40000000)]
    assert len(joined_cells(old,new))==3
    with pytest.raises(ValueError):joined_cells(old,old)


def test_report_expands_design_without_mutating_first_freeze():
    from poc.compute_study_report import extend_design
    r=request();initial=dict(seeds=r['prior_seeds'],episodes_per_seed=10,comparison_family_size=8)
    before=copy.deepcopy(initial);expanded=extend_design(initial,r)
    assert initial==before and len(expanded['seeds'])==16
    assert expanded['comparison_family_size']==8
    with pytest.raises(ValueError):extend_design({**initial,'seeds':[1,2]},r)


def test_timing_profiles_merge_as_same_seed_clusters_not_extra_replicates():
    from poc.compute_study_report import merge_profiles
    a=[dict(model_id='affine_flow',seed=1,batch_size=128,h_s=.005)]
    b=[dict(model_id='affine_flow',seed=2,batch_size=128,h_s=.005)]
    assert len(merge_profiles(a,b))==2
    with pytest.raises(ValueError):merge_profiles(a,a)
