"""Seed continuation preserves the original evidence and spending ledger."""
import importlib
import pytest


def api(name):
    try:
        module = importlib.import_module('poc.subset_precision')
    except ImportError as error:
        pytest.fail(f'Missing prospective precision continuation: {error}')
    return getattr(module, name)


def test_next_block_adds_independent_seeds_in_order_without_replacing_existing_ones():
    assert api('next_seed_block')([0, 1, 2, 3, 4]) == list(range(10))
    assert api('next_seed_block')(list(range(10))) == list(range(20))
    with pytest.raises(ValueError):
        api('next_seed_block')([0, 1, 2, 4, 5])


def test_precision_scope_does_not_rewrite_the_original_frozen_context(tmp_path):
    from poc.io import atomic_json, read_json
    prior = tmp_path / 'original'
    working = tmp_path / 'working'
    prior.mkdir(); working.mkdir()
    original = {'seed_ids': [0, 1, 2, 3, 4], 'statistical_family_size': 280, 'threshold': .5}
    for root in (prior, working):
        atomic_json(root / 'config/rejection_subset_scope.json', original)
    api('extend_working_scope')(working, list(range(10)))
    assert read_json(prior / 'config/rejection_subset_scope.json') == original
    changed = read_json(working / 'config/rejection_subset_scope.json')
    assert changed == {**original, 'seed_ids': list(range(10))}
    assert read_json(working / 'config/prior_rejection_subset_scope.json') == original


def test_next_context_retains_earlier_scope_receipts_and_rejects_in_place_reextension(tmp_path):
    from poc.io import atomic_json, read_json
    from poc.study_archive import create_snapshot
    original = {'seed_ids': list(range(5)), 'statistical_family_size': 280}
    atomic_json(tmp_path / 'config/rejection_subset_scope.json', original)
    create_snapshot(tmp_path, 'study/one')
    one = tmp_path / 'study/one'
    (one / 'STUDY_SNAPSHOT.json').rename(one / 'INITIAL_CONTEXT.json')
    api('extend_working_scope')(one, list(range(10)))
    create_snapshot(one, 'study/two')
    two = one / 'study/two'
    (two / 'STUDY_SNAPSHOT.json').rename(two / 'INITIAL_CONTEXT.json')
    api('extend_working_scope')(two, list(range(20)))
    assert read_json(one / 'config/rejection_subset_scope.json')['seed_ids'] == list(range(10))
    assert read_json(two / 'config/prior_rejection_subset_scope.json')['seed_ids'] == list(range(10))
    assert read_json(two / 'config/prior_rejection_subset_scope_before_5_seeds.json') == original
    with pytest.raises(ValueError, match='already extended'):
        api('extend_working_scope')(two, list(range(40)))
    assert read_json(two / 'config/rejection_subset_scope.json')['seed_ids'] == list(range(20))
