"""Decision serialization preserves already computed uncertainty exactly."""
import importlib
import pytest


def api(name):
    try:
        module = importlib.import_module('poc.decision_wire')
    except ImportError as error:
        pytest.fail(f'Missing decision-bound interface adapter: {error}')
    return getattr(module, name)


def test_statistical_bounds_are_serialized_as_the_declared_object_without_changing_values():
    row = {'criterion_id': 'fixture', 'status': 'pass', 'estimate': .4,
        'bounds': [.1, .7], 'threshold': .5, 'operator': '>=', 'reason': 'fixture',
        'pass_bounds': {'lower': .1}, 'look_index': 1}
    result = api('decision_row')(row)
    assert result == {'criterion_id': 'fixture', 'status': 'pass', 'estimate': .4,
        'bounds': {'lower': .1, 'upper': .7}, 'threshold': .5,
        'operator': '>=', 'reason': 'fixture'}
    assert row['bounds'] == [.1, .7]


def test_unresolved_bounds_remain_unknown_and_invalid_bounds_are_rejected():
    row = {'criterion_id': 'fixture', 'status': 'unresolved', 'estimate': None,
        'bounds': None, 'threshold': .5, 'operator': '>=', 'reason': 'fixture'}
    assert api('decision_row')(row)['bounds'] is None
    with pytest.raises(ValueError):
        api('decision_row')({**row, 'bounds': [.5, .1]})
