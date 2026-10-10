import copy
import pytest
from poc.nominal_development import validate_cells
from poc.mechanism import cell


def cells():
    return [cell('direct', 'A', 'shared', m, 9000) for m in ('gru_dt', 'node_structured')] + [
        cell('lagged', 'A', 'shared', m, 9000)
        for m in ('mlp_dt', 'gru_dt', 'node_generic', 'power_correct', 'node_structured')]


def test_only_original_missing_nominal_development_cells_are_accepted():
    assert validate_cells(cells()) == cells()


@pytest.mark.parametrize('mutation', ['seed', 'objective', 'intervals', 'architecture', 'missing', 'duplicate'])
def test_confirmation_replacement_fine_time_tuning_or_scope_change_is_rejected(mutation):
    values = copy.deepcopy(cells())
    if mutation == 'seed': values[0]['seed'] = 0
    if mutation == 'objective': values[0]['loss_mode'] = 'one_step'
    if mutation == 'intervals': values[0]['condition'] = 'fine'
    if mutation == 'architecture': values[0]['model_id'] = 'new_architecture'
    if mutation == 'missing': values.pop()
    if mutation == 'duplicate': values.append(copy.deepcopy(values[0]))
    with pytest.raises(ValueError): validate_cells(values)
