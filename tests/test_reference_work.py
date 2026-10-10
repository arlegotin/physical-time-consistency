import pytest
from poc.reference_work import reconstruct_work


def test_complete_bank_reconciles_observed_simulator_steps_and_exact_field_work():
    result = reconstruct_work(512, 3_440_640)
    assert result['oracle_physics_steps_total'] == 3_440_640
    assert [r['batched_vector_field_calls'] for r in result['physics']] == [768, 1536, 3072]
    assert [r['scalar_transition_queries'] for r in result['physics']] == [3072] * 3
    assert result['count_provenance'] == 'Derived from complete successful bank coverage and fixed solver loops'


@pytest.mark.parametrize('count,observed', [(0, 0), (511, 3_440_640), (512, 3_440_639)])
def test_incomplete_or_inconsistent_counter_cannot_be_silently_filled(count, observed):
    with pytest.raises(ValueError): reconstruct_work(count, observed)
