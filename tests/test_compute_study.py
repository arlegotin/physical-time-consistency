"""Prospective allocation, paired sampling and immutable evidence contracts."""
from pathlib import Path
import numpy as np
import pytest


def test_candidate_stream_prefixes_share_every_decision_and_iteration():
    from poc.compute_study_runs import task_conditions, normal_bank
    a=task_conditions(123,4); b=task_conditions(123,4)
    np.testing.assert_array_equal(a['initial_state'],b['initial_state'])
    np.testing.assert_array_equal(a['target'],b['target'])
    np.testing.assert_array_equal(normal_bank(123,4,16,64),normal_bank(123,4,64,64)[:,:,:16])
    assert not np.array_equal(a['target'],task_conditions(123,5)['target'])


def test_budget_selection_uses_timing_only_and_valid_eighth_elites():
    from poc.compute_study import closest_budget
    settings=closest_budget({16:.015,32:.024,48:.036,64:.050},.033)
    assert settings['candidate_count']==48 and settings['elite_count']==6
    assert settings['iterations']==4
    with pytest.raises(ValueError):closest_budget({17:.02},.03)


def test_store_reuses_complete_jobs_and_rejects_corruption_or_changed_sources(tmp_path):
    from poc.compute_study import StudyStore
    source=tmp_path/'source.py'; source.write_text('known inputs')
    store=StudyStore(tmp_path/'artifacts',{'source':str(source)},{'source':None})
    calls=[]
    def producer(folder):
        calls.append(1); (folder/'raw.txt').write_text('preserved'); return {'observed':3}
    receipt=store.produce('fixture',{'seed':2},producer)
    assert store.produce('fixture',{'seed':2},producer)==receipt and len(calls)==1
    assert store.read(receipt)[1]['observed']==3
    path,_=store.read(receipt);(path/'raw.txt').write_text('altered')
    with pytest.raises(ValueError):store.read(receipt)
    (path/'raw.txt').write_text('preserved');source.write_text('different')
    with pytest.raises(ValueError):store.read(receipt)


def test_interruption_is_not_completion_and_resume_keeps_committed_unit(tmp_path):
    from poc.compute_study import StudyStore
    source=tmp_path/'source.py';source.write_text('inputs')
    store=StudyStore(tmp_path/'artifacts',{'source':str(source)},{'source':None})
    def fail(folder):
        (folder/'unit').write_text('committed');raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):store.produce('fixture',{},fail)
    def resume(folder):
        assert (folder/'unit').read_text()=='committed';return {'resumed':True}
    receipt=store.produce('fixture',{},resume)
    folder,result=store.read(receipt)
    assert result['resumed'] and (folder/'attempts.jsonl').exists()
