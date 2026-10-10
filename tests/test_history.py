"""Physical history boundaries, immutable values and independent branches."""
import importlib
import numpy as np
import pytest
import torch


def module():
    assert importlib.util.find_spec('poc.history'), 'Physical-time history is not implemented'
    return importlib.import_module('poc.history')


def token(start, duration=40_000_000, state=None, provenance='observed'):
    return module().HistoryToken(start,np.zeros(4) if state is None else state,
                                 np.array([.2,-.3]),duration,provenance,f'packet-{start}')


def history_fingerprint(history):
    return tuple((int(t),tuple((x.start_ns,x.duration_ns,x.provenance,x.packet_id,
                   tuple(x.state),tuple(x.mean_command)) for x in tokens))
                 for t,tokens in zip(history.query_time_ns,history.tokens))


def test_completed_start_at_window_boundary_included_without_cropping():
    history = module().HistoryBatch(((token(0),token(40_000_000),token(320_000_000),token(350_000_000)),),np.array([360_000_000],dtype=np.int64))
    eligible = history.eligible()
    assert [x.start_ns for x in eligible.tokens[0]] == [40_000_000,320_000_000]
    assert eligible.tokens[0][0].duration_ns == 40_000_000


def test_twenty_ms_augmentation_gives_sixteen_completed_tokens():
    tokens = tuple(token(i*20_000_000,20_000_000) for i in range(16))
    history = module().HistoryBatch((tokens,),np.array([320_000_000],dtype=np.int64)).eligible()
    assert len(history.tokens[0]) == 16
    assert len(module().HistoryBatch(((),),np.array([0],dtype=np.int64)).eligible().tokens[0]) == 0


def test_ingested_numpy_values_and_candidate_branches_cannot_alias():
    state = np.arange(4,dtype=float)
    history = module().HistoryBatch(((token(0,state=state),),),np.array([40_000_000],dtype=np.int64))
    state[:] = 99
    np.testing.assert_array_equal(history.tokens[0][0].state,np.arange(4))
    a,b = history.fork(),history.fork()
    # Bytes-backed immutable payloads may share storage; mutable inputs may not.
    assert not a.tokens[0][0].state.flags.writeable and not b.tokens[0][0].state.flags.writeable
    assert not a.tokens[0][0].mean_command.flags.writeable and not b.tokens[0][0].mean_command.flags.writeable
    with pytest.raises(ValueError):
        a.tokens[0][0].state[0] = 100
    with pytest.raises(ValueError):
        a.tokens[0][0].state.flags.writeable = True


def test_immutable_history_payloads_do_not_need_repeated_copy_validation(monkeypatch):
    h = module()
    history = h.HistoryBatch(((token(0),),),np.array([40_000_000],dtype=np.int64))
    calls = []
    old = h.HistoryToken.__post_init__
    def counted(self):
        calls.append(self.packet_id)
        old(self)
    monkeypatch.setattr(h.HistoryToken,'__post_init__',counted)
    for _ in range(10):
        query = history.fork().eligible()
        np.testing.assert_array_equal(query.tokens[0][0].state,history.tokens[0][0].state)
        with pytest.raises(ValueError):
            query.tokens[0][0].mean_command[0] = 2.
    assert calls == []


def test_mutable_imagined_tensor_payloads_still_have_independent_candidate_storage():
    state = torch.arange(4,dtype=torch.float32,requires_grad=True)
    history = module().HistoryBatch(((token(0,state=state,provenance='imagined'),),),np.array([40_000_000],dtype=np.int64))
    a,b = history.fork(),history.fork()
    assert a.tokens[0][0].state.data_ptr() != b.tokens[0][0].state.data_ptr()
    with torch.no_grad():
        a.tokens[0][0].state[0] = 100.
    assert b.tokens[0][0].state[0] == 0.
    b.tokens[0][0].state.sum().backward()
    torch.testing.assert_close(state.grad,torch.ones_like(state))


def test_append_completed_returns_new_history_and_rejects_future_token():
    old = module().HistoryBatch(((token(0),),),np.array([40_000_000],dtype=np.int64))
    completed = token(40_000_000,provenance='imagined')
    new = old.at_time(np.array([80_000_000],dtype=np.int64)).append_completed(completed)
    assert len(old.tokens[0]) == 1 and len(new.tokens[0]) == 2
    assert new.tokens[0][-1].provenance == 'imagined'
    with pytest.raises(ValueError,match='completed'):
        old.append_completed(completed)


def test_repeated_read_only_queries_and_candidate_order_do_not_change_packets():
    history = module().HistoryBatch(((token(0),token(40_000_000)),),np.array([80_000_000],dtype=np.int64))
    before = history_fingerprint(history)
    commands = np.array([[.1,.2],[-.4,.5],[.7,-.8]])
    def query(command):
        return sum(np.sum(x.state)+np.sum(x.mean_command) for x in history.eligible().tokens[0])+np.sum(command)
    original = np.array([query(x) for x in commands])
    permutation = np.array([2,0,1])
    shuffled = np.array([query(commands[i]) for i in permutation])[np.argsort(permutation)]
    np.testing.assert_array_equal(original,shuffled)
    for _ in range(100):
        history.eligible()
    assert history_fingerprint(history) == before


def test_time_weighted_high_rate_command_is_exact_and_not_new_observation():
    result = module().mean_command(((0,10_000_000,np.array([2.,0.])),
                                   (10_000_000,40_000_000,np.array([-2.,1.]))),0,40_000_000)
    np.testing.assert_array_equal(result,np.array([-1.,.75]))
    with pytest.raises(ValueError,match='contiguous|cover'):
        module().mean_command(((0,10_000_000,np.array([2.,0.])),),0,40_000_000)


def test_imagined_tensor_history_preserves_closed_loop_training_gradient():
    state = torch.arange(4,dtype=torch.float32,requires_grad=True)
    current = module().HistoryBatch(((),),np.array([40_000_000],dtype=np.int64))
    history = current.append_completed(token(0,state=state,provenance='imagined'))
    history.fork().tokens[0][0].state.sum().backward()
    torch.testing.assert_close(state.grad,torch.ones(4))


def test_invalid_time_dtype_duplicate_packets_and_unknown_provenance_fail():
    with pytest.raises(ValueError,match='nanosecond|integer'):
        module().HistoryBatch(((),),np.array([.04]))
    with pytest.raises(ValueError,match='duplicate|Duplicate'):
        module().HistoryBatch(((token(0),token(0)),),np.array([40_000_000],dtype=np.int64))
    with pytest.raises(ValueError,match='provenance'):
        token(0,provenance='fresh_model_substep_observation')
