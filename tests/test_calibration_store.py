"""Immutable study evidence, binding isolation, and retained failures."""
import importlib.util
import json

import pytest


def store_at(path, **changes):
    assert importlib.util.find_spec('poc.calibration_store'), 'Study-local store is missing'
    from poc.calibration_store import CalibrationStore
    return CalibrationStore(path, {'root': 'root-a', 'source': 'source-a',
                                  'asset': 'asset-a', 'recipe': 'recipe-a', **changes})


def write_result(folder):
    (folder / 'values.json').write_text('{"value": 7}\n')
    return {'value': 7}


def test_repeated_produce_returns_identical_receipt_without_reexecution(tmp_path):
    store = store_at(tmp_path)
    receipt = store.produce('fit', {'session_id': 0}, write_result)
    assert set(receipt) == {'job_key', 'path', 'manifest_sha256'}
    assert json.loads(json.dumps(receipt)) == receipt
    def forbidden(_):
        pytest.fail('Completed producer executed again')
    assert store.produce('fit', {'session_id': 0}, forbidden) == receipt
    folder, manifest = store.read(receipt)
    assert manifest['protocol_id'] == 'PTCO-CAL-0.1'
    assert manifest['summary'] == {'value': 7}
    assert (folder / 'values.json').exists()
    assert store.produce('fit', {'session_id': 1}, write_result) != receipt


@pytest.mark.parametrize('binding', ['root', 'source', 'asset', 'recipe'])
def test_changed_binding_cannot_read_old_receipt(tmp_path, binding):
    receipt = store_at(tmp_path).produce('collect', {'session_id': 0}, write_result)
    changed = store_at(tmp_path, **{binding: 'changed'})
    with pytest.raises(ValueError, match='binding'):
        changed.read(receipt)
    assert changed.produce('collect', {'session_id': 0}, write_result) != receipt


@pytest.mark.parametrize('target', ['values.json', 'manifest.json', 'COMPLETED'])
def test_tampered_artifact_manifest_or_completion_is_rejected(tmp_path, target):
    store = store_at(tmp_path)
    receipt = store.produce('fit', {}, write_result)
    folder, _ = store.read(receipt)
    (folder / target).write_text('{}')
    with pytest.raises(ValueError):
        store.read(receipt)
    with pytest.raises(ValueError):
        store.produce('fit', {}, write_result)


def test_output_root_and_receipt_paths_are_bound(tmp_path):
    import shutil
    first, second = tmp_path / 'one', tmp_path / 'two'
    receipt = store_at(first).produce('fit', {}, write_result)
    shutil.copytree(first, second)
    with pytest.raises(ValueError, match='binding'):
        store_at(second).read(receipt)
    with pytest.raises(ValueError):
        store_at(first).read({**receipt, 'path': '../outside'})


def test_interrupted_attempt_retains_events_and_resumes_same_work_directory(tmp_path):
    store = store_at(tmp_path)
    def interrupted(folder):
        (folder / 'boundary.json').write_text('{"complete":true}')
        raise KeyboardInterrupt('unit fixture interruption')
    with pytest.raises(KeyboardInterrupt):
        store.produce('collect', {'session_id': 0}, interrupted)
    def resumed(folder):
        assert (folder / 'boundary.json').exists()
        return write_result(folder)
    receipt = store.produce('collect', {'session_id': 0}, resumed)
    store.read(receipt)
    events = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events] == ['started', 'failed', 'started', 'completed']
    assert events[1]['error_type'] == 'KeyboardInterrupt'
    assert events[1]['request'] == {'session_id': 0}


def test_duplicate_coordinates_and_unsafe_stage_fail_before_producer(tmp_path):
    store = store_at(tmp_path)
    with pytest.raises(ValueError, match='Duplicate'):
        store.produce('matrix', {'coordinates': [{'session': 0}, {'session': 0}]}, write_result)
    with pytest.raises(ValueError):
        store.produce('../fit', {}, write_result)


def test_missing_completion_marker_does_not_overwrite_final_manifest(tmp_path):
    store = store_at(tmp_path)
    receipt = store.produce('fit', {}, write_result)
    folder, _ = store.read(receipt)
    original = (folder / 'manifest.json').read_bytes()
    (folder / 'COMPLETED').unlink()
    with pytest.raises(ValueError, match='completion'):
        store.produce('fit', {}, write_result)
    assert (folder / 'manifest.json').read_bytes() == original


def test_unlisted_nested_manifest_named_artifact_is_detected(tmp_path):
    store = store_at(tmp_path)
    receipt = store.produce('fit', {}, write_result)
    folder, _ = store.read(receipt)
    extra = folder / 'extra'
    extra.mkdir()
    (extra / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='file set'):
        store.read(receipt)


def test_measured_store_finalization_times_real_hash_metadata_receipt_and_events(tmp_path,monkeypatch):
    from poc import calibration_store as module
    clock_value = [0.]
    def clock():
        return clock_value[0]
    hash_file,write_json,append = module.sha256_file,module.atomic_json,module.append_event
    def delayed_hash(path):
        clock_value[0] += 2.
        return hash_file(path)
    def delayed_metadata(path,value):
        if path.name in {'manifest.json','COMPLETED'}:
            clock_value[0] += .1
        return write_json(path,value)
    def delayed_event(path,event):
        if event['event']=='completed':
            clock_value[0] += .3
        elif event['event']=='store_finalization':
            clock_value[0] += .7  # The resource record's own write is explicitly excluded.
        return append(path,event)
    monkeypatch.setattr(module,'sha256_file',delayed_hash)
    monkeypatch.setattr(module,'atomic_json',delayed_metadata)
    monkeypatch.setattr(module,'append_event',delayed_event)
    store = store_at(tmp_path)
    def producer(folder):
        clock_value[0] += 1000.
        return write_result(folder)
    receipt = store.produce('timing',{},producer,measure_finalization=True,clock=clock,cpu_clock=clock)
    events = [json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    record = events[-1]
    assert record['event']=='store_finalization' and record['receipt']==receipt
    assert not record['reused']
    resources = record['resources']
    for category,want in [('artifact_hash',6.),('metadata_write_hash',2.2),
                          ('receipt_create_validate',10.),('completion_event_write',.3)]:
        assert resources[category+'_wall_s']==pytest.approx(want)
        assert resources[category+'_cpu_s']==pytest.approx(want)
    assert resources['store_finalization_wall_s']==pytest.approx(18.5)
    assert resources['store_finalization_cpu_s']==pytest.approx(18.5)
    assert resources['resource_record_write_measured'] is False
    assert set(receipt)=={'job_key','path','manifest_sha256'}
    folder,manifest = store.read(receipt)
    original = (folder/'manifest.json').read_bytes()
    def forbidden(_):
        pytest.fail('Finalized job must remain reusable')
    assert store.produce('timing',{},forbidden,measure_finalization=True,clock=clock,cpu_clock=clock)==receipt
    assert (folder/'manifest.json').read_bytes()==original
    reused = json.loads((tmp_path/'events.jsonl').read_text().splitlines()[-1])
    assert reused['reused'] and reused['resources']['receipt_create_validate_wall_s']==pytest.approx(10.)
    assert reused['resources']['artifact_hash_wall_s']==0.


def test_interrupted_store_sums_active_attempts_and_excludes_inactive_gap(tmp_path):
    store=store_at(tmp_path); now=[0.]
    def clock(): return now[0]
    def interrupted(folder):
        now[0]+=100
        raise KeyboardInterrupt('caught')
    with pytest.raises(KeyboardInterrupt):
        store.produce('fit',{},interrupted,clock=clock,cpu_clock=clock)
    now[0]+=1000
    def resumed(folder):
        now[0]+=7
        return write_result(folder)
    receipt=store.produce('fit',{},resumed,clock=clock,cpu_clock=clock)
    resources=store.read(receipt)[1]['resources']
    assert resources['wall_s']==resources['cpu_s']==107
    assert resources['accounting_status']=='measured'
    assert len({a['attempt_id'] for a in resources['attempts']})==2
    events=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    failed=next(e for e in events if e['event']=='failed')
    assert failed['resources']['wall_s']==100


def test_uncatchable_store_termination_retains_lower_bound_and_unknown_tail(tmp_path):
    import subprocess,sys
    script='''
import os,sys
from pathlib import Path
from poc.calibration_store import CalibrationStore
from poc.calibration_resources import checkpoint_resources
now=[0.]
def clock(): return now[0]
store=CalibrationStore(Path(sys.argv[1]),{'root':'root-a','source':'source-a','asset':'asset-a','recipe':'recipe-a'})
def producer(folder):
    now[0]=100.
    checkpoint_resources()
    now[0]=150.
    os._exit(23)
store.produce('fit',{},producer,clock=clock,cpu_clock=clock)
'''
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==23,result.stderr
    now=[0.]
    def resumed(folder):
        now[0]+=7
        return write_result(folder)
    store=store_at(tmp_path)
    receipt=store.produce('fit',{},resumed,clock=lambda:now[0],cpu_clock=lambda:now[0])
    resources=store.read(receipt)[1]['resources']
    assert resources['wall_s']==resources['cpu_s']==107
    assert resources['accounting_status']=='lower_bound_unknown_tail'
    assert resources['attempts'][0]['unknown_tail'] is True
    from poc.calibration_report import resource_accounting
    aggregate=resource_accounting([{'receipt':receipt,'producer_attempts':resources}])
    assert aggregate['wall_s']==107 and aggregate['accounting_status']=='lower_bound_unknown_tail'


def test_nested_resource_diagnostics_are_not_added_to_outer_spans(tmp_path):
    from poc.calibration_resources import ResourceTracker,checkpoint_resources
    now=[0.]
    def clock(): return now[0]
    store=store_at(tmp_path)
    def producer(folder):
        now[0]+=3
        with ResourceTracker(folder/'rate-resources.json',clock=clock,cpu_clock=clock) as child:
            now[0]+=10
            checkpoint_resources(lambda:now.__setitem__(0,now[0]+2))
            now[0]+=5
        return {'rate_resources':child.summary()}
    receipt=store.produce('fit',{},producer,clock=clock,cpu_clock=clock)
    _,manifest=store.read(receipt)
    assert manifest['resources']['active_wall_s']==18
    assert manifest['resources']['persistence_wall_s']==2
    assert manifest['resources']['wall_s']==20
    assert manifest['summary']['rate_resources']['wall_s']==17


def test_interrupted_resource_boundary_does_not_double_count_active_span(tmp_path,monkeypatch):
    from poc import calibration_resources as m
    now=[0.];write=m.atomic_json;tripped=[False]
    def interrupted(path,value):
        if now[0]==10 and not tripped[0]:
            tripped[0]=True
            raise KeyboardInterrupt('resource observation interrupted')
        return write(path,value)
    monkeypatch.setattr(m,'atomic_json',interrupted)
    tracker=m.ResourceTracker(tmp_path/'resources.json',clock=lambda:now[0],cpu_clock=lambda:now[0])
    with pytest.raises(KeyboardInterrupt):
        with tracker:
            now[0]=10
            m.checkpoint_resources()
    assert tracker.summary()['wall_s']==10
    assert not m._ACTIVE.get()


def test_nested_store_receipts_report_only_nonoverlapping_attempt_work(tmp_path):
    from poc.calibration_report import resource_accounting
    store=store_at(tmp_path); now=[0.];child_receipts=[]
    def clock(): return now[0]
    def child(folder):
        now[0]+=10
        return write_result(folder)
    def parent(folder):
        now[0]+=3
        child_receipts.append(store.produce('truth',{},child,clock=clock,cpu_clock=clock))
        now[0]+=5
        return write_result(folder)
    outer=store.produce('predictions',{},parent,clock=clock,cpu_clock=clock)
    rows=[{'receipt':r,'producer_attempts':store.read(r)[1]['resources']} for r in [outer,*child_receipts]]
    accounting=resource_accounting(rows)
    assert [r['producer_attempts']['wall_s'] for r in rows]==[18,10]
    assert accounting['wall_s']==accounting['cpu_s']==18
    assert accounting['accounting_status']=='measured'
    assert accounting['excluded_nested_attempts']==1
    assert resource_accounting(rows[1:])['wall_s']==10  # Parent absent: retain observed child work.


@pytest.mark.parametrize('observation', ['pre', 'post'])
@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('observation_seconds', [0, 7])
def test_resource_observation_interruption_excludes_observation_and_restarts(tmp_path, monkeypatch, observation, nested, observation_seconds):
    from contextlib import nullcontext
    from poc import calibration_resources as m
    now = [0.]; write = m.atomic_json; tripped = [False]
    def interrupted(path, value):
        if not tripped[0] and now[0] == (10 if observation == 'pre' else 12):
            tripped[0] = True
            now[0] += observation_seconds  # Ledger observation time belongs to neither measured category.
            raise KeyboardInterrupt('resource observation interrupted')
        return write(path, value)
    monkeypatch.setattr(m, 'atomic_json', interrupted)
    outer = m.ResourceTracker(tmp_path/'outer.json', clock=lambda:now[0], cpu_clock=lambda:now[0])
    child = m.ResourceTracker(tmp_path/'child.json', clock=lambda:now[0], cpu_clock=lambda:now[0]) if nested else None
    with pytest.raises(KeyboardInterrupt):
        with outer:
            with child if nested else nullcontext():
                now[0] += 10
                m.checkpoint_resources(lambda:now.__setitem__(0, now[0]+2))
    expected = 10 if observation == 'pre' else 12
    for tracker in [outer, *([child] if nested else [])]:
        assert tracker.summary()['wall_s'] == tracker.summary()['cpu_s'] == expected
        assert tracker.current['status'] == 'interrupted' and not tracker.current['unknown_tail']
    assert not m._ACTIVE.get()


def test_failed_commit_measures_work_and_excludes_observation_time(tmp_path, monkeypatch):
    from poc import calibration_resources as m
    now = [0.]; write = m.atomic_json
    def observed(path, value):
        now[0] += 3
        return write(path, value)
    monkeypatch.setattr(m, 'atomic_json', observed)
    tracker = m.ResourceTracker(tmp_path/'resources.json', clock=lambda:now[0], cpu_clock=lambda:now[0])
    def commit():
        now[0] += 2
        raise RuntimeError('failed persistence')
    with tracker:
        now[0] += 10
        with pytest.raises(RuntimeError, match='failed persistence'):
            m.checkpoint_resources(commit)
        now[0] += 5
    assert tracker.summary()['active_wall_s'] == 15
    assert tracker.summary()['persistence_wall_s'] == 2
    assert tracker.summary()['wall_s'] == tracker.summary()['cpu_s'] == 17
