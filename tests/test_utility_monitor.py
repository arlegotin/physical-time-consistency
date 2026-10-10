"""Monitoring reads committed progress without treating it as scientific evidence."""
from pathlib import Path

import pytest

from poc.io import atomic_json,append_event,job_key,sha256_file


def active_training(root):
    context=root/'study/control-development'
    parameters=dict(cell_id='direct__A__shared__gru_dt__seed9000',actuator='direct',
        condition='A',loss_mode='shared',model_id='gru_dt',seed=9000,epochs=500,learning_rate=.001)
    request=dict(protocol_version='PTCO-1.0',phase='exploratory',purpose='training',semantic_inputs=parameters)
    path=context/'config/job_requests/train-request.json';atomic_json(path,request)
    digest=job_key({'command':'train','request':request})
    canonical=context/f'config/job_requests/train-{digest}.json';path.rename(canonical)
    append_event(context/'control_nominal_completion/events.jsonl',dict(event='child_job_start',
        command='train',request_hash=digest,utc='2026-10-02T10:51:00Z'))
    folder=context/'evidence/M30/current'
    atomic_json(folder/'request.json',dict(cell_id=parameters['cell_id'],
        inputs=dict(phase='exploratory',parameters={**parameters,'normalization_sha256':'a'*64}),job_key='b'*64))
    # This marker is inspected for its committed-update counter; tensor data are never loaded.
    atomic_json(folder/'resume.complete.json',dict(checkpoint_sha256='c'*64,rng_sha256='d'*64,update=57088))
    return context,folder,canonical,digest


def test_monitor_reads_exact_current_request_and_atomic_update_marker_without_tensor_loading(tmp_path):
    from poc.utility_monitor import observe
    _,folder,_,_=active_training(tmp_path)
    result=observe(tmp_path)
    assert result['original_nominal']['receipt_state']=='not_started'
    assert result['original_nominal']['training']['cell_id']=='direct__A__shared__gru_dt__seed9000'
    assert result['original_nominal']['training']['target_epochs']==500
    assert result['original_nominal']['training']['committed_optimizer_updates']==57088
    assert result['original_nominal']['training']['source_request']==str((folder/'request.json').relative_to(tmp_path))
    assert result['primary_control']['state']=='not_started'
    assert result['scientific_decision']=='not_evaluated'
    assert result['disk_free_bytes']>0


def test_training_completion_receipt_does_not_imply_control_completion(tmp_path):
    from poc.utility_monitor import observe
    atomic_json(tmp_path/'reports/control_nominal_completion.json',dict(state='complete',cells=[{}]*7))
    result=observe(tmp_path)
    assert result['original_nominal']['receipt_state']=='complete'
    assert result['original_nominal']['completed_fits']==7
    assert result['primary_control']['state']=='not_started'
    assert result['solver_audit']['state']=='not_started'


def test_interrupted_primary_batch_is_visible_with_completed_count_and_reason(tmp_path):
    from poc.utility_monitor import observe
    atomic_json(tmp_path/'reports/control_confirmation_batch.json',dict(state='interrupted',
        completed_episodes=11,expected_episodes=1200,reason='actual disk exhausted',episodes=[{}]*11))
    result=observe(tmp_path)
    assert result['primary_control']['state']=='interrupted'
    assert result['primary_control']['completed_episodes']==11
    assert result['primary_control']['reason']=='actual disk exhausted'
    assert result['scientific_decision']=='not_evaluated'


@pytest.mark.parametrize('mutation',['request_hash','duplicate_run','truncated_event'])
def test_ambiguous_or_incomplete_training_journal_stays_unknown_in_monitor(tmp_path,mutation):
    import json
    from poc.utility_monitor import observe
    from poc.io import read_json
    context,folder,request,digest=active_training(tmp_path)
    if mutation=='request_hash':
        value=read_json(request);value['semantic_inputs']['seed']=1;atomic_json(request,value)
    elif mutation=='duplicate_run':
        atomic_json(context/'evidence/M30/other/request.json',read_json(folder/'request.json'))
    else:
        with (context/'control_nominal_completion/events.jsonl').open('a') as stream:stream.write('{"event":')
    result=observe(tmp_path)
    assert result['original_nominal']['training']['status']=='unresolved_journal'
    assert result['scientific_decision']=='not_evaluated'
