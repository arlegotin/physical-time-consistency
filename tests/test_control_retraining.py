"""Fresh training is compared as tensors, never checkpoint ZIP metadata."""
import copy

import pytest
import torch


def selected_freeze():
    return dict(state='complete',expected_episodes=1200,selected=[
        dict(actuator=a,model_id=m,seed=s,manifest=f'{a}/{m}/{s}/manifest.json')
        for a in ('direct','lagged') for m in ('mlp_dt','node_generic','power_correct')
        for s in range(5)])


def test_replication_uses_both_fixed_seeds_every_model_and_plant():
    from poc.control_retraining import training_replication_cells
    rows=training_replication_cells(selected_freeze())
    assert len(rows)==12
    assert {(r['actuator'],r['model_id'],r['seed']) for r in rows}=={
        (a,m,s) for a in ('direct','lagged')
        for m in ('mlp_dt','node_generic','power_correct') for s in (0,4)}
    assert rows==training_replication_cells(selected_freeze())


@pytest.mark.parametrize('mutation',['missing','duplicate','phase','seed'])
def test_replication_cannot_choose_favorable_or_incomplete_selected_fits(mutation):
    from poc.control_retraining import training_replication_cells
    frozen=selected_freeze()
    if mutation=='missing':frozen['selected'].pop()
    elif mutation=='duplicate':frozen['selected'][-1]=frozen['selected'][0]
    elif mutation=='phase':frozen['state']='running'
    else:frozen['selected'][0]['seed']=9000
    with pytest.raises(ValueError):training_replication_cells(frozen)


def weights():
    return {'layer.weight':torch.tensor([[0.,2.],[-1.,.5]],dtype=torch.float32),
            'layer.bias':torch.tensor([0.,.1],dtype=torch.float32)}


def validation():
    return dict(epoch=100,objective=.02,one_step=.01,four_step=.04,nominal_40_one_step=.009)


def test_exact_training_reproduction_passes_with_canonical_tensor_identity():
    from poc.control_retraining import compare_training
    result=compare_training(weights(),weights(),validation(),validation())
    assert result['status']=='pass'
    assert result['canonical_tensors_identical'] is True
    assert result['max_parameter_difference']==0.
    assert result['validation_objective_relative_difference']==0.
    assert result['fresh_training'] is True
    assert result['control_checked'] is False


def test_fixed_parameter_tolerance_uses_each_parameters_original_scale():
    from poc.control_retraining import compare_training
    original=weights();fresh=copy.deepcopy(original)
    fresh['layer.weight'][0,1]+=2e-6
    result=compare_training(original,fresh,validation(),validation())
    assert result['status']=='pass' and result['canonical_tensors_identical'] is False
    fresh['layer.bias'][0]=1.2e-6
    result=compare_training(original,fresh,validation(),validation())
    assert result['status']=='outside_tolerance' and result['diagnosis_required'] is True
    assert result['parameters']['layer.bias']['within_tolerance'] is False


def test_validation_mismatch_does_not_disappear_when_weights_match():
    from poc.control_retraining import compare_training
    changed=validation();changed['objective']*=1.0002
    result=compare_training(weights(),weights(),validation(),changed)
    assert result['status']=='outside_tolerance' and result['diagnosis_required'] is True


@pytest.mark.parametrize('mutation',['name','shape','dtype','nonfinite','no_tensors','negative_validation','nonfinite_validation'])
def test_corrupt_or_different_training_payload_is_invalid(mutation):
    from poc.control_retraining import compare_training
    original=weights();fresh=copy.deepcopy(original);score=validation()
    if mutation=='name':fresh.pop('layer.bias')
    elif mutation=='shape':fresh['layer.bias']=fresh['layer.bias'][:1]
    elif mutation=='dtype':fresh['layer.bias']=fresh['layer.bias'].double()
    elif mutation=='nonfinite':fresh['layer.bias'][0]=float('nan')
    elif mutation=='no_tensors':fresh['layer.bias']=[0.,.1]
    elif mutation=='negative_validation':score['objective']=-1
    else:score['objective']=float('inf')
    with pytest.raises(ValueError):compare_training(original,fresh,validation(),score)


def test_zero_validation_denominator_is_exact_identity_or_unresolved_not_zero_filled():
    from poc.control_retraining import compare_training
    first=validation();second=validation();first['objective']=second['objective']=0.
    assert compare_training(weights(),weights(),first,second)['status']=='pass'
    second['objective']=1e-12
    result=compare_training(weights(),weights(),first,second)
    assert result['status']=='outside_tolerance'
    assert result['validation_objective_relative_difference'] is None


def context_fixture(root):
    from poc.io import atomic_json,sha256_file
    (root/'poc').mkdir();(root/'poc/engine.py').write_text('original engine\n')
    (root/'schemas').mkdir();(root/'schemas/manifest.schema.json').write_text('{}')
    for name in ('environment.json','wheels.json','requirements.lock'):
        path=root/'locks'/name;path.parent.mkdir(exist_ok=True);path.write_text('{}')
    root.joinpath('config').mkdir();atomic_json(root/'config/base.json',{'fixture':True})
    paths={}
    for role in ('train','validation'):
        path=root/f'data/direct/A/{role}/manifest.json';path.parent.mkdir(parents=True)
        path.with_name('trajectories.npz').write_bytes(b'original labels '+role.encode())
        atomic_json(path,dict(role=role,actuator='direct',condition='A',
            files={'trajectories.npz':sha256_file(path.with_name('trajectories.npz'))}))
        paths[role+'_manifest']=str(path.relative_to(root))
    output=root/'evidence/M20/reference/reference.json';output.parent.mkdir(parents=True);output.write_text('{}')
    manifest=output.parent/'manifest.json'
    atomic_json(manifest,dict(produced_files=[dict(path=str(output.relative_to(root)),sha256=sha256_file(output))],
        job_inputs=dict(dependencies={'poc/engine.py':sha256_file(root/'poc/engine.py')},
                        configuration_keys={'config/base.json':{'fixture':True}},wheel_hashes={}),parent_decisions=[]))
    marker=output.parent/'COMPLETED';atomic_json(marker,dict(manifest_sha256=sha256_file(manifest)))
    decision=dict(gate_id='G20',decision_id='fixture',evidence=[dict(path=str(manifest.relative_to(root)),
        sha256=sha256_file(manifest),role='manifest')])
    atomic_json(root/'decisions/G20/fixture.json',decision)
    cached=root/'evidence/M30/old/weights.pt';cached.parent.mkdir(parents=True);cached.write_bytes(b'cached neural weights')
    return dict(**paths,actuator='direct',condition='A',loss_mode='shared',model_id='power_correct',seed=0),decision


def test_fresh_context_inherits_only_engine_labels_and_early_validity_not_cached_training(tmp_path):
    from poc.control_retraining import retraining_context_inventory
    from poc.io import sha256_file
    parameters,parent=context_fixture(tmp_path)
    result=retraining_context_inventory(tmp_path,parameters,[parent])
    assert result['files']['poc/engine.py']==sha256_file(tmp_path/'poc/engine.py')
    assert result['files']['data/direct/A/train/trajectories.npz']==sha256_file(
        tmp_path/'data/direct/A/train/trajectories.npz')
    assert not any(n.startswith('evidence/M30/') or '/confirmation/' in n for n in result['files'])
    assert set(result['gate_inputs']['gates'])=={'G20'}
    assert result['cached_training_copied'] is False


@pytest.mark.parametrize('mutation',['labels','role','parent','cached_dependency'])
def test_fresh_context_refuses_changed_labels_or_hidden_cache_dependency(tmp_path,mutation):
    from poc.control_retraining import retraining_context_inventory
    from poc.io import atomic_json,read_json,sha256_file
    parameters,parent=context_fixture(tmp_path)
    path=tmp_path/parameters['train_manifest']
    if mutation=='labels':path.with_name('trajectories.npz').write_bytes(b'changed')
    elif mutation=='role':
        data=read_json(path);data['role']='confirmation';atomic_json(path,data)
    elif mutation=='parent':(tmp_path/'evidence/M20/reference/reference.json').write_text('changed')
    else:
        manifest=tmp_path/'evidence/M20/reference/manifest.json';data=read_json(manifest)
        data['job_inputs']['dependencies']['evidence/M30/old/weights.pt']=sha256_file(tmp_path/'evidence/M30/old/weights.pt')
        atomic_json(manifest,data);parent['evidence'][0]['sha256']=sha256_file(manifest)
        atomic_json(tmp_path/'decisions/G20/fixture.json',parent)
    with pytest.raises(ValueError):retraining_context_inventory(tmp_path,parameters,[parent])


def test_context_copy_preserves_source_and_has_no_existing_training_receipts(tmp_path):
    from poc.control_retraining import copy_retraining_context,retraining_context_inventory
    from poc.io import read_json,sha256_file
    source=tmp_path/'source';source.mkdir();parameters,parent=context_fixture(source)
    inventory=retraining_context_inventory(source,parameters,[parent]);target=tmp_path/'fresh'
    copy_retraining_context(source,target,inventory)
    assert all(sha256_file(target/n)==digest for n,digest in inventory['files'].items())
    assert not (target/'evidence/M30').exists()
    assert read_json(target/'decisions/gate_inputs.json')==inventory['gate_inputs']
    assert read_json(target/'FRESH_TRAINING_CONTEXT.json')['cached_training_copied'] is False
    (target/'data/direct/A/train/trajectories.npz').write_bytes(b'local replacement')
    assert sha256_file(source/'data/direct/A/train/trajectories.npz')==inventory['files']['data/direct/A/train/trajectories.npz']


def test_stale_or_partial_context_cannot_be_resumed_as_ready(tmp_path):
    from poc.control_retraining import copy_retraining_context,retraining_context_inventory
    source=tmp_path/'source';source.mkdir();parameters,parent=context_fixture(source)
    inventory=retraining_context_inventory(source,parameters,[parent]);target=tmp_path/'fresh'
    copy_retraining_context(source,target,inventory)
    (target/'poc/engine.py').write_text('changed')
    with pytest.raises(ValueError):copy_retraining_context(source,target,inventory)
    partial=tmp_path/'partial';partial.mkdir()
    with pytest.raises(ValueError):copy_retraining_context(source,partial,inventory)


def test_retraining_prepare_cannot_launch_from_a_partial_replay(tmp_path):
    from poc.control_retraining import prepare
    from poc.io import atomic_json
    atomic_json(tmp_path/'reports/control_replay.json',dict(state='running',completed_episodes=1))
    with pytest.raises(ValueError,match='Complete.*replay'):prepare(tmp_path)
    assert not (tmp_path/'config/control_retraining_jobs.json').exists()


def test_fresh_runner_launches_once_then_reverifies_complete_fit_without_training_again(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import poc.control_retraining as module
    from poc.io import atomic_json,read_json,sha256_file
    source=tmp_path/'source';source.mkdir();parameters,parent=context_fixture(source)
    inventory=module.retraining_context_inventory(source,parameters,[parent])
    job=dict(cell_id='direct__A__shared__power_correct__seed0',definition=dict(parameters),
        source='source',context='fresh',inventory=inventory,original=dict(manifest='original/manifest.json'))
    registry=dict(jobs=[job],input_hashes={})
    calls=[]
    def child(argv,**kwargs):
        calls.append(argv)
        folder=tmp_path/'fresh/evidence/M30/new';folder.mkdir(parents=True)
        atomic_json(folder/'manifest.json',dict(fixture=True))
        return SimpleNamespace(returncode=0,stdout='{"manifest_path":"evidence/M30/new/manifest.json"}\n',stderr='')
    monkeypatch.setattr(module,'verify_registry',lambda root,value:value)
    monkeypatch.setattr(module,'_observe_fit',lambda root,job,path:dict(cell_id=job['cell_id'],
        manifest=str(path.relative_to(root)),manifest_sha256=sha256_file(path),
        comparison={'status':'pass','diagnosis_required':False},resources={'wall_seconds':1.}))
    monkeypatch.setattr(module.subprocess,'run',child)
    monkeypatch.setattr('poc.io._process_start',lambda pid:'private-metadata')
    first=module.run_fits(tmp_path,registry,'a'*64,resume=False)
    assert first['state']=='complete' and len(calls)==1 and first['control_checked'] is False
    digest=sha256_file(tmp_path/'reports/control_retraining.json')
    assert module.run_fits(tmp_path,registry,'a'*64,resume=True)==first
    assert len(calls)==1 and sha256_file(tmp_path/'reports/control_retraining.json')==digest
    assert 'replication' in calls[0][2]
    atomic_json(tmp_path/'fresh/evidence/M30/new/manifest.json',dict(changed=True))
    with pytest.raises(ValueError):module.run_fits(tmp_path,registry,'a'*64,resume=True)


def test_failed_fresh_fit_is_retained_instead_of_reusing_original_checkpoint(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import poc.control_retraining as module
    from poc.io import read_json
    source=tmp_path/'source';source.mkdir();parameters,parent=context_fixture(source)
    registry=dict(input_hashes={},jobs=[dict(cell_id='fixed-cell',definition=dict(parameters),source='source',
        context='fresh',inventory=module.retraining_context_inventory(source,parameters,[parent]))])
    monkeypatch.setattr(module,'verify_registry',lambda root,value:value)
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout='',stderr='fit interrupted'))
    monkeypatch.setattr('poc.io._process_start',lambda pid:'private-metadata')
    with pytest.raises(RuntimeError):module.run_fits(tmp_path,registry,'a'*64)
    receipt=read_json(tmp_path/'reports/control_retraining.json')
    assert receipt['state']=='interrupted' and receipt['completed_fits']==0
    assert len(list((tmp_path/'reports/control_retraining_attempts').glob('*/stderr.log')))==1


def test_independently_recomputed_validation_must_match_saved_selection_and_counts():
    from poc.control_retraining import assert_validation_scores
    observed=dict(objective=.02,one_step=.01,four_step=.04,one_count=8192,four_count=7424)
    assert_validation_scores(observed,dict(observed,epoch=100))
    changed=dict(observed);changed['four_count']-=1
    with pytest.raises(ValueError):assert_validation_scores(observed,changed)
    changed=dict(observed);changed['objective']*=1.01
    with pytest.raises(ValueError):assert_validation_scores(observed,changed)


def test_retraining_registry_rejects_changed_audit_implementation_before_using_cached_fit(tmp_path):
    from poc.control_retraining import verify_registry
    from poc.io import sha256_file
    path=tmp_path/'poc/train.py';path.parent.mkdir();path.write_text('frozen audit')
    registry=dict(input_hashes={'poc/train.py':sha256_file(path)})
    path.write_text('changed audit')
    with pytest.raises(ValueError,match='Changed retraining dependency'):verify_registry(tmp_path,registry)
