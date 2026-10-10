import pytest
from reports.audit_nominal_selections import assert_scores


def values():
    return dict(objective=.01,one_step=.004,four_step=.024,one_count=8192,four_count=7424)


def test_consistent_selected_nominal_checkpoint_matches():
    assert_scores(values(),values())


@pytest.mark.parametrize('mutation',['old_weights','nan','missing','coverage'])
def test_torn_or_incomplete_nominal_selection_cannot_pass(mutation):
    recorded,observed=values(),values()
    if mutation=='old_weights':observed['objective']=.011
    if mutation=='nan':observed['four_step']=float('nan')
    if mutation=='missing':observed.pop('one_step')
    if mutation=='coverage':observed['four_count']=8192
    with pytest.raises(ValueError):assert_scores(observed,recorded)


def test_dry_run_lists_missing_frozen_cells_without_loading_models(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys
    import shutil
    root=Path(__file__).resolve().parents[1]
    context=tmp_path/'context';(context/'poc').mkdir(parents=True)
    for name in ('__init__.py','io.py'):
        shutil.copyfile(root/'poc'/name,context/'poc'/name)
    output=tmp_path/'audit.json'
    result=subprocess.run([sys.executable,str(root/'reports/audit_nominal_selections.py'),
        '--root',str(context),'--output',str(output),'--dry-run'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert report['jobs']==80 and len(report['missing_frozen_cells'])==80
    assert report['training'] is False and not output.exists()
    assert not list(context.rglob('__pycache__'))


def test_frozen_loader_events_are_redirected_and_original_sink_restored(tmp_path):
    from types import SimpleNamespace
    from reports.audit_nominal_selections import load_validation_for_audit
    frozen_journal = tmp_path/'frozen/evidence/data_access.jsonl'
    external_journal = tmp_path/'audit.data_access.jsonl'
    calls=[]
    def original_sink(path,event):
        calls.append((path,event))
    def loader(path,**kwargs):
        assert kwargs==dict(phase='confirmatory',role='validation',purpose='selection')
        module.append_event(frozen_journal,{'event':'dataset_open'})
        return {'known':'arrays'}
    module=SimpleNamespace(append_event=original_sink,load_corpus=loader)
    assert load_validation_for_audit(module,tmp_path/'manifest.json',external_journal)=={'known':'arrays'}
    assert calls==[(external_journal,{'event':'dataset_open'})]
    assert module.append_event is original_sink
    assert not frozen_journal.exists()


def test_frozen_loader_sink_is_restored_on_validation_failure(tmp_path):
    from types import SimpleNamespace
    from reports.audit_nominal_selections import load_validation_for_audit
    def original_sink(path,event):
        raise AssertionError('No access event should be written for a failed load')
    def loader(path,**kwargs):
        raise ValueError('corrupt corpus')
    module=SimpleNamespace(append_event=original_sink,load_corpus=loader)
    with pytest.raises(ValueError,match='corrupt corpus'):
        load_validation_for_audit(module,tmp_path/'manifest.json',tmp_path/'audit.jsonl')
    assert module.append_event is original_sink
