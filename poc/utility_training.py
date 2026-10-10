"""Train only the missing nominal fits for the prospective control cohort."""
from __future__ import annotations

import argparse
from pathlib import Path

from .control_confirmation import _check_selected_fit, expected_checkpoints, freeze_cells
from .io import append_event, atomic_json, job_key, job_lock, read_json, safe_path, sha256_file, utc_now


def validate_training_request(request: dict, design: dict) -> dict:
    if (request['protocol_id']!=design['protocol_id'] or request['phase']!='confirmatory'
            or request['context']!='study/control-development'):
        raise ValueError('Changed prospectively registered nominal training scope')
    reused=request['reused_selected']
    reuse_keys=[(r['actuator'],r['model_id'],r['seed']) for r in reused]
    expected_reuse={('direct',m,s) for m in ('node_generic','power_correct') for s in range(5)}
    if len(reused)!=10 or set(reuse_keys)!=expected_reuse:
        raise ValueError('Only the ten already audited confirmatory fits may be reused')
    missing=expected_checkpoints(design)-expected_reuse
    cells=request['confirmation_cells_to_train']
    keys=[]
    for cell in cells:
        a,m,s=cell['actuator'],cell['model_id'],cell['seed']
        expected=dict(actuator=a,model_id=m,seed=s,condition='A',loss_mode='shared',
                      cell_id=f'{a}__A__shared__{m}__seed{s}')
        if cell!=expected or type(s) is not int:
            raise ValueError('Changed nominal data, loss or independent seed')
        keys.append((a,m,s))
    if len(cells)!=20 or set(keys)!=missing:
        raise ValueError('Incomplete or duplicated missing nominal fits')
    return request


def require_completed_development(root: Path) -> None:
    path=Path(root)/'reports/control_nominal_completion.json'
    if not path.exists():
        raise ValueError('Complete original nominal development fits before starting this serial cohort')
    receipt=read_json(path)
    request=read_json(Path(root)/'config/control_development.json') if (Path(root)/'config/control_development.json').exists() else None
    if (receipt.get('state')!='complete' or request is None or len(receipt.get('cells',[]))!=7
            or receipt['request_sha256']!=sha256_file(Path(root)/'config/control_development.json')
            or [r['cell'] for r in receipt['cells']]!=request['nominal_fits_to_complete']):
        raise ValueError('Complete original nominal development fits before starting this serial cohort')
    from .control_models import load_selected
    for result in receipt['cells']:
        cell=result['cell'];manifest=safe_path(Path(root),result['manifest'])
        record=read_json(manifest)
        if (record['milestone_id']!='M30' or record['phase']!='exploratory'
                or any(record['job_inputs']['parameters'].get(k)!=v for k,v in cell.items())):
            raise ValueError('Changed original development fit identity, data or objective')
        load_selected(manifest,result['manifest_sha256'],sha256_file(manifest.parent/'weights.pt'),
                      actuator=cell['actuator'],model_id=cell['model_id'],seed=cell['seed'])


def plumbing_requests(request: dict) -> list:
    return [('controls',dict(actuator=a,model_id=m,seed=8100)) for a,m in
            sorted({(r['actuator'],r['model_id']) for r in request['confirmation_cells_to_train']})]


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);owner=args.root.resolve()
    request_path=owner/'config/control_confirmation_training.json'
    request_hash=sha256_file(request_path);request=read_json(request_path)
    design=read_json(owner/'config/control_confirmation_design.json')
    validate_training_request(request,design)
    if request['design_sha256']!=sha256_file(owner/'config/control_confirmation_design.json'):
        raise ValueError('Changed prospective control design')
    if sha256_file(owner/'reports/selection_integrity_forty.json')!=request['selection_audit_sha256']:
        raise ValueError('Changed selected-checkpoint audit for reused confirmatory fits')
    context=safe_path(owner,request['context'])
    if sha256_file(context/'INITIAL_CONTEXT.json')!=request['context_inventory_sha256']:
        raise ValueError('Changed writable training-context provenance')
    for filename,expected in {**request['engine_source_hashes'],**request['data_hashes']}.items():
        if sha256_file(safe_path(context,filename))!=expected:
            raise ValueError('Changed registered nominal engine or corpus: '+filename)
    if sha256_file(owner/'poc/mechanism.py')!=request['engine_source_hashes']['poc/mechanism.py']:
        raise ValueError('Parent nominal coordinator differs from its registered engine')
    if args.dry_run:
        print(dict(execute=False,reused_checkpoints=10,missing_nominal_fits=20,
                   freeze_complete=(owner/'config/control_confirmation_cells.json').exists(),
                   prerequisite='Complete the original seven nominal development fits',
                   scientific_outcomes_accessed=False))
        return 0
    require_completed_development(owner)
    for item in request['reused_selected']:
        _check_selected_fit(owner,item)
    from .mechanism import run_worker,train_cell
    from .train import qualify_nominal
    from .io import run_directory
    folder=context/'control_utility_nominal';folder.mkdir(exist_ok=True)
    receipt_path=owner/'reports/control_confirmation_nominal.json'
    if receipt_path.exists() and not args.resume:
        raise ValueError('Existing nominal cohort requires --resume')
    selected=list(request['reused_selected'])
    with job_lock(job_key({'utility_training_sha256':request_hash}),owner):
        try:
            for command,parameters in plumbing_requests(request):
                run_worker(context,folder,command,parameters,args.resume)
            for cell in request['confirmation_cells_to_train']:
                if sha256_file(request_path)!=request_hash:
                    raise ValueError('Nominal cohort registration changed during execution')
                append_event(folder/'events.jsonl',dict(event='control_nominal_fit_start',cell=cell,utc=utc_now()))
                record=train_cell(context,folder,cell,'confirmatory',resume=args.resume)
                if not qualify_nominal(record,context)['qualified']:
                    raise ValueError('Valid nominal optimization remains unqualified; retain and diagnose this fit')
                path=run_directory(context,record)/'manifest.json'
                selected.append(dict(actuator=cell['actuator'],model_id=cell['model_id'],seed=cell['seed'],
                    manifest=str(path.relative_to(owner)),manifest_sha256=sha256_file(path),
                    checkpoint_sha256=sha256_file(path.parent/'weights.pt'),
                    canonical_tensor_sha256=record['model_hashes']['canonical_tensor_sha256']))
                atomic_json(receipt_path,dict(state='running',request_sha256=request_hash,selected=selected,
                    selection='Nominal validation only',updated_utc=utc_now()))
            frozen_path=owner/'config/control_confirmation_cells.json'
            if not frozen_path.exists():
                freeze_cells(owner,design,selected)
            else:
                from .control_confirmation import verify_freeze
                verify_freeze(owner,design,read_json(frozen_path))
            atomic_json(receipt_path,dict(state='complete',request_sha256=request_hash,selected=selected,
                selection='Nominal validation only',updated_utc=utc_now()))
            return 0
        except BaseException as error:
            atomic_json(receipt_path,dict(state='interrupted',request_sha256=request_hash,selected=selected,
                selection='Nominal validation only',reason=f'{type(error).__name__}: {error}',updated_utc=utc_now()))
            raise


if __name__=='__main__':
    raise SystemExit(main())
