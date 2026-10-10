"""Read committed utility-run progress; monitoring never assigns a scientific verdict."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from .io import atomic_json, job_key, job_lock, read_json, sha256_file, utc_now


def _training(root,folder_name):
    context=root/'study/control-development'
    journal=context/folder_name/'events.jsonl'
    if not journal.exists():return dict(status='no_training_journal')
    try:
        events=[json.loads(line) for line in journal.read_text().splitlines()]
        starts=[(i,event) for i,event in enumerate(events) if event.get('event')=='child_job_start']
        if not starts:return dict(status='no_training_worker_recorded')
        index,start=starts[-1]
        if start['command']!='train':return dict(status='learning_controls',request_sha256=start['request_hash'])
        request_path=context/'config/job_requests'/f"train-{start['request_hash']}.json"
        request=read_json(request_path)
        if job_key({'command':'train','request':request})!=start['request_hash'] or request['purpose']!='training':
            raise ValueError('Training journal request hash/purpose mismatch')
        semantic=request['semantic_inputs']
        matches=[]
        for path in (context/'evidence/M30').glob('*/request.json'):
            value=read_json(path);parameters=value['inputs']['parameters']
            if (value['inputs']['phase']==request['phase']
                    and all(parameters.get(k)==v for k,v in semantic.items())):
                matches.append(path)
        partial=[p for p in matches if not (p.parent/'COMPLETED').exists()]
        candidates=partial if partial else matches
        if not candidates:return dict(status='worker_starting',request_sha256=start['request_hash'],
            cell_id=semantic['cell_id'],target_epochs=semantic['epochs'])
        if len(candidates)!=1:raise ValueError('Multiple matching worker attempts; no modification-time selection')
        path=candidates[0];marker=path.parent/'resume.complete.json'
        committed=read_json(marker) if marker.exists() else None
        completed=any(event.get('event')=='child_job_complete' for event in events[index+1:])
        return dict(status='worker_completion_recorded' if completed else 'checkpoint_progress',
            request_sha256=start['request_hash'],cell_id=semantic['cell_id'],target_epochs=semantic['epochs'],
            source_request=str(path.relative_to(root)),source_request_sha256=sha256_file(path),
            committed_optimizer_updates=committed['update'] if committed else None,
            recorded_checkpoint_sha256=committed['checkpoint_sha256'] if committed else None,
            checkpoint_tensor_hash_verified=False,
            note='Progress marker only; no process-liveness claim or scientific qualification')
    except (ValueError,KeyError,OSError,TypeError) as error:
        return dict(status='unresolved_journal',reason=f'{type(error).__name__}: {error}')


def _stage(root,name,expected):
    path=root/'reports'/name
    if not path.exists():return dict(state='not_started',completed_episodes=None,expected_episodes=expected)
    try:
        value=read_json(path)
        return dict(state=value['state'],completed_episodes=value.get('completed_episodes'),
            expected_episodes=value.get('expected_episodes',expected),reason=value.get('reason'),
            receipt=str(path.relative_to(root)),receipt_sha256=sha256_file(path))
    except (ValueError,KeyError,OSError,TypeError) as error:
        return dict(state='unresolved_receipt',completed_episodes=None,expected_episodes=expected,
                    reason=f'{type(error).__name__}: {error}')


def observe(root):
    root=Path(root).resolve()
    nominal=root/'reports/control_nominal_completion.json'
    record=read_json(nominal) if nominal.exists() else dict(state='not_started',cells=[])
    confirmation=root/'reports/control_confirmation_nominal.json'
    selected=read_json(confirmation) if confirmation.exists() else dict(state='not_started',selected=[])
    return dict(schema_version=1,observed_utc=utc_now(),scientific_decision='not_evaluated',
        scope='Operational progress snapshots; scientific completeness/validity requires the separate audits',
        disk_free_bytes=shutil.disk_usage(root).free,
        original_nominal=dict(receipt_state=record['state'],completed_fits=len(record['cells']),
            expected_fits=7,reason=record.get('reason'),training=_training(root,'control_nominal_completion')
                if record['state']!='complete' else dict(status='completed_cohort_receipt')),
        confirmation_nominal=dict(receipt_state=selected['state'],selected_fits=len(selected['selected']),
            expected_fits=30,reason=selected.get('reason'),training=_training(root,'control_utility_nominal')
                if selected['state']!='complete' else dict(status='completed_cohort_receipt')),
        remaining_development=_stage(root,'control_development_remaining.json',140),
        primary_control=_stage(root,'control_confirmation_batch.json',1200),
        physical_references=_stage(root,'control_reference_batch.json',450),
        finer_physics=_stage(root,'control_refinement_subset_batch.json',300),
        solver_audit=_stage(root,'control_confirmation_solver.json',100))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--watch',action='store_true',help='Read-only experiment monitor; never launch jobs')
    parser.add_argument('--interval',type=float,default=60.)
    args=parser.parse_args(argv);root=args.root.resolve()
    if not 0<args.interval<float('inf'):
        parser.error('Monitoring interval must be finite and positive')
    key=hashlib.sha256(b'utility operational monitor').hexdigest()
    with job_lock(key,root):
        while True:
            result=observe(root);atomic_json(root/'reports/control_live_monitor.json',result)
            print(json.dumps(result,allow_nan=False),flush=True)
            if not args.watch:return 0
            time.sleep(args.interval)


if __name__=='__main__':
    raise SystemExit(main())
