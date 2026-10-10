"""Frozen task-only precision: preserve weights and add paired episode IDs."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from .compute_study import CONFIGS,MODELS,ALLOCATION,StudyStore,context as initial_context
from .compute_study_precision import context as seed_context
from .io import atomic_json,read_json,sha256_file,utc_now

REQUEST='config/compute_study_episode_003.json'
FREEZE='config/compute_study_episode_003_freeze.json'
PREFIX='reports/compute_study_episode_003'
PARENT='reports/provenance/compute_study_look_002_result.json'


def validate_request(r):
    seeds=r['seeds'];old=r['prior_episodes_per_seed'];total=r['episodes_per_seed']
    if (r['look']!=3 or len(seeds)<5 or any(type(s) is not int for s in seeds)
            or seeds!=sorted(set(seeds)) or old!=10 or total!=2*old
            or r['new_episode_ids']!=list(range(old,total))
            or not .7<r['within_episode_fraction']<=1
            or not math.isfinite(r['variance']) or r['variance']<=0
            or not math.isfinite(r['mean_contrast']) or r['planning_sd_margin']!=1.25
            or not r['reason'] or any(len(r[k])!=64 for k in ('prior_result_sha256','allocation_sha256'))
            or any(not math.isfinite(v) or v<=0 for v in r['resource_projection'].values())):
        raise ValueError('Prospective paired episode expansion and measured within-task uncertainty required')
    return r['new_episode_ids']


def episode_ids(r,kind):
    ids=validate_request(r)
    if kind not in ('primary','replay','finer'):raise ValueError('Unregistered episode cohort')
    return ids if kind=='primary' else ids[:1]


def prepare(root):
    root=Path(root);base,old_store=initial_context(root);prior,previous_store=seed_context(root)
    r=read_json(root/REQUEST);validate_request(r);parent=read_json(root/PARENT)
    if (r['seeds']!=base['seeds']+prior['request']['new_seeds'] or r['seeds']!=parent['seeds']
            or r['prior_episodes_per_seed']!=parent['episodes_per_seed']
            or sha256_file(root/PARENT)!=r['prior_result_sha256']
            or sha256_file(root/ALLOCATION)!=r['allocation_sha256']):
        raise ValueError('Changed episode precision parent evidence')
    if (root/FREEZE).exists():return context(root)[0]
    hashes={**previous_store.hashes,REQUEST:sha256_file(root/REQUEST),PARENT:sha256_file(root/PARENT),
            'poc/compute_study_episodes.py':sha256_file(Path(__file__)),
            r['contract_path']:sha256_file(root/r['contract_path']),
            r['variance_evidence']:sha256_file(root/r['variance_evidence'])}
    frozen=dict(request=r,source_hashes=hashes,store_root=str(old_store.base/'episode-003'),created_utc=utc_now())
    atomic_json(root/FREEZE,frozen);return frozen


def context(root):
    root=Path(root);seed_context(root);f=read_json(root/FREEZE);r=read_json(root/REQUEST);validate_request(r)
    if f['request']!=r:raise ValueError('Changed episode request')
    hashes={**f['source_hashes'],FREEZE:sha256_file(root/FREEZE)}
    store=StudyStore(f['store_root'],{k:root/k for k in hashes},hashes);store.check_inputs();return f,store


def load_registered_model(root,name,seed):
    from .compute_study import load_model
    from .compute_study_precision import load_model as load_added
    f,_=context(root);base,_=initial_context(root)
    if name not in MODELS or seed not in f['request']['seeds']:raise ValueError('Unregistered model/seed')
    return load_model(root,name,seed) if seed in base['seeds'] else load_added(root,name,seed)


def controls(root,kind='primary'):
    from .compute_study import selected_receipt
    from .compute_study_precision import fit_path
    from .compute_study_runs import execute_episode,npz
    f,store=context(root);base,old_store=initial_context(root);allocation=read_json(Path(root)/ALLOCATION)
    old_store.read(allocation['receipt']);ids=episode_ids(f['request'],kind)
    path=Path(root)/f'{PREFIX}_{kind}.json';ledger=read_json(path) if path.exists() else dict(kind=kind,completed=[],state='running')
    for seed in f['request']['seeds']:
        models={name:load_registered_model(root,name,seed) for name in MODELS}
        for episode in ids:
            for name,step in CONFIGS:
                settings=allocation['settings'][f'{name}:{step}']
                selected=(selected_receipt(root,name,seed,base) if seed in base['seeds']
                          else read_json(fit_path(root,name,seed)))
                def produce(folder):
                    raw,summary=execute_episode(models[name],settings,seed,episode,step,125000 if kind=='finer' else 250000,
                        allocation['maximum_candidates'],Path(root)/'assets')
                    npz(folder/'episode.npz',**raw);summary.update(model_id=name,kind=kind);return summary
                receipt=store.produce('control',dict(model_id=name,seed=seed,episode_id=episode,step_ns=step,kind=kind,
                    settings=settings,maximum=allocation['maximum_candidates'],allocation_sha256=sha256_file(Path(root)/ALLOCATION),selected_model=selected),produce)
                if receipt not in ledger['completed']:ledger['completed'].append(receipt)
                ledger.update(completed_count=len(ledger['completed']),updated_utc=utc_now());atomic_json(path,ledger)
                print(f'Episode precision {kind} {name}/{step//1000000}ms seed{seed} task{episode} [{len(ledger["completed"])}]',flush=True)
    ledger['state']='complete';atomic_json(path,ledger)


def records(root,kind):
    _,store=context(root);ledger=read_json(Path(root)/f'{PREFIX}_{kind}.json')
    if ledger['state']!='complete':raise ValueError('Incomplete episode precision cohort')
    rows=[];paths=[]
    for receipt in ledger['completed']:
        folder,summary=store.read(receipt);rows.append(summary);paths.append(folder)
    return rows,paths,ledger


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','controls','reproduce'))
    p.add_argument('--root',type=Path,default=Path.cwd());args=p.parse_args(argv)
    from .execution import cpu_defaults
    cpu_defaults(2026100800)
    if args.command=='prepare':prepare(args.root)
    elif args.command=='controls':controls(args.root)
    else:controls(args.root,'replay');controls(args.root,'finer')


if __name__=='__main__':main()
