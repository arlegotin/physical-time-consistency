"""Exact serial CLI orchestration; no scientific implementation lives here."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, resource, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = Path('/private/tmp/physical-time-consistency-hybrid-component')
REPLICA = Path('/private/tmp/physical-time-consistency-hybrid-component-replication')
MODELS = ('hybrid_affine_flow', 'hybrid_affine_rk4', 'hybrid_affine_aware')

def run(raw, stage, *extra):
    logs = raw / 'execution-logs'
    logs.mkdir(parents=True, exist_ok=True)
    label = '-'.join((stage, *[x for x in extra if not x.startswith('--')]))
    path = logs / (label + '.execution.json')
    if path.exists():
        raise RuntimeError(f'Execution record already exists: {path}')
    command = [str(ROOT/'.venv/bin/python'), '-m', 'poc.calibration_study', stage,
               '--matrix', 'hybrid', '--output-root', str(raw), *extra]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    started = time.perf_counter(); before = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = {'command':command, 'cwd':str(ROOT), 'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'thread_environment':{k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')},
              'accounting_scope':'inclusive child process per serial CLI; never add to overlapping producer ledgers'}
    path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'event':'start','command':command}), flush=True)
    with (logs/(label+'.stdout')).open('w') as out, (logs/(label+'.stderr')).open('w') as err:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err)
        while process.poll() is None:
            try: process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                print(json.dumps({'event':'running','stage':label,'elapsed_wall_s':time.perf_counter()-started}), flush=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    record.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  elapsed_wall_s=time.perf_counter()-started,
                  child_cpu_s=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
                  returncode=process.returncode)
    path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    if process.returncode: raise SystemExit(process.returncode)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode', choices=('primary','replica','stage'))
    parser.add_argument('stage', nargs='?'); args, extra=parser.parse_known_args()
    if args.mode=='primary':
        for stage in ('prepare','collect','train','predict','decisions','time'): run(PRIMARY,stage)
        run(PRIMARY,'report','--destination',str(ROOT/'reports/hybrid_component'))
        run(PRIMARY,'verify')
    elif args.mode=='replica':
        run(REPLICA,'prepare'); run(REPLICA,'collect','--session','1')
        for model in MODELS: run(REPLICA,'train','--session','1','--model',model,'--budget','30','--initialization','0')
    else:
        if args.stage is None: parser.error('stage required')
        run(PRIMARY,args.stage,*extra)
