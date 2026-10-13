"""Execute the archived fresh-root numerical driver with separate resources."""
from pathlib import Path
import datetime, hashlib, json, os, resource, subprocess, sys, time
from run_study import ROOT, PRIMARY, REPLICA, run

if __name__=='__main__':
    logs=REPLICA/'execution-logs'; logs.mkdir(parents=True,exist_ok=True)
    driver=ROOT/'reports/hybrid_component/replication_driver.py'
    command=[str(ROOT/'.venv/bin/python'),str(driver),str(PRIMARY),str(REPLICA)]
    path=logs/'numerical-replication.execution.json'
    if path.exists():raise RuntimeError('Numerical execution already recorded')
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    before=resource.getrusage(resource.RUSAGE_CHILDREN); start=time.perf_counter()
    record={'command':command,'cwd':str(ROOT),'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'driver_sha256':hashlib.sha256(driver.read_bytes()).hexdigest(),
            'orchestration_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Independent replica numerical process; excluded from all primary resource totals'}
    path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    with (logs/'numerical-replication.stdout').open('w') as out,(logs/'numerical-replication.stderr').open('w') as err:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err)
        while process.poll() is None:
            try:process.wait(timeout=45)
            except subprocess.TimeoutExpired:print(json.dumps({'event':'running','stage':'numerical-replication','elapsed_wall_s':time.perf_counter()-start}),flush=True)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    record.update(returncode=process.returncode,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  elapsed_wall_s=time.perf_counter()-start,
                  child_cpu_s=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
    path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    if process.returncode:raise SystemExit(process.returncode)
    run(REPLICA,'verify')
