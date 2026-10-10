from pathlib import Path
import os,subprocess,time,json,signal
root=Path.cwd();parent=25900;worker=8061
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONPATH']=str(root)
def command(pid):return subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True,check=False).stdout.strip()
old=command(parent);child=command(worker)
assert old.endswith('.superpowers/sdd/2026-10-08-affine-flow-refinement/precision-practical-queue.py'),old
assert child.endswith('-m poc.affine_precision finer'),child
batch=json.loads((root/'reports/affine_precision_001_finer_physics_batch.json').read_text())
assert batch['state']=='partial' and len(batch['completed'])<44
receipt=dict(state='handoff',original_launcher_pid=parent,preserved_scientific_worker_pid=worker,
 original_launcher_command=old,scientific_worker_command=child,complete_finer_at_handoff=len(batch['completed']),
 user_authorization='Use another volume on this computer for heavy artifacts',
 reason='Existing sandboxed launcher cannot write migrated work contexts; preserve current worker and authorize only remaining fixed-design stages',
 recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
(root/'reports/affine_volume_continuation_launch.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
os.kill(parent,signal.SIGTERM)  # orchestration only; never signal the live scientific worker
assert command(worker).endswith('-m poc.affine_precision finer')
print('HANDOFF: original orchestration stopped; active finer-physics worker continues',worker,flush=True)
while True:
 batch=json.loads((root/'reports/affine_precision_001_finer_physics_batch.json').read_text())
 if batch['state']=='complete' and len(batch['completed'])==44:break
 print('WAITING for existing finer-physics worker',len(batch['completed']),'/44',flush=True)
 time.sleep(50)
commands=[['-m','poc.affine_fresh','retrain','--seed','2026100800'],['-m','poc.affine_fresh','retrain','--seed','2026100810'],['-m','poc.affine_precision','reproduce'],['-m','poc.affine_precision','aggregate','--look','3'],['reports/provenance/independent_affine_audit.py'],['-m','poc.affine_precision','demo']]
for args in commands:
 print('START',time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),args,flush=True)
 subprocess.run([str(root/'.venv/bin/python'),*args],cwd=root,check=True)
 print('COMPLETE',args,flush=True)
print('PRACTICAL_MEASUREMENT_AND_REPRODUCTION_QUEUE_COMPLETE',flush=True)
