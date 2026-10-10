from pathlib import Path
import os,subprocess,time,json
root=Path.cwd()
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONPATH']=str(root)
assert json.loads((root/'reports/storage_venv_relocation.json').read_text())['environment_unchanged']
commands=[['-m','poc.affine_fresh','retrain','--seed','2026100800'],['-m','poc.affine_fresh','retrain','--seed','2026100810'],['-m','poc.affine_precision','reproduce'],['-m','poc.affine_precision','aggregate','--look','3'],['reports/provenance/independent_affine_audit.py'],['-m','poc.affine_precision','demo']]
for args in commands:
 print('START',time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),args,flush=True)
 subprocess.run([str(root/'.venv/bin/python'),*args],cwd=root,check=True)
 if args==['-m','poc.affine_precision','reproduce']:
  assert json.loads((root/'reports/affine_precision_001_reproduction.json').read_text())['pass'] is True,'Reproduction reports failure'
 print('COMPLETE',args,flush=True)
print('PRACTICAL_MEASUREMENT_AND_REPRODUCTION_QUEUE_COMPLETE',flush=True)
