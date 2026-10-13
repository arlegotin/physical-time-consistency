import subprocess,sys,json
from pathlib import Path
raw=Path('/private/tmp/physical-time-consistency-short-calibration-replication')
runner=raw/'execution-logs/run_stage.py'
commands=[['prepare'],['collect','--session','0']]+[['train','--session','0','--model',m,'--budget','30','--initialization','0'] for m in ('affine_rk4','node_actuator_aware','physics_residual')]
for command in commands:
 result=subprocess.run([sys.executable,str(runner),str(raw),'pilot',*command])
 if result.returncode:sys.exit(result.returncode)
with (raw/'execution-logs/numerical-replication.stdout').open('w') as out,(raw/'execution-logs/numerical-replication.stderr').open('w') as err:
 result=subprocess.run([sys.executable,'/private/tmp/physical-time-consistency-short-calibration/execution-logs/replicate_numeric.py','/private/tmp/physical-time-consistency-short-calibration',str(raw)],stdout=out,stderr=err)
print(json.dumps({'numerical_replication_returncode':result.returncode}),flush=True)
sys.exit(result.returncode)
