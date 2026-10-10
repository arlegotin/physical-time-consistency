from pathlib import Path
import json, os, subprocess, sys, time
root=Path.cwd()
print('Report handoff started; waiting for all frozen pilot cells.',flush=True)
while True:
    batch=json.loads((root/'reports/efficiency_pilot_batch.json').read_text())
    if batch['state']=='complete':
        break
    try:
        os.kill(42956,0)
    except PermissionError:
        pass
    except ProcessLookupError:
        raise RuntimeError('Experiment coordinator stopped before complete evidence; inspect preserved execution log. No report result inferred.')
    last=(root/'reports/efficiency_pilot_execution.log').read_text().splitlines()[-1]
    if 'RuntimeError' in last or 'Traceback' in last:
        raise RuntimeError('Experiment reports a failure; preserve it and diagnose before reporting.')
    print(f'Waiting: {len(batch["completed"])} / 408 complete',flush=True)
    time.sleep(50)
print('Complete batch found; invoking the existing strict report command.',flush=True)
subprocess.run([sys.executable,'-m','poc.efficiency_result','report'],cwd=root,check=True)
print('Final pilot artifacts generated successfully.',flush=True)
