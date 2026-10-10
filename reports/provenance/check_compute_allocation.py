"""Prospective timing-only check: shared normal banks are prepared before timing.

No confirmation episode is executed or scored by this audit. The original
calibration includes bank preparation; actual episodes prepare it beforehand.
This independent check guards against an allocation mismatch from that overhead.
"""
from pathlib import Path
import time
import numpy as np
import torch
from poc.compute_study import CONFIGS,MODELS,ALLOCATION,context,load_model
from poc.compute_study_runs import normal_bank,task_conditions
from poc.execution import cpu_defaults
from poc.io import atomic_json,read_json,sha256_file
from poc.stateless_mpc import search


def main():
    root=Path.cwd();cpu_defaults(2026100800);f,store=context(root)
    allocation=read_json(root/ALLOCATION);store.read(allocation['receipt'])
    models={n:load_model(root,n,f['seeds'][0]) for n in MODELS}
    inputs=task_conditions(9000,0);state=torch.tensor(inputs['initial_state']);target=torch.tensor(inputs['target'])
    readings={}
    for repeat in range(4):
        bank=normal_bank(9000,200+repeat,allocation['maximum_candidates'],allocation['maximum_candidates'])[0]
        order=CONFIGS[repeat:]+CONFIGS[:repeat]
        for name,step in order:
            key=f'{name}:{step}';cfg=allocation['settings'][key]
            start=time.perf_counter_ns()
            out=search(models[name],state,target,None,None,bank[:,:cfg['candidate_count']],cfg,
                       (step,)*(40_000_000//step),40_000_000)
            elapsed=(time.perf_counter_ns()-start)/1e9
            if out['status']!='ok':raise ValueError('Calibration search failure')
            if repeat:readings.setdefault(key,[]).append(elapsed)
    target_seconds=float(np.median(readings['node_structured:5000000']))
    ratios={k:float(np.median(v))/target_seconds for k,v in readings.items()}
    value=dict(allocation_sha256=sha256_file(root/ALLOCATION),source_sha256=sha256_file(Path(__file__)),
               samples_seconds=readings,reference_seconds=target_seconds,ratios=ratios,
               matched=all(abs(r-1)<=.1 for r in ratios.values()),
               scope='Timing-only development check; shared RNG preparation outside actual planner timer')
    atomic_json(root/'reports/compute_study_allocation_check.json',value);print(value)
    if not value['matched']:raise ValueError('Prospective direct timer mismatch; refine allocation before opening new tasks')


if __name__=='__main__':main()
