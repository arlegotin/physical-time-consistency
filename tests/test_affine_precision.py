import copy

import numpy as np
import pytest

from poc import affine_precision as precision


def request():
    return dict(old_seeds=list(range(2026100800,2026100805)),
                new_seeds=list(range(2026100805,2026100811)),total_seeds=11,
                episodes_per_seed=10,look_index=2,alpha=.0125)


def test_precision_seed_registry_rejects_replacement_overlap_and_nonascending_ids():
    assert precision.precision_seeds(request()) == list(range(2026100800,2026100811))
    for field,value in [('new_seeds',[2026100804,2026100805]),
                        ('new_seeds',[2026100806,2026100805]),
                        ('old_seeds',[2026100800,2026100802]),('total_seeds',10)]:
        damaged={**request(),field:value}
        with pytest.raises(ValueError,match='seed'):
            precision.precision_seeds(damaged)


def block(seed):
    raw=dict(probe_ids=np.arange(2),strata=np.array(['cold','warm']),
             horizons_ns=np.array([5000000,10000000]),truth_zero=np.zeros((2,2,6)),
             truth_command=np.zeros((2,2,6)),model_zero=np.zeros((2,2,6)),
             model_command=np.full((2,2,6),.01),model_composed=np.full((2,2,6),.02))
    raws={(name,seed):copy.deepcopy(raw) for name in ('affine_flow','node_structured')}
    profiles=[dict(model_id=name,seed=seed,batch_size=batch,h_s=h,
                   samples_ns=[100]*(500 if batch==1 else 100),median_ns=100.,p95_ns=100.)
              for name in ('affine_flow','node_structured') for batch in (1,128) for h in (.005,.01,.04)]
    return dict(raws=raws,profiles=profiles)


def test_precision_merge_recomputes_raw_rows_and_requires_every_original_and_new_seed():
    first,second=block(2026100800),block(2026100801)
    coordinates={k:first['raws'][('affine_flow',2026100800)][k] for k in ('probe_ids','strata','horizons_ns')}
    rows,profiles=precision.merge_measurements([first,second],[2026100800,2026100801],coordinates)
    assert len(rows)==32 and len(profiles)==24
    assert rows[0]['total_error_mean']==pytest.approx(.01*np.sqrt(2))
    with pytest.raises(ValueError,match='coverage'):
        precision.merge_measurements([first],[2026100800,2026100801],coordinates)
    with pytest.raises(ValueError,match='[Dd]uplicate'):
        precision.merge_measurements([first,first],[2026100800],coordinates)
    damaged=copy.deepcopy(second);damaged['raws'][('affine_flow',2026100801)]['model_command'][0,0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):
        precision.merge_measurements([first,damaged],[2026100800,2026100801],coordinates)


def test_merged_inference_keeps_a_real_loss_and_uses_all_training_pairs():
    seeds=list(range(2026100800,2026100805)); blocks=[block(s) for s in seeds]
    coordinates={k:blocks[0]['raws'][('affine_flow',seeds[0])][k] for k in ('probe_ids','strata','horizons_ns')}
    for i,b in enumerate(blocks):
        b['raws'][('affine_flow',seeds[i])]['model_command']*=1.5+i*.1
        for p in b['profiles']:
            if p['model_id']=='affine_flow':
                p['samples_ns']=[200+i]*len(p['samples_ns']);p['median_ns']=200.+i;p['p95_ns']=200.+i
    rows,profiles=precision.merge_measurements(blocks,seeds,coordinates)
    result=precision.prediction_inference(rows,profiles,seeds,2,{'q':0.,'v':0.})
    assert not result['primary_combined_gain_supported']
    assert all(r['contrast']['n']==5 for r in result['predictions'])
    assert all(r['point_improvement']<0 for r in result['predictions'])
    assert result['predictions'][0]['contrast']['upper']<0


def test_precision_entrypoint_sets_thread_environment_before_real_numerical_imports():
    import os
    from pathlib import Path
    import subprocess
    import sys
    variables=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
    script='''
import os,sys
variables=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
class ThreadGuard:
    def find_spec(self,name,*args):
        if name in ('numpy','torch'):
            assert all(os.environ.get(v)=='1' for v in variables), 'late thread configuration: '+name
        return None
sys.meta_path.insert(0,ThreadGuard())
import poc.affine_precision
'''
    result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).parents[1],
        env={**os.environ,**{v:'2' for v in variables}},capture_output=True,text=True)
    assert result.returncode==0,result.stderr
