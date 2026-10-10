import copy
import json

import numpy as np
import pytest

from poc.affine_release import prepare_fresh_clips, render_fresh_replay


def clips():
    return [dict(model=mode,seed=seed,task=task,time_s=(np.arange(161)/100).tolist(),
                 q=np.zeros((161,2)).tolist(),target=[.2,-.1],cost=1.,planner_seconds=.1)
            for seed in [2026100800,2026100801] for task in range(2)
            for mode in ['affine_flow_5','node_structured_5','node_structured_40','affine_flow_40']]


def test_fresh_replay_cannot_omit_losses_or_mismatch_paired_tasks():
    values=clips()
    assert len(prepare_fresh_clips(values,[2026100800,2026100801],2))==16
    for bad in [values[:-1],values+[values[0]]]:
        with pytest.raises(ValueError,match='coverage'):
            prepare_fresh_clips(bad,[2026100800,2026100801],2)
    changed=copy.deepcopy(values);changed[1]['target']=[.3,-.1]
    with pytest.raises(ValueError,match='paired'):
        prepare_fresh_clips(changed,[2026100800,2026100801],2)


def test_fresh_replay_embeds_actual_cost_and_physical_motion():
    values=clips();values[0]['cost']=1.23
    html=render_fresh_replay(values,[2026100800,2026100801],2)
    start=html.index('<script type="application/json" id="clips">')
    start=html.index('>',start)+1
    end=html.index('</script>',start)
    raw=json.loads(html[start:end])
    assert raw[0]['cost']==1.23
    assert raw[0]['time_s'][-1]==1.6
    assert len(raw)==16
