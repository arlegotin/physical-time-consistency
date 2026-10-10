"""Independent selected-plan next-interval errors on saved physical anchors."""
from __future__ import annotations

import math
import numpy as np


def first_interval_metrics(decisions,anchors,segments,target):
    truth=np.asarray(anchors,dtype=np.float64);goal=np.asarray(target,dtype=np.float64)
    if (truth.shape not in ((40,4),(40,6)) or goal.shape!=(2,)
            or not np.isfinite(truth).all() or not np.isfinite(goal).all()
            or len(decisions)!=40 or len(segments)!=40):
        raise ValueError('Forty real common anchors/decisions and a finite target required')
    rows=[];q_squared=[];v_squared=[];gaps=[];actual_costs=[];failures=0
    for i,(decision,state,segment) in enumerate(zip(decisions,truth,segments)):
        now=i*40_000_000;command=np.asarray(segment['command'],dtype=np.float64)
        if (decision['time_ns']!=now or segment['start_ns']!=now or segment['duration_ns']!=40_000_000
                or command.shape!=(2,) or not np.isfinite(command).all() or np.any(np.abs(command)>2)
                or not np.array_equal(decision['actual_first_interval_state'],state)):
            raise ValueError('Selected-plan gap differs from actual held path, next anchor or clocks')
        def cost(x):
            return .04*math.fsum([math.fsum(float(v*v) for v in x[:2]-goal),
                .05*math.fsum(float(v*v) for v in x[2:4]),
                .001*math.fsum(float(v*v) for v in command)])
        actual=cost(state)
        if not math.isclose(actual,decision['actual_first_interval_cost'],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Logged actual first-interval cost does not reconcile with raw state/command')
        actual_costs.append(actual)
        if decision['status']=='all_candidates_failed':
            if (decision['predicted_anchors'] is not None or decision['predicted_first_interval_cost'] is not None
                    or np.any(command)):
                raise ValueError('Failed-search fallback has an invented prediction or nonzero command')
            failures+=1
            rows.append(dict(time_ns=now,prediction_status='missing_due_to_fallback',q_error_rad=None,
                v_error_rad_s=None,actual_interval_cost=actual,predicted_interval_cost=None,cost_gap=None))
            continue
        forecast=np.asarray(decision['predicted_anchors'],dtype=np.float64)
        if decision['status']!='ok' or forecast.shape!=(6,truth.shape[1]) or not np.isfinite(forecast).all():
            raise ValueError('Missing/nonfinite selected-plan predictions cannot yield a measured gap')
        predicted=cost(forecast[0])
        if not math.isclose(predicted,decision['predicted_first_interval_cost'],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Logged predicted first-interval cost does not reconcile with saved first anchor')
        q_error=forecast[0,:2]-state[:2];v_error=forecast[0,2:4]-state[2:4]
        q_squared.extend(float(v*v) for v in q_error);v_squared.extend(float(v*v) for v in v_error)
        gaps.append(actual-predicted)
        rows.append(dict(time_ns=now,prediction_status='measured',q_error_rad=q_error.tolist(),
            v_error_rad_s=v_error.tolist(),actual_interval_cost=actual,predicted_interval_cost=predicted,
            cost_gap=actual-predicted))
    count=len(gaps)
    return dict(n_decisions=40,n_predictions=count,fallback_count=failures,
        q_prediction_rmse_rad=math.sqrt(math.fsum(q_squared)/len(q_squared)) if count else None,
        v_prediction_rmse_rad_s=math.sqrt(math.fsum(v_squared)/len(v_squared)) if count else None,
        mean_real_minus_predicted_cost=math.fsum(gaps)/count if count else None,
        mean_absolute_cost_gap=math.fsum(abs(gap) for gap in gaps)/count if count else None,
        sum_actual_interval_costs=math.fsum(actual_costs),terminal_cost_included=False,decisions=rows,
        interpretation='Own selected plan versus its actual next interval; subsequent replanning is excluded')
