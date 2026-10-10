"""Complete matrix action errors and empirical numerical uncertainty intervals."""
from __future__ import annotations

import numpy as np


def validate_row(row):
    from pathlib import Path
    from .contracts import _validate_schema
    from .io import canonical_json,read_json
    _validate_schema(row,read_json(Path(__file__).resolve().parent.parent / 'schemas/metrics.schema.json'))
    canonical_json(row)
    if row['sample_count'] < 0 or (row['seed'] is not None and row['seed'] < 0):
        raise ValueError('Metric sample/seed identifiers must be nonnegative')
    if row['status'] == 'ok' and row['value'] is None:
        raise ValueError('An observed ok metric cannot have a missing value')


def jacobian_metrics(model_jac,reference_jac,envelope):
    model,reference = np.asarray(model_jac,dtype=np.float64),np.asarray(reference_jac,dtype=np.float64)
    if model.shape != (2,2) or reference.shape != (2,2):
        raise ValueError('Expected physical two-command/two-state-component Jacobian matrices')
    eta,zeta = float(envelope['reference']),float(envelope['model'])
    if not np.isfinite([eta,zeta]).all() or min(eta,zeta) < 0:
        raise ValueError('Finite nonnegative empirical numerical envelopes are required')
    empty = {'error':None,'gain':None,'error_bounds':None,'gain_bounds':None,'alignment':None,
             'singular_values':None,'gain_certain_inside':False,'gain_possible_inside':False}
    n = np.linalg.norm(reference)
    if not np.isfinite(reference).all() or not np.isfinite(n) or n <= max(100*eta,1e-14):
        return {'status':'unresolved_reference',**empty}
    if not np.isfinite(model).all():
        return {'status':'model_failure',**empty}
    a,d = np.linalg.norm(model),np.linalg.norm(model-reference)
    if not np.isfinite([a,d]).all():
        return {'status':'model_failure',**empty}
    errors = [max(0.,d-eta-zeta)/(n+eta),(d+eta+zeta)/(n-eta)]
    gains = [max(0.,a-zeta)/(n+eta),(a+zeta)/(n-eta)]
    return {'status':'ok','error':float(d/n),'gain':float(a/n),'error_bounds':[float(x) for x in errors],
            'gain_bounds':[float(x) for x in gains],
            'alignment':float(np.sum(model*reference)/(a*n)) if a>0 else None,
            'singular_values':np.linalg.svd(model,compute_uv=False).tolist(),
            'gain_certain_inside':bool(gains[0] >= .5 and gains[1] <= 1.5),
            'gain_possible_inside':bool(gains[0] <= 1.5 and gains[1] >= .5)}


def fit_slope(h_s,norms,norm_bounds):
    h,norms,bounds = np.asarray(h_s,dtype=float),np.asarray(norms,dtype=float),np.asarray(norm_bounds,dtype=float)
    if h.shape != (3,) or norms.shape != (3,) or bounds.shape != (3,2) or not np.isfinite(h).all() or np.any(h<=0) or len(np.unique(h)) != 3:
        raise ValueError('Slope audit requires three distinct positive physical horizons')
    empty = {'slope':None,'intercept':None,'residual_sum_squares':None,'slope_bounds':None}
    if not np.isfinite(norms).all() or not np.isfinite(bounds).all():
        return {'status':'model_failure',**empty}
    if np.any(norms <= 0):
        return {'status':'zero_model_gain',**empty}
    if np.any(bounds[:,0] <= 0):
        return {'status':'unresolved_gain',**empty}
    if np.any(bounds[:,0] > norms) or np.any(bounds[:,1] < norms):
        raise ValueError('Numerical norm bounds must contain the measured norm')
    x,y = np.log(h),np.log(norms)
    centered = x-x.mean()
    coefficients = centered/np.sum(centered**2)
    slope = float(coefficients@y)
    intercept = float(y.mean()-slope*x.mean())
    lower = float(np.sum(coefficients*np.where(coefficients>=0,np.log(bounds[:,0]),np.log(bounds[:,1]))))
    upper = float(np.sum(coefficients*np.where(coefficients>=0,np.log(bounds[:,1]),np.log(bounds[:,0]))))
    return {'status':'ok','slope':slope,'intercept':intercept,
            'residual_sum_squares':float(np.sum((y-(intercept+slope*x))**2)),
            'slope_bounds':[lower,upper],'norms':norms.tolist(),'h_s':h.tolist()}
