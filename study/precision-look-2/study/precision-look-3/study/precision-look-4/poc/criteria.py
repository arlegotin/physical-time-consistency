"""Finite, preregistered PTCO-1.0 atomic criteria; independent of result files."""

from itertools import product

from poc.contracts import CriterionRecord
from poc.io import canonical_json

COMPETITORS = ('mlp_dt', 'mlp_h', 'gru_dt', 'lstm_dt', 'attention_dt', 'ssm_dt', 'node_generic')
ACTUATORS = ('direct', 'lagged')
STRATA = ('cold', 'warm')
HORIZONS = (5_000_000, 10_000_000)


def expand_criteria() -> list[CriterionRecord]:
    records = []

    def add(family, dims, threshold, formula, precision=None, strict=False, statistical=True):
        if statistical or family in ('K01.identity', 'K03.identity'):
            fixed = family in ('K02.actuator', 'K03.jacobian', 'K03.attribution', 'K01.identity', 'K03.identity')
            dims = {**dims, 'training_condition': 'F' if fixed else 'A',
                    'loss_mode': 'one_step' if fixed else 'shared', 'method': 'power_correct'}
        identifier = family + ''.join(f'/{key}={value}' for key, value in dims.items())
        return_milestone = ('M70' if family in ('K03.attribution', 'K04.shift', 'K04.nominal', 'K04.task')
                            else 'M80' if family.startswith('K05.') else 'M60')
        records.append({
            'criterion_id': identifier, 'claim_id': 'C' + family[1:3],
            'metric': formula, 'direction': '>' if strict else '>=',
            'dimensions': dims, 'threshold': threshold, 'required_precision': precision,
            'evidence_schema': 'metrics.schema.json' if statistical else 'validity.json',
            'validity_dependencies': ['K01.oracle', 'K01.clocks', 'K01.data', 'K01.mutations'],
            'statistical': statistical,
            'allowed_next_action': {'milestone_id': 'MX', 'branch': 'precision' if statistical else 'repair',
                                    'return_milestone': return_milestone},
        })

    for actor, stratum, horizon, competitor in product(ACTUATORS, STRATA, HORIZONS, COMPETITORS):
        for contrast in ('relative', 'strict_improvement'):
            add('K02.relative', {'actuator': actor, 'stratum': stratum, 'h_ns': horizon,
                                'competitor': competitor, 'contrast': contrast},
                .5 if contrast == 'relative' else 0.,
                '(1-r)*baseline-method' if contrast == 'relative' else 'baseline-method',
                .20, strict=contrast == 'strict_improvement')
    for actor, stratum, horizon, component, statistic in product(ACTUATORS, STRATA, HORIZONS, ('q', 'v'), ('median', 'p90')):
        limit = .5 if statistic == 'median' else 1.
        add('K02.absolute', {'actuator': actor, 'stratum': stratum, 'h_ns': horizon,
                            'component': component, 'statistic': statistic}, limit, 'limit-error', limit/2)
    for actor, stratum, component, statistic in product(ACTUATORS, STRATA, ('q', 'v'), ('median', 'p90')):
        limit = .2 if statistic == 'median' else .4
        add('K02.order', {'actuator': actor, 'stratum': stratum, 'component': component,
                         'statistic': statistic}, limit, 'limit-absolute_slope_error', limit/2)
    for actor, stratum, horizon, component in product(ACTUATORS, STRATA, HORIZONS, ('q', 'v')):
        add('K02.gain', {'actuator': actor, 'stratum': stratum, 'h_ns': horizon, 'component': component},
            .9, 'fraction_in_[0.5,1.5]-limit', .9/2)
    for stratum, horizon in product(STRATA, HORIZONS):
        add('K02.actuator', {'actuator': 'lagged', 'stratum': stratum, 'h_ns': horizon},
            .25, '0.75*mismatched-correct', .20)
    for actor, stratum, horizon in product(ACTUATORS, STRATA, HORIZONS):
        add('K03.jacobian', {'actuator': actor, 'stratum': stratum, 'h_ns': horizon},
            0., 'low-correct', .20, strict=True)
    for actor, contrast in product(ACTUATORS, ('A', 'B', 'A-0.5B')):
        add('K03.attribution', {'actuator': actor, 'contrast': contrast}, 0., contrast,
            strict=contrast in ('A', 'B'))
    for family, floor, formula in (
        ('K04.shift', .2, '0.8*baseline-method'),
        ('K04.nominal', 1.05, '1.05*baseline-method'),
    ):
        for actor, competitor in product(ACTUATORS, COMPETITORS):
            add(family, {'actuator': actor, 'competitor': competitor}, floor, formula, .10)
    for actor in ACTUATORS:
        add('K04.task', {'actuator': actor}, .2, '0.8*zero_command-method', .10)
    for actor, rate, competitor in product(ACTUATORS, (20_000_000, 10_000_000), COMPETITORS):
        add('K05.actuator_rate', {'actuator': actor, 'h_act_ns': rate, 'competitor': competitor},
            .1, '0.9*baseline-method', .10)
    for family in ('K05.noise', 'K05.disturbance'):
        for actor, competitor in product(ACTUATORS, COMPETITORS):
            add(family, {'actuator': actor, 'competitor': competitor}, .1, '0.9*baseline-method', .10)

    for family in ('K01.identity', 'K03.identity'):
        for actor in ACTUATORS:
            add(family, {'actuator': actor}, 1e-12, 'fixed-h0-output-state-command-weight-gradient-identity', statistical=False)
    for family in ('K01.oracle', 'K01.clocks', 'K01.mutations', 'K01.data', 'K04.reliability',
                   'K05.convergence', 'K06.replay', 'K06.retrain', 'K06.coverage', 'K06.provenance'):
        add(family, {}, 0., 'all-required-checks-pass', statistical=False)
    return records


def validate_registry(records: list[CriterionRecord]) -> dict:
    expected = expand_criteria()
    if canonical_json(records) != canonical_json(expected):
        raise ValueError('Atomic registry differs from the preregistered PTCO-1.0 expansion')
    n = sum(c['statistical'] for c in records)
    if n != 280 or len({c['criterion_id'] for c in records}) != len(records):
        raise ValueError('Atomic registry count/identity mismatch')
    return {'statistical_family_size': n, 'total_criteria': len(records)}
