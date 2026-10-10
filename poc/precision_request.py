"""Prospective requests for the fixed, rejection-only sensitivity comparison."""
import math
from pathlib import Path
import re

from .io import utc_now
from .subset_precision import next_seed_block


def build_request(parent, scope, evidence, resources, *, request_id, context_relative):
    allowed = parent.get('permitted_next_actions', [])
    if (parent.get('gate_id') != 'GR' or parent.get('outcome') != 'INCONCLUSIVE'
            or parent.get('validity_status') != 'valid' or parent.get('phase') != 'confirmatory'
            or not any(a.get('milestone_id') == 'MX' and a.get('branch') == 'precision' for a in allowed)):
        raise ValueError('A valid unresolved precision transition is required')
    if (scope.get('negative_only') is not True or scope.get('statistical_family_size') != 280
            or (scope.get('actuator'), scope.get('condition'), scope.get('loss_mode'), scope.get('competitor'))
            != ('direct', 'A', 'shared', 'node_generic')):
        raise ValueError('Precision request cannot change the frozen comparison')
    if not re.fullmatch(r'MX-precision-\d{4}', request_id):
        raise ValueError('Invalid precision request identifier')
    if Path(context_relative).is_absolute() or '..' in Path(context_relative).parts:
        raise ValueError('Precision context must be project-relative')
    if (evidence.get('role') != 'manifest' or not re.fullmatch('[0-9a-f]{64}', evidence.get('sha256', ''))
            or Path(evidence.get('path', '')).is_absolute() or '..' in Path(evidence.get('path', '')).parts):
        raise ValueError('Precision requires an exact parent manifest reference')
    for key in ('measured_new_training_attempts', 'measured_new_training_wall_seconds', 'measured_peak_rss_bytes'):
        value = resources.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('A measured resource projection is required: ' + key)
    old = scope['seed_ids']
    seeds = next_seed_block(old)
    if seeds[:len(old)] != old:
        raise ValueError('Precision must preserve every existing seed in order')
    new = seeds[len(old):]
    look = parent.get('look_index')
    if type(look) is not int or look < 1:
        raise ValueError('Precision must retain a valid statistical look ledger')
    look += 1
    command = f'python -m poc.subset_precision config/revisions/{request_id}.json --root {context_relative} --resume'
    contract = {
        '1': 'S12/MX and S13.4 precision: discriminate C02 under unchanged PTCO-1.0; no subset success.',
        '2': f"Exact valid unresolved look{look-1} GR decision, {len(old)} paired seeds, no supported mandatory rejection.",
        '3': f'Copy the identical scientific source and immutable binaries; preserve prior receipts and append seeds{new[0]}–{new[-1]}.',
        '4': f'Commit this request before new outcomes; retain all {len(old)} old clusters and every prior statistical look. Data, models, losses, nominal selection and thresholds stay fixed.',
        '5': command,
        '6': 'MR raw/summary artifacts, initial context, exact parent hashes and an appended look ledger entry.',
        '7': 'Full paired coverage, raw-matrix reaggregation, qualified nominal fits, same-field solver checks; retain every bad seed and failed attempt.',
        '8': 'Original conservative t/bootstrap, Fieller, numerical-envelope and influence criteria; no change to the 50% floor or K280.',
        '9': 'Valid precise rejection -> MF; subset support -> unfinished M30; unresolved -> a newly justified named request; exact external prerequisite -> BLOCKED.',
        '10': 'Measured prior-block training projects the next block; CPU only, serial model/seed jobs, no artificial resource caps.',
        '11': 'Resume only matching jobs; preserve original results and all attempts. Diagnose validity defects instead of altering scientific floors.',
        '12': f'Complete {len(new)} new paired clusters and look{look}; preserve untested control claims and the original alpha ledger.'}
    factor = len(new) / max(5, len(old)//2)
    return {'schema_version': 1, 'protocol_version': 'PTCO-1.0', 'request_id': request_id,
        'branch': 'precision', 'parent_gate': 'GR', 'return_milestone': 'MR', 'claim_ids': ['C02'],
        'parent_decision_id': parent['decision_id'], 'prior_manifest': evidence['path'],
        'prior_manifest_sha256': evidence['sha256'], 'input_evidence': [evidence],
        'unresolved_question': 'Which side of the frozen paired sensitivity improvement boundary is supported?',
        'live_explanations': ['the correction clears the required improvement floor',
                              'its improvement is materially below that floor'],
        'new_information': f'{len(new)} independent paired training-seed clusters under the unchanged design.',
        'action_parameters': {'seed_ids': seeds, 'new_seed_ids': new, 'next_look': look,
                              'alpha': .05/2**look},
        'expected_artifacts': ['INITIAL_CONTEXT.json', 'config/precision_parent.json',
            'appended training/E01 receipts', 'appended statistical look', 'precision_decision.json'],
        'progress_condition': 'New independent clusters reduce or diagnose the unresolved contrast, ratio precision or seed influence.',
        'resource_projection': {**resources, 'projected_new_training_wall_seconds': factor*resources['measured_new_training_wall_seconds'],
            'new_selected_training_fits': 2*len(new), 'new_E01_jobs': 3*len(new),
            'additional_audit_time': 'Project from the measured audit jobs; reuse compatible prior references.',
            'sample_rule': 'Prespecified seed doubling; further continuation requires a new prospective request.',
            'owner_imposed_caps': None},
        'authorization_reference': 'config/authorization.json', 'statistical_family_size': 280,
        'contract': contract, 'working_context': f'study/precision-look-{look}',
        'recorded_utc': utc_now(), 'status': 'prospective'}
