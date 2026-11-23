"""Resolve the nominal user decision and the exact, measured whole-chain allocation."""
import copy
import hashlib
import json
from pathlib import Path

from training_data_origin import read_bound
from runtime_limited_plan import resolve_effective_budget


def load_total_budget_contract(root, plan):
    root = Path(root).resolve()
    binding = plan.get('global_training_budget')
    if binding is None:
        if plan.get('context_scaling', {}).get('budget_included_in_pretrain_1t') is True:
            raise ValueError('an included context budget needs its bound nominal and effective allocation')
        return None
    expected = {'nominal_file', 'nominal_sha256', 'effective_file', 'effective_sha256',
                'execution_plan_file', 'execution_plan_sha256'}
    if not isinstance(binding, dict) or set(binding) != expected:
        raise ValueError('whole-chain budget binding must contain the exact nominal/effective/derivation roster')
    documents, inputs = {}, {}
    for key in ('nominal', 'effective', 'execution_plan'):
        name = binding[key+'_file']
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('whole-chain budget assets must be repository-relative')
        raw, sha = read_bound(root/name, binding[key+'_sha256'])
        documents[key] = json.loads(raw)
        inputs[name] = sha
    if binding['nominal_file'] != 'data/health/user-total-pretrain-and-length-budget.json':
        raise ValueError('whole-chain budget must bind the actual active user decision')
    nominal, actual, document = (documents[k] for k in ('nominal', 'effective', 'execution_plan'))
    if document.get('nominal_budget_sha256') != binding['nominal_sha256']:
        raise ValueError('measured allocation belongs to another nominal user budget')
    effective, forecast = resolve_effective_budget(nominal, document['proposal'])
    expected_actual = copy.deepcopy(effective)
    expected_actual.update(applied_to_production_configuration=True,
        allocation_role='PREPARED_EXECUTION_REQUIRES_FINAL_PRODUCTION_GATES',
        runtime_budget_derivation={'nominal_budget_file': binding['nominal_file'],
            'nominal_budget_sha256': binding['nominal_sha256'],
            'execution_plan_file': binding['execution_plan_file'],
            'execution_plan_sha256': binding['execution_plan_sha256']})
    if actual != expected_actual:
        raise ValueError('effective training budget differs from the independently reproduced runtime allocation')
    for name, expected_sha in document['source_inputs'].items():
        read_bound(root/name, expected_sha)
        inputs[name] = expected_sha
    for name in ('total_budget_contract.py', 'runtime_limited_plan.py', 'global_training_budget.py'):
        path = Path(__file__).with_name(name)
        inputs[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'budget': effective, 'nominal': nominal, 'forecast': forecast,
            'measurements': document['proposal']['measurements'], 'inputs': inputs}
