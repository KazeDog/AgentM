"""Typed, outcome-independent mutation rules for new controlled diagnostics."""
from copy import deepcopy
from collections.abc import Mapping
import math
import random

from agentm.runtime.compat import to_jsonable
from agentm.runtime.prediction_semantics import annotate_multiclass_ddi_prediction, is_multiclass_ddi_prediction
from .regression_injection import regression_profile, numeric_candidates


class InjectionNotApplicable(ValueError):
    """Valid precondition boundary; not a successful mutation or a retry signal."""


class InjectedPrediction(dict):
    """Keep mapping and attribute-based runtime consumers compatible."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def to_dict(self):
        return to_jsonable(deepcopy(dict(self)))


def normalize_prediction(value):
    if not isinstance(value, Mapping) and not hasattr(value, 'to_dict'):
        from dataclasses import is_dataclass
        if not is_dataclass(value):
            raise ValueError('prediction_must_be_structured_not_repr')
    payload = to_jsonable(value)
    if not isinstance(payload, dict) or 'label' not in payload:
        raise ValueError('prediction_missing_structured_label')
    return deepcopy(payload)


def same_label(a, b):
    # Numeric class IDs are sometimes supplied as strings by dataset loaders.
    return str(a) == str(b) or (isinstance(a, (int, float)) and isinstance(b, (int, float)) and a == b)


def mutate_prediction(value, fault, *, seed, truth_reference=None, context=None):
    item = normalize_prediction(value)
    original = deepcopy(item)
    if fault == 'missing_field_or_overclaim':
        removed = next((k for k in ('model_type', 'score', 'probability', 'confidence', 'uncertainty') if k in item), None)
        if removed is None:
            item['claim_status'] = 'experimentally_confirmed'
        else:
            item.pop(removed)
        return InjectedPrediction(item), {'removed_field':removed, 'overclaim_inserted':removed is None,
                                          'mutation_protocol':'structured_prediction'}
    if fault != 'plausible_but_wrong_prediction':
        raise ValueError('unsupported_prediction_mutation')
    if not isinstance(truth_reference, Mapping) or truth_reference.get('value') is None:
        raise ValueError('wrong_prediction_requires_independent_gold')
    gold = truth_reference['value']
    rng = random.Random(seed)
    details = item.get('details') or {}
    profile = regression_profile(item, context)
    if profile is not None:
        old,alternatives=numeric_candidates(item,gold,profile)
        if not alternatives:
            raise InjectionNotApplicable('no_train_scaled_regression_candidate_available')
        wrong=rng.choice(alternatives)
        item['label']=item['score']=wrong
        if 'label' in details:
            item['details']['label']=wrong
        return InjectedPrediction(item), {'original_prediction':old,'injected_prediction':wrong,
            'mutation_kind':'admet_regression','mutation_protocol':'admet_regression_train_quantiles',
            'scale_profile_sha256':profile['profile_sha256'],'minimum_separation':profile['minimum_separation'],
            'unit':profile['unit'],'gold_used_only_by_injector':True,'probability_preserved':True}
    if item.get('task_type') == 'reaction_yield_prediction':
        old = float(details.get('predicted_yield', item['score']))
        truth = float(gold)
        if not all(math.isfinite(x) and 0 <= x <= 100 for x in (old, truth)):
            raise ValueError('yield_reference_outside_0_100')
        alternatives = [x for x in (0., 25., 50., 75., 100.) if abs(x-truth) >= 10 and abs(x-old) >= 10]
        wrong = rng.choice(alternatives)
        item['score'] = wrong
        item['label'] = 'high_yield' if wrong >= 70 else 'moderate_yield' if wrong >= 30 else 'low_yield'
        item.setdefault('details', {})['predicted_yield'] = wrong
        kind = 'yield_regression'
        original_value = old
    else:
        classes = details.get('classes')
        if not isinstance(classes, (list, tuple)) or not classes:
            if isinstance(item['label'], (bool, int, float)) and item['label'] in (0,1) and gold in (0,1) and not is_multiclass_ddi_prediction(item):
                classes = [0,1]
            else:
                raise ValueError('missing_valid_class_vocabulary')
        if not any(same_label(item['label'], x) for x in classes):
            raise ValueError('baseline_label_outside_class_vocabulary')
        if not any(same_label(gold, x) for x in classes):
            raise ValueError('gold_label_outside_class_vocabulary')
        alternatives = sorted({str(x):x for x in classes if not same_label(x,gold) and not same_label(x,item['label'])}.values(), key=lambda x:(type(x).__name__,str(x)))
        if not alternatives:
            # In particular, a binary baseline already different from gold
            # cannot be changed again while remaining wrong. Never select a
            # new case based on this outcome or pretend an unchanged value is
            # a successfully applied mutation.
            if len({str(x) for x in classes})==2 and not same_label(item['label'],gold):
                raise InjectionNotApplicable('no_distinct_wrong_class_available')
            raise ValueError('no_distinct_wrong_class_available')
        wrong = rng.choice(alternatives)
        original_value = item['label']
        item['label'] = wrong
        if 'label' in details:
            item['details']['label'] = wrong
        if is_multiclass_ddi_prediction(item):
            names = []
            for molecule in (context or {}).get('molecules', []) or []:
                molecule = to_jsonable(molecule)
                if isinstance(molecule, Mapping):
                    names.append(str(molecule.get('name') or molecule.get('canonical_smiles') or molecule.get('original_input') or ''))
            annotate_multiclass_ddi_prediction(item, drug_names=names or None,
                                               interaction_map_path=details.get('interaction_type_mapping_source'))
        # Preserve scalar confidence. The fault is a plausible high-confidence
        # wrong label, not a second corruption of probabilities or score.
        kind = 'classification'
    return InjectedPrediction(item), {'original_prediction':original_value, 'injected_prediction':wrong,
                                      'mutation_kind':kind, 'mutation_protocol':'structured_prediction',
                                      'gold_used_only_by_injector':True,
                                      'probability_preserved':item.get('probability') == original.get('probability')}
