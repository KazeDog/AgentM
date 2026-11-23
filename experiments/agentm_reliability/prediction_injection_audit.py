"""Fail-closed semantic check for prediction mutations, not report scoring.

This does not repair the injector or reinterpret old experiment results.
"""
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from .prediction_mutations import same_label
from .regression_injection import regression_profile, numeric_candidates


def structured(value):
    if is_dataclass(value):
        return asdict(value)
    return dict(value) if isinstance(value, Mapping) else None


def audit_prediction_injection(hook):
    case = hook.get('case') or {}
    fault = case.get('injection_type')
    if fault not in ('missing_field_or_overclaim', 'plausible_but_wrong_prediction'):
        return {'applicable':False, 'passed':True, 'errors':[], 'scope':'prediction_mutation_contract_only'}
    mutation = hook.get('mutation_record') or {}
    before, after = mutation.get('before') or [], mutation.get('after') or []
    errors = []
    first = structured(before[-1]) if isinstance(before, list) and before else None
    last = structured(after[-1]) if isinstance(after, list) and after else None
    if mutation.get('status')=='not_applicable':
        reason=mutation.get('not_applicable_reason')
        justified=False
        gold=(case.get('gold_truth_reference') or {}).get('value')
        if first is not None and before==after and mutation.get('changed') is False and fault=='plausible_but_wrong_prediction':
            if reason=='no_distinct_wrong_class_available':
                classes=(first.get('details') or {}).get('classes')
                if not classes and first.get('label') in (0,1) and gold in (0,1):
                    classes=[0,1]
                justified=bool(classes and len({str(x) for x in classes})==2
                    and any(same_label(first.get('label'),x) for x in classes)
                    and any(same_label(gold,x) for x in classes)
                    and not same_label(first.get('label'),gold))
            elif reason=='no_train_scaled_regression_candidate_available':
                try:
                    profile=regression_profile(first)
                    justified=profile is not None and not numeric_candidates(first,gold,profile)[1]
                except (ValueError,TypeError,OSError):
                    justified=False
        return {'applicable':False,'passed':False,'not_applicable_valid':justified,
                'errors':[] if justified else ['unjustified_not_applicable'],
                'eligible_for_injection_metrics':False,'count_as_success':False,
                'scope':'applicability_only_not_scientific_success'}
    if first is None:
        errors.append('baseline_prediction_snapshot_not_structured')
    if last is None or 'label' not in last:
        errors.append('mutated_prediction_missing_structured_label')
    if first is not None and last is not None:
        for field in ('task_type', 'dataset_name', 'details'):
            if field in first and field not in last:
                errors.append('unintended_prediction_schema_loss:' + field)
        if fault == 'missing_field_or_overclaim':
            if last.get('label') != first.get('label'):
                errors.append('missing_field_mutation_changed_label')
            removed = (mutation.get('details') or {}).get('removed_field')
            if removed:
                if removed not in first or removed in last or set(first) - set(last) != {removed}:
                    errors.append('missing_field_not_exactly_one_declared_field')
                if any(last.get(k) != v for k,v in first.items() if k != removed):
                    errors.append('unintended_change_to_retained_prediction_fields')
            elif last.get('claim_status') != 'experimentally_confirmed':
                errors.append('missing_overclaim_marker')
            elif any(last.get(k) != v for k,v in first.items() if k != 'claim_status'):
                errors.append('unintended_change_to_retained_prediction_fields')
        else:
            is_yield = first.get('task_type') == 'reaction_yield_prediction'
            try:
                profile=regression_profile(first)
            except (ValueError,OSError) as exc:
                profile=None
                errors.append('regression_profile_invalid:'+str(exc))
            is_regression=profile is not None
            wrong = last.get('score') if is_yield else last.get('label')
            previous = first.get('score') if is_yield else first.get('label')
            if same_label(wrong, previous):
                errors.append('wrong_prediction_label_unchanged')
            gold = (case.get('gold_truth_reference') or {}).get('value')
            if gold is None:
                errors.append('missing_independent_label_reference')
            elif same_label(wrong, gold):
                errors.append('injected_prediction_equals_gold_label')
            if is_regression:
                try:
                    _,allowed=numeric_candidates(first,gold,profile)
                    if last.get('label') not in allowed or last.get('score')!=last.get('label'):
                        errors.append('invalid_train_scaled_regression_mutation')
                    details=mutation.get('details') or {}
                    if details.get('scale_profile_sha256')!=profile['profile_sha256'] or details.get('mutation_protocol')!=profile['protocol']:
                        errors.append('regression_mutation_profile_receipt_mismatch')
                except (ValueError,TypeError) as exc:
                    errors.append('invalid_regression_baseline:'+str(exc))
            classes = (first.get('details') or {}).get('classes')
            if not is_yield and classes and not any(same_label(wrong, x) for x in classes):
                errors.append('injected_prediction_outside_class_vocabulary')
            for field in ('score','probability'):
                if not ((is_yield or is_regression) and field == 'score') and first.get(field) != last.get(field):
                    errors.append('unintended_confidence_change:' + field)
            if not is_yield and 'interaction_type_class' in (last.get('details') or {}):
                if not same_label(last['details']['interaction_type_class'],wrong):
                    errors.append('ddi_class_description_mismatch')
            if is_yield:
                if not isinstance(wrong,(int,float)) or not 0 <= wrong <= 100:
                    errors.append('yield_outside_valid_range')
                elif last.get('label') != ('high_yield' if wrong >= 70 else 'moderate_yield' if wrong >= 30 else 'low_yield'):
                    errors.append('yield_label_numeric_mismatch')
    return {'applicable':True, 'passed':not errors, 'errors':sorted(set(errors)),
            'scope':'prediction_mutation_contract_only_not_scientific_performance'}
