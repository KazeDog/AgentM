"""Train-only numeric corruption, restricted to pinned model resources."""
import json
import math
from pathlib import Path
from .common import file_sha256, stable_hash

PROFILE_PATH=Path(__file__).parent/'configs/regression_injection_train_scale.json'


def regression_profile(prediction, context=None):
    if prediction.get('task_type') not in ('admet_prediction','property_prediction'):
        return None
    payload=json.loads(PROFILE_PATH.read_text())
    profile=payload['profiles'].get(prediction.get('dataset_name'))
    if profile is None:
        return None
    if payload['protocol']!='admet_regression_train_quantiles' or profile['mode']!='regression':
        raise ValueError('invalid_regression_injection_profile')
    for name in ('model','data','reproducibility'):
        if file_sha256(profile[name+'_path'])!=profile[name+'_sha256']:
            raise ValueError('regression_profile_resource_hash_mismatch:'+name)
    selected=((context or {}).get('resource_resolution',{}).get('model',{}) or {}).get('selected_path')
    if selected and Path(selected).resolve()!=Path(profile['model_path']).resolve():
        raise ValueError('regression_profile_model_binding_mismatch')
    return dict(profile,profile_sha256=stable_hash(profile),protocol=payload['protocol'])


def numeric_candidates(prediction, gold, profile):
    # Do not reinterpret a probability or a categorical ID as a regression.
    if prediction.get('probability') is not None or (prediction.get('details') or {}).get('classes'):
        raise ValueError('regression_prediction_schema_mismatch')
    values=(prediction.get('label'),prediction.get('score'),gold)
    if any(isinstance(v,bool) for v in values):
        raise ValueError('regression_values_must_be_finite_numbers')
    try:
        old,score,truth=map(float,values)
    except (ValueError,TypeError) as exc:
        raise ValueError('regression_values_must_be_finite_numbers') from exc
    if not all(math.isfinite(v) for v in (old,score,truth)) or old!=score:
        raise ValueError('regression_prediction_label_score_inconsistent')
    delta=float(profile['minimum_separation'])
    if not math.isfinite(delta) or delta<=0 or not math.isclose(delta,.5*profile['iqr']):
        raise ValueError('invalid_regression_train_scale')
    grid=sorted(set(float(v) for v in profile['candidate_values']))
    if not grid or not all(math.isfinite(v) for v in grid):
        raise ValueError('invalid_regression_candidate_grid')
    return old,[v for v in grid if abs(v-old)>=delta and abs(v-truth)>=delta]
