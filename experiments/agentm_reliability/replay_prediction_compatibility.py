"""Replay saved predictions; re-execute local reaction classifier to check its new schema."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from unittest.mock import patch
from .common import (PROJECT_ROOT, RELIABILITY_ROOT, file_sha256, load_jsonl, load_config,
                     resolve_path, source_tree_hash, stable_hash, write_json)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root,output=args.root.resolve(),args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    os.environ.update(AGENTM_ENABLE_NETWORK_LOOKUP='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
    from agentm.schema import PredictionResult
    from agentm.runtime.reaction import ReactionRuntime
    from agentm.runtime.tools import AgentAction,ToolResult
    from .injections import ControlledInjectionHook
    from .prediction_injection_audit import audit_prediction_injection
    from .runtime_guards import api_budget_guard
    from .regression_injection import PROFILE_PATH
    output.mkdir()
    cases=load_jsonl(root/'preparation/stress_combined_candidates.jsonl')
    targets=[c for c in cases if c['injection_type'] in ('missing_field_or_overclaim','plausible_but_wrong_prediction')]
    if len(targets)!=32:
        raise ValueError('Unexpected candidate scope')
    config=load_config(RELIABILITY_ROOT/'configs/stress_single_build.yaml')
    source=source_tree_hash()
    profiles=json.loads(PROFILE_PATH.read_text())
    hashes={str(PROFILE_PATH):file_sha256(PROFILE_PATH),str(root/'preparation/stress_combined_candidates.jsonl'):file_sha256(root/'preparation/stress_combined_candidates.jsonl')}
    rows=[]
    calls=0
    with patch('socket.socket.connect',side_effect=AssertionError('Compatibility replay forbids network')), \
         patch('socket.create_connection',side_effect=AssertionError('Compatibility replay forbids network')), \
         api_budget_guard(output,max_calls=0,max_total_tokens=0):
        for case in targets:
            path=root/'candidate_freeze_preflight/prediction_probes'/f"{case['case_id']}.json"
            hashes[str(path)]=file_sha256(path)
            previous=json.loads(path.read_text())
            base=case['base_case']
            prediction=deepcopy(previous['baseline_prediction'])
            model=Path(previous['model_path'])
            if file_sha256(model)!=previous['model_sha256']:
                raise ValueError('Previous probe model resource changed')
            row={'case_id':case['case_id'],'previous_status':previous['status'],
                 'model_path':str(model),'model_sha256':file_sha256(model),'formal_statistics_eligible':False}
            if base['task_type']=='reaction_classification':
                runtime=ReactionRuntime(output_dir=output/'classifier'/case['case_id'],
                    data_root=config['paths']['reaction_data_root'],
                    model_dir=resolve_path(config['paths']['reaction_model_dir']))
                c={'request':base['request'],'task_type':base['task_type'],'dataset_name':base['dataset_name']}
                result=runtime.classify_reaction_type({},c)
                calls+=1
                if result.status!='completed' or result.observation['model_path']!=str(model):
                    raise ValueError('Unexpected classifier/model failure')
                prediction=result.observation['prediction']
                stripped=deepcopy(prediction)
                stripped['details'].pop('classes',None)
                stripped['details'].pop('class_vocabulary_source',None)
                row['baseline_prediction_unchanged_except_vocab']=stripped==previous['baseline_prediction']
                if not row['baseline_prediction_unchanged_except_vocab']:
                    raise ValueError('Classifier baseline changed, stopping without retries')
                row['class_vocabulary_count']=len(prediction['details']['classes'])
            variants=[]
            for input_kind in ('mapping','dataclass'):
                value=deepcopy(prediction) if input_kind=='mapping' else PredictionResult(**deepcopy(prediction))
                context={'task_type':base['task_type'],'dataset_name':base['dataset_name'],'predictions':[value],
                         'metadata':{},'resource_resolution':{'model':{'selected_path':str(model)}}}
                if base['task_type'].startswith('reaction_'):
                    context['reaction_predictions']=[value]
                tool='classify_reaction_type' if base['task_type']=='reaction_classification' else (
                    'predict_buchwald_hartwig_yield' if base['task_type']=='reaction_yield_prediction' else (
                    'predict_ddi' if base['task_type']=='ddi_prediction' else 'predict_single'))
                hook=ControlledInjectionHook(case)
                hook.after_tool(action=AgentAction('GaugeAgent',tool),result=ToolResult(tool,'completed',{'prediction':prediction}),context=context)
                exported=json.loads(json.dumps(hook.export()))
                checked=audit_prediction_injection(exported)
                status='not_applicable' if checked.get('not_applicable_valid') else ('applicable_pass' if checked['passed'] and hook.applied else 'failed')
                variants.append({'input_kind':input_kind,'status':status,'audit':checked,'hook':exported})
            if len({v['status'] for v in variants})!=1:
                raise ValueError('Dataclass and mapping disagree')
            row.update(status=variants[0]['status'],variants=variants,
                       eligible_for_injection_metrics=variants[0]['status']=='applicable_pass',
                       counts_as_scientific_success=False,scientific_quality_not_evaluated=True)
            write_json(output/'cases'/f"{case['case_id']}.json",row,overwrite=False)
            rows.append(row)
            print(json.dumps({'case_id':row['case_id'],'status':row['status']}),flush=True)
            if row['status']=='failed':
                break
    unchanged=all(file_sha256(p)==h for p,h in hashes.items())
    summary={'candidate_count':32,'checked_cases':len(rows),'component_variants':sum(len(r['variants']) for r in rows),
             'applicable_cases_passed':sum(r['status']=='applicable_pass' for r in rows),
             'not_applicable_cases':[r['case_id'] for r in rows if r['status']=='not_applicable'],
             'failed_cases':[r['case_id'] for r in rows if r['status']=='failed'],
             'local_classifier_calls':calls,'external_llm_calls':0,'external_llm_tokens':0,
             'inputs_and_prior_artifacts_unchanged':unchanged,'input_sha256':hashes,
             'source_tree_hash':source,'source_unchanged_during_replay':source_tree_hash()==source,
             'regression_policy_sha256':file_sha256(PROFILE_PATH),
             'formal_statistics_eligible':False,'not_applicable_not_counted_as_success':True,
             'scope':'component_replay_not_full_controller_or_formal_experiment',
             'case_statuses':[{k:v for k,v in r.items() if k!='variants'} for r in rows]}
    summary['passed']=len(rows)==32 and unchanged and not summary['failed_cases'] and summary['source_unchanged_during_replay']
    write_json(output/'summary.json',summary,overwrite=False)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('case_statuses','input_sha256')},ensure_ascii=False))


if __name__=='__main__':
    main()
