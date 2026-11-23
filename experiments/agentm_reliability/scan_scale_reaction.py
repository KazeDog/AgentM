"""Local-only reaction retrieval preflight, zero API budget, no prediction."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
from .common import load_config, load_jsonl, RELIABILITY_ROOT, resolve_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    with patch('socket.socket.connect', side_effect=RuntimeError('Reaction preflight forbids network')):
        from agentm.runtime.reaction import ReactionRuntime
        from .runtime_guards import api_budget_guard, run_real_rag_preflight
        config = load_config(RELIABILITY_ROOT / 'configs/stress_single_build.yaml')
        cases = [r for r in load_jsonl(args.preparation / 'stress_additional_candidates.jsonl') if r['task_type'].startswith('reaction_')]
        rows = []
        with api_budget_guard(args.output, max_calls=0, max_total_tokens=0):
            runtime = ReactionRuntime(output_dir=args.output / 'runtime', data_root=config['paths']['reaction_data_root'],
                                      model_dir=resolve_path(config['paths']['reaction_model_dir']),
                                      index_dir=resolve_path(config['paths']['reaction_index_dir']))
            for row in cases:
                record = {'case_id': row['case_id']}
                try:
                    result = run_real_rag_preflight(None, case=row['base_case'], runtime=config['runtime'],
                                                   output_path=args.output / f"{row['case_id']}.json", label=row['case_id'],
                                                   reaction_runtime=runtime)
                    record['passed'] = result['passed']
                except Exception as exc:
                    record.update(passed=False, error=f'{type(exc).__name__}: {exc}')
                rows.append(record)
                print(json.dumps(record), flush=True)
                if not record['passed']:
                    break
        summary = {'cases': len(cases), 'checked': len(rows), 'passed': sum(r['passed'] for r in rows), 'records': rows,
                   'external_llm_calls': 0, 'external_llm_tokens': 0, 'scope': 'local_retrieval_not_prediction_model_execution'}
        write_json(args.output / 'summary.json', summary, overwrite=False)


if __name__ == '__main__':
    main()
