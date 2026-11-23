"""Bounded zero-LLM retrieval probe of the 32 new molecular stress bases."""
import argparse
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch

from .common import load_jsonl, write_json
from .preflight_scale_sources import INDEX


def scan(preparation, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    os.environ.update(AGENTM_ENABLE_NETWORK_LOOKUP='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      AGENTM_ES_EMBEDDING_DEVICE='cpu', TOKENIZERS_PARALLELISM='false')
    import torch
    torch.set_num_threads(2)
    from .runtime_guards import api_budget_guard
    from .scan_direct_evidence_candidates import _resolve_context
    from .evidence_provenance import audit_evidence_provenance
    from .validation import is_verified_elasticsearch_hybrid_attempt
    from agentm.runtime.tools.scientific import ScientificToolRuntime
    cases = [r for r in load_jsonl(preparation / 'stress_additional_candidates.jsonl') if r['task_type'] in ('admet', 'ddi')]
    rows = []
    with api_budget_guard(output, max_calls=0, max_total_tokens=0):
        runtime = ScientificToolRuntime(output_dir=output / 'resolver', feature_set='fingerprint_only',
                                        require_real_model=False, require_real_rag=True, use_external_evidence=False,
                                        use_smiles_encoder=False, use_mllm_visual_semantic=False,
                                        retrieval_backend='es', es_url='http://127.0.0.1:9200', es_index_name=INDEX,
                                        use_llm_reasoning=False, require_llm_reasoning=False, llm_profile=None)
        runtime.hybrid_retriever.embedding_device = 'cpu'
        runtime.hybrid_retriever.max_retries = 0
        runtime.hybrid_retriever.request_timeout_seconds = 15
        for case in cases:
            context, status = _resolve_context(runtime, case)
            row = {'case_id': case['case_id'], 'fault': case['injection_type'], 'resolution_status': status,
                   'resolved_names': runtime._molecule_names(context)}
            if status == 'completed':
                result = runtime.hybrid_retriever.retrieve(runtime._rag_query(context),
                                                          metadata=runtime._retrieval_metadata(context), backend='es', top_k=3)
                row.update(retrieval=result.to_dict(), provenance=audit_evidence_provenance({'evidence': result.evidence_items}),
                           retrieval_execution_valid=is_verified_elasticsearch_hybrid_attempt(result.to_dict()))
            else:
                row['retrieval_execution_valid'] = False
            write_json(output / f"{case['case_id']}.json", row, overwrite=False)
            rows.append(row)
            print(json.dumps({'case_id': case['case_id'], 'valid': row['retrieval_execution_valid'],
                              'direct': row.get('provenance', {}).get('direct_evidence_count', 0)}), flush=True)
            if not row['retrieval_execution_valid']:
                break
    summary = {'external_llm_calls': 0, 'external_llm_tokens': 0, 'execution_mode': 'local_CPU_embedding_plus_frozen_ES_read_only',
               'expected_cases': len(cases), 'checked_cases': len(rows),
               'valid_retrieval_cases': sum(r['retrieval_execution_valid'] for r in rows),
               'direct_support_cases': sum(r.get('provenance', {}).get('direct_evidence_count', 0) > 0 for r in rows),
               'stopped_on_pipeline_error': len(rows) < len(cases) or any(not r['retrieval_execution_valid'] for r in rows),
               'status': 'NO_LLM_AUTHORIZATION',
               'scope': 'retrieval_preflight_not_full_model_or_agent_execution',
               'records': [{k: v for k, v in r.items() if k != 'retrieval'} for r in rows]}
    write_json(output / 'summary.json', summary, overwrite=False)
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = socket.socket.connect
    def connect(sock, address):
        if not isinstance(address, tuple) or address[:2] != ('127.0.0.1', 9200):
            raise RuntimeError('External network forbidden; no paid API allowed')
        return original(sock, address)
    with patch('socket.socket.connect', connect):
        scan(args.preparation.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
