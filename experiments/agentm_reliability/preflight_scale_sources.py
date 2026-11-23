"""Read-only source/gold/index audit; never constructs a controller/API client."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import socket
from unittest.mock import patch

from .common import PROJECT_ROOT, RELIABILITY_ROOT, file_sha256, load_jsonl, stable_hash, write_json
from .prepare_scale_reflection import entity_key, object_key
from .evidence_provenance import evidence_is_direct, classify_evidence_layer

INDEX = 'agentm_evidence_stress_single_build_20260813'


def index_snapshot():
    import requests
    session = requests.Session()
    session.trust_env = False
    base = 'http://127.0.0.1:9200/' + INDEX
    def get(suffix, **kwargs):
        response = session.get(base + suffix, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()
    settings = get('/_settings')[INDEX]['settings']['index']
    data = get('/_search', json={'size': 5000, 'track_total_hits': True,
                              '_source': {'excludes': ['embedding']}, 'query': {'match_all': {}}})
    total = data['hits']['total']['value']
    hits = data['hits']['hits']
    if len(hits) != total or data.get('timed_out') or data.get('_shards', {}).get('failed'):
        raise RuntimeError('incomplete index export')
    docs = [{'_id': h['_id'], **h['_source']} for h in sorted(hits, key=lambda h: h['_id'])]
    return {'index': INDEX, 'uuid': settings['uuid'], 'document_count': total,
            'write_block': str(settings.get('blocks', {}).get('write')).lower() == 'true',
            'projection': 'all_source_fields_except_embedding', 'projection_sha256': stable_hash(docs), 'documents': docs}


def gold_audit(cases):
    from scripts.audit_balanced100_scientific_performance import Canonicalizer, load_admet_source, load_ddi_source, classify_membership
    canonical = Canonicalizer()
    cache = {}
    root = PROJECT_ROOT
    rows = []
    resources = {}
    for case in cases:
        if case.get('gold_value') is None:
            continue
        result = {'case_id': case.get('case_id', case.get('name')), 'dataset': case['dataset_name']}
        model = root / str(case['model_path'])
        result['worker_explicit_path_exists'] = (PROJECT_ROOT / case['model_path']).exists()
        result['source_model_path'] = str(model)
        try:
            if str(model) not in resources:
                resources[str(model)] = {'exists': model.exists(), 'sha256': file_sha256(model) if model.exists() else None}
            if case['task_type'] == 'ddi_prediction':
                source = load_ddi_source(model.parent, canonicalize=canonical, cache=cache)
                pair = tuple(canonical(s.strip()) for s in case['request'].split(',', 1))
                hits = source['index'].get(pair, [])
                result['double_cold'] = all(s not in source['train_drugs'] for s in pair)
            else:
                source = load_admet_source(case['dataset_name'], model.parent, resource_root=root, canonicalize=canonical, cache=cache)
                hits = source['index'].get(canonical(case['request']), [])
            membership = classify_membership(hits)
            result.update(membership_status=membership['membership_status'], actual_gold=membership['gold_value'],
                          gold_matches=membership['gold_value'] == case['gold_value'],
                          source_data_hash_verified=source['data_hash_verified'],
                          row_id_matches=any(str(h['row_id']) == str(case['gold_row_id']) and h['split'] == 'test' for h in hits))
            result['passed'] = (result['gold_matches'] and membership['gold_split'] == 'test'
                                and result['row_id_matches'] and result['source_data_hash_verified']
                                and result.get('double_cold', True) and model.exists())
        except Exception as exc:
            result.update(passed=False, error=f'{type(exc).__name__}: {exc}')
        rows.append(result)
    return rows, resources


def corpus_match(case, docs, aliases):
    """Conservative source-content coverage, NOT a hybrid retrieval acceptance."""
    endpoint = str(case.get('expected_endpoint') or case.get('dataset_name', '')).casefold()
    target = case['request']
    ddi = case['task_type'] == 'ddi_prediction'
    names = [s.strip().casefold() for s in target.split(',')] if ddi else aliases.get(entity_key(target), [])
    hits = []
    direct = []
    for doc in docs:
        doc_names = []
        for k in ('compound_names', 'drug_names', 'pair_drug_names', 'entity_terms', 'compound_synonyms'):
            value = doc.get(k, [])
            doc_names.extend([value] if isinstance(value, str) else value or [])
        doc_names = {str(v).casefold() for v in doc_names}
        structure = doc.get('canonical_smiles')
        structures = [structure] if isinstance(structure, str) else structure or []
        entity_match = (bool(names) and all(n in doc_names for n in names)) if ddi else (
            any(n in doc_names for n in names) or any(entity_key(s) == entity_key(target) for s in structures))
        if not entity_match:
            continue
        if not ddi:
            scope = {str(doc.get('endpoint', '')).casefold(), str(doc.get('dataset_name', '')).casefold()}
            if endpoint not in scope and str(case.get('dataset_name', '')).casefold() not in scope:
                continue
        hits.append(doc['_id'])
        # Use only existing source fields/body. Do not attach synthetic task
        # headers, endpoint flags or declarations to make a record pass.
        if evidence_is_direct(doc):
            direct.append(doc['_id'])
    return {'matched_document_ids': hits, 'direct_source_document_ids': direct,
            'source_content_direct_support_found': bool(direct),
            'scope': 'corpus_coverage_only_actual_hybrid_retrieval_not_tested'}


def audit(preparation, output, *, fetch_index):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    groups = {g: load_jsonl(preparation / f'{g}_additional_candidates.jsonl') for g in ('clean', 'stress', 'open_task', 'multi_turn')}
    gold, resources = gold_audit(groups['clean'])
    write_json(output / 'gold_source_audit.json', {'rows': gold, 'resources': resources}, overwrite=False)
    from scripts.run_reaction_benchmark import materialize_case
    reactions = []
    for row in groups['stress']:
        base = row['base_case']
        if not base['task_type'].startswith('reaction_'):
            continue
        original = {k: v for k, v in base.items() if k not in ('request', 'source_label', 'source_yield')}
        result = {'case_id': row['case_id'], 'source_dataset': base.get('source_dataset'), 'part': base.get('part')}
        try:
            reconstructed = materialize_case(original, 'data/reaction_sources')
            result['request_matches_raw_row'] = reconstructed['request'] == base['request']
            result['label_matches_raw_row'] = all(reconstructed.get(k) == base.get(k) for k in ('source_label', 'source_yield'))
            result['passed'] = result['request_matches_raw_row'] and result['label_matches_raw_row'] and base.get('part') == 'test'
        except Exception as exc:
            result.update(passed=False, error=str(exc))
        reactions.append(result)
    from .run_matrix import _open_task_contract
    contracts = []
    for row in groups['open_task']:
        contract = _open_task_contract(row)
        contracts.append({'case_id': row['case_id'], 'subtask_count': len(contract.get('subtasks', [])),
                          'no_gold_leakage': 'gold_value' not in json.dumps(contract),
                          'runtime_contract': contract, 'status': 'structural_only_not_scientific_acceptance'})
    write_json(output / 'reaction_source_audit.json', reactions, overwrite=False)
    write_json(output / 'open_contract_audit.json', contracts, overwrite=False)
    named = load_jsonl(PROJECT_ROOT / 'configs/molecular_safety_cases.jsonl')
    aliases = defaultdict(list)
    for r in named:
        if r['category'] == 'admet':
            name = r['name'].rsplit('_', 1)[0].replace('_', ' ')
            if name not in aliases[entity_key(r['request'])]:
                aliases[entity_key(r['request'])].append(name)
    coverage = []
    index_info = {'status': 'not_requested'}
    if fetch_index:
        snapshot = index_snapshot()
        write_json(output / 'index_source_projection.json', snapshot, overwrite=False)
        index_info = {k: v for k, v in snapshot.items() if k != 'documents'}
        for row in groups['stress']:
            base = row['base_case']
            if base['task_type'].startswith('reaction_'):
                continue
            coverage.append({'case_id': row['case_id'], 'fault': row['injection_type'],
                             'base_case_pool': row['base_case_pool'], **corpus_match(base, snapshot['documents'], aliases)})
        index_info['layers'] = dict(Counter(classify_evidence_layer(d) for d in snapshot['documents']))
        write_json(output / 'stress_corpus_coverage.json', coverage, overwrite=False)
    # Cross-family object reuse: these are not additional independent molecules.
    shared = defaultdict(list)
    for group in ('clean', 'stress', 'open_task'):
        for row in groups[group]:
            base = row.get('base_case', row)
            shared[object_key(base)].append([group, row['case_id']])
    overlap = [v for v in shared.values() if len(v) > 1]
    write_json(output / 'shared_base_objects.json', overlap, overwrite=False)
    summary = {'status': 'NOT_READY_FOR_LLM', 'external_llm_calls': 0, 'external_llm_tokens': 0,
               'gold_cases': len(gold), 'gold_verified': sum(r['passed'] for r in gold),
               'worker_model_path_missing': sum(not r['worker_explicit_path_exists'] for r in gold),
               'reaction_source_verified': sum(r['passed'] for r in reactions), 'reaction_source_cases': len(reactions),
               'open_contract_cases': len(contracts), 'cross_family_shared_object_clusters': len(overlap),
               'index': index_info,
               'stress_molecular_source_direct_coverage': sum(r['source_content_direct_support_found'] for r in coverage),
               'stress_molecular_source_checked': len(coverage),
               'remaining': ['Actual hybrid retrieval and model execution not run',
                             'Explicit model-path binding requires verification before job freezing',
                             'Name/structure aliases and multi-turn gold contracts require semantic checks',
                             'Source coverage cannot substitute for end-to-end real-RAG acceptance']}
    write_json(output / 'summary.json', summary, overwrite=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fetch-index', action='store_true')
    args = parser.parse_args()
    original = socket.socket.connect
    def local_only(sock, address):
        if not args.fetch_index or not isinstance(address, tuple) or address[:2] != ('127.0.0.1', 9200):
            raise RuntimeError('No API/network access allowed except read-only local index audit')
        return original(sock, address)
    with patch('socket.socket.connect', local_only):
        audit(args.preparation.resolve(), args.output.resolve(), fetch_index=args.fetch_index)


if __name__ == '__main__':
    main()
