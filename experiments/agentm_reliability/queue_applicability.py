"""Pre-runtime applicability gate for new depth-aware jobs, never authorization.

The ledger is shared across arms. Unknown conditions may only enter an explicit
zero-API heuristic diagnostic; they must never enter a paid/formal queue.
Historical jobs without a depth protocol are not reclassified here.
"""
from pathlib import Path

from .common import file_sha256, load_jsonl, read_json, source_tree_hash, stable_hash

PROTOCOL = 'case_applicability_ledger'
DIAGNOSTIC_SCOPE = 'zero_API_applicability_probe_only'


def validate_queue_applicability(job):
    binding = job.get('applicability_binding') or {}
    path = Path(binding.get('path', ''))
    if not path.is_absolute() or not path.is_file() or file_sha256(path) != binding.get('sha256'):
        raise PermissionError('missing or changed applicability ledger')
    ledger = read_json(path)
    if ledger.get('protocol') != PROTOCOL:
        raise PermissionError('unknown applicability protocol')
    if ledger.get('source_tree_hash') != job.get('source_tree_hash') or source_tree_hash() != job.get('source_tree_hash'):
        raise PermissionError('applicability source snapshot mismatch')
    manifest = Path(job.get('source_manifest', ''))
    if (not manifest.is_absolute() or not manifest.is_file()
            or ledger.get('manifest_hashes', {}).get(str(manifest)) != file_sha256(manifest)):
        raise PermissionError('applicability manifest mismatch')
    case = job['case']
    if case.get('case_id') != job.get('case_id') or case.get('dataset_group') != job.get('dataset_group'):
        raise PermissionError('applicability job/case identity mismatch')
    originals = [r for r in load_jsonl(manifest) if r.get('case_id') == job['case_id']]
    if len(originals) != 1 or stable_hash(originals[0]) != stable_hash(case):
        raise PermissionError('applicability case differs from manifest')
    matches = [r for r in ledger.get('cases', []) if r.get('case_id') == job['case_id']]
    if len(matches) != 1 or matches[0].get('case_sha256') != stable_hash(case):
        raise PermissionError('missing, duplicate or altered applicability case')
    row = matches[0]
    status = row.get('status')
    if status not in ('eligible', 'pending_probe'):
        raise PermissionError('case applicability forbids execution: ' + str(status))
    runtime = job['config']['runtime']
    diagnostic = (job.get('authorization_scope') == DIAGNOSTIC_SCOPE
        and job.get('formal_statistics_eligible') is False
        and job.get('orchestrator') == 'heuristic'
        and type(runtime.get('max_llm_calls_per_run')) is int and runtime['max_llm_calls_per_run'] == 0
        and type(runtime.get('max_total_tokens_per_run')) is int and runtime['max_total_tokens_per_run'] == 0
        and runtime.get('use_llm_reasoning') is False
        and runtime.get('require_llm_reasoning') is False
        and runtime.get('require_llm_orchestrator') is False
        and runtime.get('llm_profile') is None)
    if diagnostic:
        if ledger.get('diagnostic_probe_permitted') is not True:
            raise PermissionError('ledger does not permit diagnostic probes')
    elif (status != 'eligible' or ledger.get('formal_freeze_ready') is not True
          or row.get('formal_eligible') is not True):
        raise PermissionError('applicability has not passed the formal freeze gate')
    return {'protocol': PROTOCOL, 'case_id': job['case_id'], 'status': status,
            'ledger_sha256': binding['sha256'], 'diagnostic_only': diagnostic}
