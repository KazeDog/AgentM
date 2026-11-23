"""Validate read-only web routes without starting a server or making model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from offline_guard import activate, verify_import_roots, NETWORK_ATTEMPTS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = args.report.resolve() if args.report else None
    activate()
    imports = verify_import_roots()
    from fastapi.testclient import TestClient
    from agentm.runtime.web.demo_server import create_app, WebDemoConfig

    checks = {}
    with tempfile.TemporaryDirectory(prefix='agentm-web-check-') as temporary:
        app = create_app(WebDemoConfig(output_dir=Path(temporary),
                                      use_llm_orchestrator=False,
                                      use_llm_reasoning=False,
                                      use_rag_evidence=False,
                                      allow_network_lookup=False))
        with TestClient(app) as client:
            page = client.get('/')
            health = client.get('/api/health')
            runs = client.get('/api/runs')
            schema = client.get('/openapi.json')
        checks['index_html'] = page.status_code == 200 and 'AgentM' in page.text
        checks['health'] = health.status_code == 200 and health.json()['status'] == 'ok'
        checks['empty_run_list'] = runs.status_code == 200 and runs.json()['runs'] == []
        checks['project_name_api_metadata'] = schema.status_code == 200 and schema.json()['info']['version'] == 'AgentM'
    summary = {'passed': all(checks.values()) and not NETWORK_ATTEMPTS,
               'checks': checks, 'api_calls_sent': 0,
               'network_operations_blocked': len(NETWORK_ATTEMPTS),
               'import_roots': imports,
               'scope': 'in-process read-only web routes; no scientific execution'}
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
