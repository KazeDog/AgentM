"""Check command-line imports and help text with all network access blocked."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys

from offline_guard import activate, verify_import_roots, NETWORK_ATTEMPTS, ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = args.report.resolve() if args.report else None
    activate()
    imports = verify_import_roots()
    sys.path.insert(0, str(ROOT / 'scripts'))
    paths = sorted((ROOT / 'scripts').glob('*.py'))
    modules = [
        'experiments.agentm_reliability.' + name for name in (
            'run_matrix', 'build_clean_manifest', 'build_stress_manifest',
            'build_open_task_manifest', 'build_multi_turn_manifest',
            'prepare_single_build_stress', 'preflight_single_build_stress',
            'prepare_depth_jobs',
            'multi_turn_runner', 'canonicalize_runs', 'offline_evaluator',
            'statistics', 'render_outputs', 'render_manuscript_composites',
        )
    ]
    results = []
    for entry in [*paths, *modules]:
        label = str(entry.relative_to(ROOT)) if isinstance(entry, Path) else entry
        stdout, stderr = io.StringIO(), io.StringIO()
        sys.argv = [label, '--help']
        error = None
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                if isinstance(entry, Path):
                    runpy.run_path(str(entry), run_name='__main__')
                else:
                    runpy.run_module(entry, run_name='__main__')
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    error = f'exit {exc.code}: {stderr.getvalue()}'
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        results.append({'entrypoint': label, 'passed': error is None and 'usage:' in stdout.getvalue().lower(),
                        'error': error})
    summary = {'passed': all(row['passed'] for row in results) and not NETWORK_ATTEMPTS,
               'entrypoints_checked': len(results), 'results': results,
               'network_operations_blocked': len(NETWORK_ATTEMPTS), 'api_calls_sent': 0,
               'import_roots': imports}
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=2))
    for row in results:
        if not row['passed']:
            print(json.dumps(row))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
