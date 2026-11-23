"""Offline regression checks for the consolidated repository layout."""
from pathlib import Path
import ast
import hashlib
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), default=str).encode()).hexdigest()


class RepositoryLayoutTests(unittest.TestCase):
    def test_public_controller_is_the_guarded_controller(self):
        from agentm import AgentMMultiAgentController
        from agentm.runtime.controller import AgentMMultiAgentController as expected
        self.assertIs(AgentMMultiAgentController, expected)

    def test_component_import_roots_are_inside_the_repository(self):
        import agentm
        import agentm.runtime
        for module in (agentm, agentm.runtime):
            self.assertTrue(Path(module.__file__).resolve().is_relative_to(ROOT))

    def test_resource_roots_are_unified(self):
        from agentm.runtime import compat as runtime
        self.assertEqual(runtime.RESOURCE_ROOT, ROOT)
        self.assertEqual(runtime.PROJECT_ROOT, ROOT)
        self.assertEqual(runtime.RUNTIME_ROOT, ROOT)

    def test_web_template_is_self_contained(self):
        from agentm.runtime.web.demo_server import INDEX_HTML
        template = ROOT / 'agentm/runtime/web/templates/index.html'
        self.assertEqual(INDEX_HTML, template.read_text(encoding='utf-8'))
        self.assertIn('function renderWorkflow(data)', INDEX_HTML)

    def test_default_resource_paths_are_project_relative(self):
        from agentm.runtime.resources import default_resource_path
        self.assertEqual(Path(default_resource_path('data/rag_corpus/tdc_real')),
                         ROOT / 'data/rag_corpus/tdc_real')

    def test_local_imports_resolve_statically(self):
        for path in [*ROOT.glob('agentm/**/*.py'), *ROOT.glob('scripts/*.py'),
                     *ROOT.glob('experiments/**/*.py'), *ROOT.glob('tests/*.py')]:
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                    names = [node.module]
                for name in names:
                    if name.startswith(('agentm.', 'experiments.agentm_reliability.', 'scripts.')):
                        module = ROOT.joinpath(*name.split('.'))
                        self.assertTrue(module.is_dir() or module.with_suffix('.py').is_file(),
                                        (str(path.relative_to(ROOT)), node.lineno, name))

    def test_no_duplicate_dictionary_keys_in_source(self):
        for path in [*ROOT.glob('scripts/*.py'), *ROOT.glob('agentm/**/*.py')]:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Dict):
                    keys = [key.value for key in node.keys
                            if isinstance(key, ast.Constant) and isinstance(key.value, str)]
                    self.assertEqual(len(keys), len(set(keys)), (path, node.lineno))

    def test_stress_base_case_hashes_match_packaged_records(self):
        directory = ROOT / 'experiments/agentm_reliability/manifests'
        for path in directory.glob('stress*.jsonl'):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                self.assertEqual(row['base_case_hash'], stable_hash(row['base_case']))

    def test_adaptive_case_hashes_match_packaged_records(self):
        directory = ROOT / 'experiments/agentm_reliability/manifests'
        for name in ('open_tasks.jsonl', 'multi_turn_tasks.jsonl'):
            for line in (directory / name).read_text().splitlines():
                row = json.loads(line)
                self.assertEqual(row['case_hash'], stable_hash({k: v for k, v in row.items() if k != 'case_hash'}))

    def test_clean_source_hashes_match_packaged_cases(self):
        source = {r['name']: r for r in map(json.loads, (ROOT / 'configs/molecular_safety_cases.jsonl').read_text().splitlines())}
        for line in (ROOT / 'experiments/agentm_reliability/manifests/clean.jsonl').read_text().splitlines():
            row = json.loads(line)
            self.assertEqual(row['source_case_hash'], stable_hash(source[row['name']]))


if __name__ == '__main__':
    unittest.main()
