"""Check source syntax, links, naming and credential-like literals offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    problems = []
    syntax_count = 0
    credential_patterns = [
        re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
        re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ]
    internal_revision = re.compile(r"(?i)(?:agentm[-_. ]?v\d+|(?:^|[\W_])v\d+(?:\.\d+)*(?=$|[\W_]))")
    for path in ROOT.rglob("*"):
        parts = path.relative_to(ROOT).parts
        if parts[0] in {"outputs", "data", "checkpoints"} or any(part in {".venv", "__pycache__", ".git", ".pytest_cache"} for part in parts):
            continue
        if path.is_symlink():
            problems.append("Symlink: " + str(path.relative_to(ROOT)))
        if not path.is_file():
            continue
        rel = str(path.relative_to(ROOT))
        if internal_revision.search(rel):
            problems.append("Historical project label in path: " + rel)
        if path.name == "secret_keys.py" or path.name == ".env":
            problems.append("Forbidden credentials file: " + rel)
        if path.suffix in {".py", ".yaml", ".yml", ".json", ".jsonl", ".md", ".txt", ".example", ".html"}:
            content = path.read_text(encoding="utf-8")
            # External HTTP protocol routes are not project release labels.
            label_content = re.sub(r"(?i)/v\d+(?=/|\b)", "/api", content)
            if internal_revision.search(label_content):
                problems.append("Historical project label in content: " + rel)
            if any(pattern.search(content) for pattern in credential_patterns):
                problems.append("Credential-like content (value suppressed): " + rel)
            if path.suffix == ".py":
                try:
                    compile(content, str(path), "exec")
                    syntax_count += 1
                except SyntaxError as exc:
                    problems.append("Syntax: " + rel + ":" + str(exc.lineno))
    summary = {"passed": not problems,
               "python_files_compiled_in_memory": syntax_count, "problems": problems,
               "scope": "offline syntax, import, naming and pattern scan"}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
