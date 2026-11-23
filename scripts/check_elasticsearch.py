#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    argparse.ArgumentParser(description="Check the configured Elasticsearch service.").parse_args()
    url = os.getenv("AGENTM_ES_URL", "http://127.0.0.1:9200")
    try:
        from elasticsearch import Elasticsearch

        client = Elasticsearch(url)
        info = client.info()
        print(json.dumps({"status": "ok", "url": url, "cluster_name": info.get("cluster_name"), "version": info.get("version", {})}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "url": url, "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
