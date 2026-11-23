from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agentm.llm.provider import LLMClient, build_llm_client
from agentm.tools.name_resolver import LOCAL_NAME_TO_SMILES, network_lookup_enabled, resolve_name_to_smiles
from agentm.tools.smiles import likely_smiles, standardize_smiles
from agentm.utils.io import ensure_dir, write_json


@dataclass
class DrugResolution:
    query: str
    smiles: Optional[str] = None
    canonical_smiles: Optional[str] = None
    canonical_name: Optional[str] = None
    source: str = "unresolved"
    is_valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DrugResolverAgent:
    """Resolve drug names or SMILES with traceable fallbacks.

    Resolution order:
    local dictionary -> direct SMILES -> verified cache -> PubChem -> optional LLM candidate.
    """

    def __init__(
        self,
        cache_path: str | Path | None = None,
        *,
        allow_network: Optional[bool] = None,
        use_llm: Optional[bool] = None,
        llm_profile: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.cache_path = Path(cache_path) if cache_path else self._default_cache_path()
        self.allow_network = network_lookup_enabled(default=True) if allow_network is None else bool(allow_network)
        self.use_llm = self._llm_enabled(default=False) if use_llm is None else bool(use_llm)
        self.llm_profile = llm_profile
        self.llm_client = llm_client if llm_client is not None else (build_llm_client(llm_profile) if self.use_llm else None)

    def resolve(self, value: str) -> DrugResolution:
        query = (value or "").strip()
        result = DrugResolution(query=query, canonical_name=query or None)
        if not query:
            result.errors.append("empty_query")
            result.trace.append(self._trace("input", "failed", reason="empty_query"))
            return result

        local_smiles = LOCAL_NAME_TO_SMILES.get(query.lower())
        if local_smiles:
            return self._resolution_from_smiles(
                query=query,
                smiles=local_smiles,
                source="local_dictionary",
                canonical_name=query,
                trace=[self._trace("local_dictionary", "hit")],
            )

        if likely_smiles(query):
            return self._resolution_from_smiles(
                query=query,
                smiles=query,
                source="smiles",
                canonical_name=None,
                trace=[self._trace("smiles_parse", "attempt")],
            )

        cached = self._cache_entry(query)
        if cached:
            trace = [self._trace("cache", "hit", source=cached.get("source"))]
            resolved = self._resolution_from_smiles(
                query=query,
                smiles=str(cached.get("canonical_smiles") or cached.get("smiles") or ""),
                source=str(cached.get("cache_source") or f"cache:{cached.get('source', 'unknown')}"),
                canonical_name=str(cached.get("canonical_name") or query),
                trace=trace,
            )
            resolved.metadata["cache_entry"] = cached
            return resolved
        result.trace.append(self._trace("cache", "miss", cache_path=str(self.cache_path)))

        if self.allow_network:
            pubchem_smiles, pubchem_source = resolve_name_to_smiles(query, allow_network=True)
            result.trace.append(self._trace("pubchem", "success" if pubchem_smiles else "failed", source=pubchem_source))
            if pubchem_smiles:
                resolved = self._resolution_from_smiles(
                    query=query,
                    smiles=pubchem_smiles,
                    source=pubchem_source,
                    canonical_name=query,
                    trace=result.trace,
                )
                if resolved.is_valid:
                    self._store_cache(query, resolved, source=pubchem_source)
                return resolved
            result.errors.append(f"pubchem_lookup_failed:{pubchem_source}")
        else:
            result.trace.append(self._trace("pubchem", "skipped", reason="network_lookup_disabled"))

        llm_candidate = self._try_llm_candidate(query)
        result.trace.extend(llm_candidate.trace)
        result.warnings.extend(llm_candidate.warnings)
        if llm_candidate.is_valid:
            llm_candidate.trace = list(result.trace)
            self._store_cache(query, llm_candidate, source=llm_candidate.source)
            return llm_candidate

        result.errors.extend(llm_candidate.errors)
        if not result.errors:
            result.errors.append("unresolved")
        return result

    def _resolution_from_smiles(
        self,
        *,
        query: str,
        smiles: str,
        source: str,
        canonical_name: Optional[str],
        trace: List[Dict[str, Any]],
    ) -> DrugResolution:
        canonical, err = standardize_smiles(smiles)
        result = DrugResolution(
            query=query,
            smiles=smiles,
            canonical_smiles=canonical,
            canonical_name=canonical_name,
            source=source,
            is_valid=err is None and bool(canonical),
            trace=list(trace),
        )
        if err:
            result.errors.append(err)
            result.trace.append(self._trace(source, "validation_failed", error=err))
        else:
            result.trace.append(self._trace("rdkit_validation", "success"))
        return result

    def _try_llm_candidate(self, query: str) -> DrugResolution:
        result = DrugResolution(query=query, canonical_name=query)
        if not self.use_llm:
            result.trace.append(self._trace("llm_drug_resolver", "skipped", reason="disabled"))
            return result
        if self.llm_client is None or not self.llm_client.available:
            result.trace.append(self._trace("llm_drug_resolver", "skipped", reason="unavailable"))
            return result

        try:
            payload = self._llm_payload(query)
            candidate_smiles = str(payload.get("candidate_smiles") or payload.get("smiles") or "").strip()
            canonical_name = str(payload.get("canonical_name") or query).strip()
            if not candidate_smiles:
                result.errors.append("llm_returned_no_smiles")
                result.trace.append(self._trace("llm_drug_resolver", "failed", reason="no_smiles"))
                return result
            resolved = self._resolution_from_smiles(
                query=query,
                smiles=candidate_smiles,
                source="llm_candidate_rdkit_validated",
                canonical_name=canonical_name,
                trace=[self._trace("llm_drug_resolver", "candidate_returned", profile=self.llm_client.config.profile)],
            )
            resolved.metadata["llm_payload"] = payload
            resolved.warnings.append(
                "LLM supplied a candidate SMILES that passed RDKit validation but was not cross-checked against PubChem."
            )
            return resolved
        except Exception as exc:
            result.errors.append(f"llm_lookup_failed:{exc}")
            result.trace.append(self._trace("llm_drug_resolver", "failed", error=str(exc)))
            return result

    def _llm_payload(self, query: str) -> Dict[str, Any]:
        assert self.llm_client is not None
        response = self.llm_client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You resolve drug names to cautious, auditable molecular identifiers. "
                        "Return JSON only. If uncertain, return an empty candidate_smiles."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Resolve this drug mention for a small-molecule workflow.\n"
                        f"Drug mention: {query}\n"
                        "Return fields: canonical_name, synonyms, candidate_smiles, confidence, notes. "
                        "Use a SMILES only if it is a specific small molecule."
                    ),
                },
            ],
            temperature=0.0,
        )
        return self._parse_json_object(response)

    def _parse_json_object(self, response: str) -> Dict[str, Any]:
        text = (response or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("LLM drug resolver response must be a JSON object.")
        return payload

    def _cache_entry(self, query: str) -> Optional[Dict[str, Any]]:
        data = self._read_cache()
        entries = data.get("entries", {})
        entry = entries.get(self._cache_key(query))
        return entry if isinstance(entry, dict) else None

    def _store_cache(self, query: str, resolution: DrugResolution, *, source: str) -> None:
        if not resolution.is_valid or not resolution.canonical_smiles:
            return
        data = self._read_cache()
        entries = data.setdefault("entries", {})
        entries[self._cache_key(query)] = {
            "query": query,
            "canonical_name": resolution.canonical_name or query,
            "canonical_smiles": resolution.canonical_smiles,
            "source": source,
            "cache_source": f"cache:{source}",
            "validated_by": "rdkit",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": list(resolution.warnings),
        }
        data["schema"] = "agentm_drug_resolution"
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(self.cache_path, data)

    def _read_cache(self) -> Dict[str, Any]:
        if not self.cache_path.exists():
            return {"schema": "agentm_drug_resolution", "entries": {}}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {"schema": "agentm_drug_resolution", "entries": {}}
        if not isinstance(data, dict):
            return {"schema": "agentm_drug_resolution", "entries": {}}
        if not isinstance(data.get("entries"), dict):
            data["entries"] = {}
        return data

    def _cache_key(self, query: str) -> str:
        return " ".join(query.strip().lower().split())

    def _trace(self, stage: str, status: str, **extra: Any) -> Dict[str, Any]:
        item = {"stage": stage, "status": status}
        item.update({key: value for key, value in extra.items() if value is not None})
        return item

    def _default_cache_path(self) -> Path:
        env_path = os.getenv("AGENTM_DRUG_RESOLVER_CACHE")
        if env_path:
            return Path(env_path)
        project_root = Path(__file__).resolve().parents[2]
        ensure_dir(project_root / "data")
        return project_root / "data" / "drug_name_cache.json"

    def _llm_enabled(self, default: bool = False) -> bool:
        value = os.getenv("AGENTM_ENABLE_LLM_DRUG_RESOLVER")
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}
