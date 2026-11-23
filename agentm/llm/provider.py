from __future__ import annotations

import ast
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import requests

from agentm.utils.api_usage import record_api_call


DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KIMI_COMPATIBLE_BASE_URL = "https://api.moonshot.cn/v1"

DEFAULT_LLM_PROFILE = "qwen_flash"
DEFAULT_LLM_PROFILES: Dict[str, Dict[str, str]] = {
    "qwen_flash": {
        "provider": "qwen",
        "model": "qwen3.6-flash",
        "base_url": DASHSCOPE_COMPATIBLE_BASE_URL,
        "base_url_env": "DASHSCOPE_BASE_URL",
        "base_url_secret": "dashscope_base_url",
        "api_key_env": "DASHSCOPE_API_KEY",
        "api_key_secret": "dashscope_api_key",
    },
    "qwen_3_7": {
        "provider": "qwen",
        "model": "qwen3.7",
        "base_url": DASHSCOPE_COMPATIBLE_BASE_URL,
        "base_url_env": "DASHSCOPE_BASE_URL",
        "base_url_secret": "dashscope_base_url",
        "api_key_env": "DASHSCOPE_API_KEY",
        "api_key_secret": "dashscope_api_key",
    },
    "kimi": {
        "provider": "kimi",
        "model": "kimi-k2-latest",
        "base_url": KIMI_COMPATIBLE_BASE_URL,
        "base_url_env": "KIMI_BASE_URL",
        "base_url_secret": "kimi_base_url",
        "api_key_env": "KIMI_API_KEY",
        "api_key_secret": "kimi_api_key",
    },
    "openai": {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "base_url": "https://api.openai.com/v1",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url_secret": "openai_base_url",
        "api_key_env": "OPENAI_API_KEY",
        "api_key_secret": "openai_api_key",
    },
}


@dataclass
class LLMConfig:
    profile: str = DEFAULT_LLM_PROFILE
    provider: str = "qwen"
    model: str = "qwen3.6-flash"
    base_url: str = DASHSCOPE_COMPATIBLE_BASE_URL
    api_key: Optional[str] = None
    temperature: float = 0.1
    timeout: int = 60


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    @property
    def available(self) -> bool:
        return bool(self.config.api_key)

    def chat_json(self, messages: List[Dict[str, str]], *, temperature: Optional[float] = None) -> str:
        if not self.available:
            raise RuntimeError("No LLM API key configured.")

        provider = self.config.provider.lower()
        if provider in {"qwen", "dashscope", "bailian", "kimi", "moonshot"}:
            return self._openai_compatible_chat(messages, temperature=temperature)
        if provider in {"openai", "openai_compatible"}:
            return self._openai_compatible_chat(messages, temperature=temperature)
        raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    def _openai_compatible_chat(self, messages: List[Dict[str, str]], *, temperature: Optional[float] = None) -> str:
        base_url = self.config.base_url.rstrip("/")
        endpoint = f"{base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "response_format": {"type": "json_object"},
        }
        started_at = time.perf_counter()
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data: Dict[str, Any] = response.json()
            content = data["choices"][0]["message"]["content"] or "{}"
            record_api_call(
                provider=self.config.provider,
                profile=self.config.profile,
                model=self.config.model,
                call_type="chat_json",
                status="completed",
                started_at=started_at,
                messages=messages,
                response_text=content,
                usage=data.get("usage") if isinstance(data.get("usage"), dict) else None,
                endpoint=endpoint,
                temperature=payload["temperature"],
                extra={"response_id": data.get("id")},
            )
            return content
        except Exception as exc:
            record_api_call(
                provider=self.config.provider,
                profile=self.config.profile,
                model=self.config.model,
                call_type="chat_json",
                status="failed",
                started_at=started_at,
                messages=messages,
                endpoint=endpoint,
                temperature=payload["temperature"],
                error=str(exc),
            )
            raise


def _local_secret_module_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "secret_keys.py"


def _read_local_secret_literal(name: str) -> Any:
    secret_path = _local_secret_module_path()
    if not secret_path.exists():
        return None

    try:
        tree = ast.parse(secret_path.read_text(encoding="utf-8"), filename=str(secret_path))
    except Exception:
        return None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if name in target_names:
                try:
                    return ast.literal_eval(node.value)
                except Exception:
                    return None
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            try:
                return ast.literal_eval(node.value)
            except Exception:
                return None

    return None


def _read_local_secret(name: str) -> Optional[str]:
    value = _read_local_secret_literal(name)
    if value is None:
        return None

    value_text = str(value).strip()
    return value_text or None


def _configured_value(env_names: List[str], secret_names: List[str], default: Optional[str] = None) -> Optional[str]:
    for env_name in env_names:
        value = os.getenv(env_name)
        if value:
            return value
    for secret_name in secret_names:
        value = _read_local_secret(secret_name)
        if value:
            return value
    return default


def _local_profile(profile_name: str) -> Dict[str, str]:
    profiles = _read_local_secret_literal("agentm_llm_profiles")
    if not isinstance(profiles, dict):
        return {}
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        return {}
    return {str(key): str(value) for key, value in profile.items() if value is not None}


def _profile_config(profile_name: str) -> Dict[str, str]:
    config = dict(DEFAULT_LLM_PROFILES.get(profile_name, {}))
    config.update(_local_profile(profile_name))
    if config:
        return config

    fallback = dict(DEFAULT_LLM_PROFILES[DEFAULT_LLM_PROFILE])
    fallback.update(_local_profile(DEFAULT_LLM_PROFILE))
    return fallback


def _profile_exists(profile_name: str) -> bool:
    return profile_name in DEFAULT_LLM_PROFILES or bool(_local_profile(profile_name))


def _profile_api_key(profile: Dict[str, str]) -> Optional[str]:
    env_name = profile.get("api_key_env")
    if env_name:
        value = os.getenv(env_name)
        if value:
            return value

    secret_name = profile.get("api_key_secret")
    if secret_name:
        value = _read_local_secret(secret_name)
        if value:
            return value

    value = profile.get("api_key")
    if value:
        return value
    return None


def _profile_base_url(profile: Dict[str, str]) -> Optional[str]:
    env_name = profile.get("base_url_env")
    if env_name:
        value = os.getenv(env_name)
        if value:
            return value

    secret_name = profile.get("base_url_secret")
    if secret_name:
        value = _read_local_secret(secret_name)
        if value:
            return value

    value = profile.get("base_url")
    if value:
        return value
    return None


def build_llm_client(profile_name: Optional[str] = None) -> LLMClient:
    active_profile = profile_name or _configured_value(
        ["AGENTM_LLM_PROFILE"],
        ["agentm_active_llm_profile"],
        DEFAULT_LLM_PROFILE,
    )
    resolved_profile = active_profile if active_profile and _profile_exists(active_profile) else DEFAULT_LLM_PROFILE
    profile = _profile_config(resolved_profile)
    provider = _configured_value(
        ["AGENTM_LLM_PROVIDER"],
        ["agentm_llm_provider"],
        profile.get("provider", "qwen"),
    )
    model = _configured_value(
        ["AGENTM_LLM_MODEL"],
        ["agentm_llm_model"],
        profile.get("model", "qwen3.6-flash"),
    )
    base_url = _configured_value(["AGENTM_LLM_BASE_URL"], [], _profile_base_url(profile))
    api_key = _configured_value(
        ["AGENTM_LLM_API_KEY"],
        ["agentm_llm_api_key"],
    ) or _profile_api_key(profile)
    temperature = float(os.getenv("AGENTM_LLM_TEMPERATURE", "0.1"))
    timeout = int(os.getenv("AGENTM_LLM_TIMEOUT", "60"))
    return LLMClient(
        LLMConfig(
            profile=resolved_profile,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout,
        )
    )
