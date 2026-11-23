from __future__ import annotations

import ast
import base64
from dataclasses import dataclass
import mimetypes
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from agentm.utils.api_usage import record_api_call


DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MOONSHOT_COMPATIBLE_BASE_URL = "https://api.moonshot.cn/v1"

DEFAULT_MLLM_PROFILE = "kimi_k2_6_bailian"
DEFAULT_MLLM_PROFILES: Dict[str, Dict[str, str]] = {
    "kimi_k2_6_bailian": {
        "provider": "bailian_kimi",
        "model": "kimi-k2.6",
        "base_url": DASHSCOPE_COMPATIBLE_BASE_URL,
        "base_url_env": "DASHSCOPE_BASE_URL",
        "base_url_secret": "dashscope_base_url",
        "api_key_env": "DASHSCOPE_API_KEY",
        "api_key_secret": "dashscope_api_key",
    },
    "kimi_k2_6_moonshot": {
        "provider": "moonshot_kimi",
        "model": "kimi-k2.6",
        "base_url": MOONSHOT_COMPATIBLE_BASE_URL,
        "base_url_env": "KIMI_BASE_URL",
        "base_url_secret": "kimi_base_url",
        "api_key_env": "KIMI_API_KEY",
        "api_key_secret": "kimi_api_key",
    },
    "qwen_vl_bailian": {
        "provider": "qwen_vl",
        "model": "qwen-vl-plus",
        "base_url": DASHSCOPE_COMPATIBLE_BASE_URL,
        "base_url_env": "DASHSCOPE_BASE_URL",
        "base_url_secret": "dashscope_base_url",
        "api_key_env": "DASHSCOPE_API_KEY",
        "api_key_secret": "dashscope_api_key",
    },
}


@dataclass
class MLLMConfig:
    profile: str = DEFAULT_MLLM_PROFILE
    provider: str = "bailian_kimi"
    model: str = "kimi-k2.6"
    base_url: str = DASHSCOPE_COMPATIBLE_BASE_URL
    api_key: Optional[str] = None
    temperature: float = 0.1
    timeout: int = 90
    max_retries: int = 3
    retry_backoff_sec: float = 5.0
    request_interval_sec: float = 0.0


class MLLMClient:
    def __init__(self, config: MLLMConfig):
        self.config = config
        self._last_request_time = 0.0
        self._request_interval_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.config.api_key)

    def image_json(self, *, image_path: str | Path, prompt: str, temperature: Optional[float] = None) -> str:
        if not self.available:
            raise RuntimeError("No MLLM API key configured.")

        image_path_obj = Path(image_path)
        image_url = _image_data_url(image_path_obj)
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        started_at = time.perf_counter()
        try:
            response = self._post_with_retries(payload)
            data: Dict[str, Any] = response.json()
            content = data["choices"][0]["message"]["content"] or "{}"
            record_api_call(
                provider=self.config.provider,
                profile=self.config.profile,
                model=self.config.model,
                call_type="image_json",
                status="completed",
                started_at=started_at,
                prompt=prompt,
                response_text=content,
                usage=data.get("usage") if isinstance(data.get("usage"), dict) else None,
                endpoint=endpoint,
                temperature=payload["temperature"],
                extra={
                    "image_path": str(image_path_obj),
                    "image_bytes": image_path_obj.stat().st_size if image_path_obj.exists() else None,
                    "max_retries": self.config.max_retries,
                    "response_id": data.get("id"),
                },
            )
            return content
        except Exception as exc:
            record_api_call(
                provider=self.config.provider,
                profile=self.config.profile,
                model=self.config.model,
                call_type="image_json",
                status="failed",
                started_at=started_at,
                prompt=prompt,
                endpoint=endpoint,
                temperature=payload["temperature"],
                error=str(exc),
                extra={
                    "image_path": str(image_path_obj),
                    "image_bytes": image_path_obj.stat().st_size if image_path_obj.exists() else None,
                    "max_retries": self.config.max_retries,
                },
            )
            raise

    def _post_with_retries(self, payload: Dict[str, Any]) -> requests.Response:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        attempts = max(1, int(self.config.max_retries) + 1)
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            self._respect_request_interval()
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
                if response.status_code < 400:
                    return response
                if response.status_code not in {429, 500, 502, 503, 504} or attempt == attempts - 1:
                    raise RuntimeError(_http_error_message(response))
                last_error = RuntimeError(_http_error_message(response))
            except requests.RequestException as exc:
                last_error = exc
                if attempt == attempts - 1:
                    raise RuntimeError(f"MLLM request failed after {attempts} attempts: {exc}") from exc
            sleep_sec = self.config.retry_backoff_sec * (2**attempt)
            time.sleep(max(0.0, sleep_sec))
        raise RuntimeError(str(last_error) if last_error else "MLLM request failed")

    def _respect_request_interval(self) -> None:
        interval = max(0.0, float(self.config.request_interval_sec))
        with self._request_interval_lock:
            if interval > 0:
                elapsed = time.monotonic() - self._last_request_time
                if elapsed < interval:
                    time.sleep(interval - elapsed)
            self._last_request_time = time.monotonic()


def _image_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _http_error_message(response: requests.Response) -> str:
    text = response.text.strip().replace("\n", " ")
    if len(text) > 1000:
        text = text[:1000] + "..."
    return (
        f"{response.status_code} {response.reason} for url: {response.url}; "
        f"response_body={text or '<empty>'}"
    )


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
    text = str(value).strip()
    return text or None


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
    profiles = _read_local_secret_literal("agentm_mllm_profiles")
    if not isinstance(profiles, dict):
        return {}
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        return {}
    return {str(key): str(value) for key, value in profile.items() if value is not None}


def _profile_exists(profile_name: str) -> bool:
    return profile_name in DEFAULT_MLLM_PROFILES or bool(_local_profile(profile_name))


def _profile_config(profile_name: str) -> Dict[str, str]:
    config = dict(DEFAULT_MLLM_PROFILES.get(profile_name, {}))
    config.update(_local_profile(profile_name))
    if config:
        return config
    fallback = dict(DEFAULT_MLLM_PROFILES[DEFAULT_MLLM_PROFILE])
    fallback.update(_local_profile(DEFAULT_MLLM_PROFILE))
    return fallback


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


def build_mllm_client(profile_name: Optional[str] = None) -> MLLMClient:
    active_profile = profile_name or _configured_value(
        ["AGENTM_MLLM_PROFILE", "AGENTM_MLMM_PROFILE"],
        ["agentm_active_mllm_profile", "agentm_active_mlmm_profile"],
        DEFAULT_MLLM_PROFILE,
    )
    resolved_profile = active_profile if active_profile and _profile_exists(active_profile) else DEFAULT_MLLM_PROFILE
    profile = _profile_config(resolved_profile)
    provider = _configured_value(["AGENTM_MLLM_PROVIDER", "AGENTM_MLMM_PROVIDER"], ["agentm_mllm_provider"], profile.get("provider"))
    model = _configured_value(["AGENTM_MLLM_MODEL", "AGENTM_MLMM_MODEL"], ["agentm_mllm_model"], profile.get("model"))
    base_url = _configured_value(["AGENTM_MLLM_BASE_URL", "AGENTM_MLMM_BASE_URL"], ["agentm_mllm_base_url"], _profile_base_url(profile))
    api_key = _configured_value(["AGENTM_MLLM_API_KEY", "AGENTM_MLMM_API_KEY"], ["agentm_mllm_api_key"], _profile_api_key(profile))
    temperature = float(os.getenv("AGENTM_MLLM_TEMPERATURE", os.getenv("AGENTM_MLMM_TEMPERATURE", "0.1")))
    timeout = int(os.getenv("AGENTM_MLLM_TIMEOUT", os.getenv("AGENTM_MLMM_TIMEOUT", "90")))
    max_retries = int(os.getenv("AGENTM_MLLM_MAX_RETRIES", os.getenv("AGENTM_MLMM_MAX_RETRIES", "3")))
    retry_backoff_sec = float(os.getenv("AGENTM_MLLM_RETRY_BACKOFF_SEC", os.getenv("AGENTM_MLMM_RETRY_BACKOFF_SEC", "5")))
    request_interval_sec = float(os.getenv("AGENTM_MLLM_REQUEST_INTERVAL_SEC", os.getenv("AGENTM_MLMM_REQUEST_INTERVAL_SEC", "0")))
    return MLLMClient(
        MLLMConfig(
            profile=resolved_profile,
            provider=provider or "bailian_kimi",
            model=model or "kimi-k2.6",
            base_url=base_url or DASHSCOPE_COMPATIBLE_BASE_URL,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff_sec=retry_backoff_sec,
            request_interval_sec=request_interval_sec,
        )
    )
