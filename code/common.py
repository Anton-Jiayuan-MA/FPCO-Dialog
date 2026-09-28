"""Shared I/O, credentials, and bounded retry helpers."""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, TypeVar


T = TypeVar("T")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
BASE_DIR = Path(os.getenv("FPCO_DIALOG_BASE_DIR") or Path(__file__).resolve().parent.parent).expanduser()


def dataset_paths(base_dir: Path) -> tuple[Path, Path]:
    """Prefer the published layout; support the original experiment layout too."""
    def resolve(published_name: str, legacy_name: str, env_name: str) -> Path:
        override = os.getenv(env_name)
        if override:
            return Path(override).expanduser()
        published = base_dir / "dataset" / published_name
        legacy = base_dir / legacy_name
        return legacy if not published.exists() and legacy.is_dir() else published

    return (
        resolve("image", "image", "FPCO_DIALOG_IMAGE_DIR"),
        resolve("metadata_and_questions", "image_metadata", "FPCO_DIALOG_METADATA_DIR"),
    )


DEFAULT_IMAGE_DIR, DEFAULT_METADATA_DIR = dataset_paths(BASE_DIR)
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_RESULT_DIR = BASE_DIR / "result"


class InvalidModelResponse(ValueError):
    """A successful request returned unusable model output."""


def require_api_key(*env_names: str) -> str:
    for name in env_names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    raise ValueError(f"Set {' or '.join(env_names)} in the environment before running this script.")


def get_gpt_api_key(api_provider: str) -> str:
    if api_provider == "openai":
        return require_api_key("OPENAI_API_KEY")
    if api_provider == "openrouter":
        return require_api_key("OPENROUTER_API_KEY")
    raise ValueError(f"Unsupported API provider: {api_provider}")


def build_gpt_client(api_provider: str, *, sdk_retries: int = 2) -> Any:
    from openai import OpenAI

    api_key = get_gpt_api_key(api_provider)
    kwargs = {"api_key": api_key, "max_retries": sdk_retries}
    if api_provider == "openrouter":
        kwargs["base_url"] = OPENROUTER_BASE_URL
    return OpenAI(**kwargs)


def resolve_model_name(model_name: str, api_provider: str) -> str:
    if api_provider == "openrouter" and "/" not in model_name:
        return f"openai/{model_name}"
    return model_name


def gemini_generate_content(
    api_key: str, model_name: str, payload: dict[str, Any], timeout_seconds: float,
) -> dict[str, Any]:
    """Send a verified HTTPS request; SSL_CERT_FILE/SSL_CERT_DIR configure trust."""
    url = f"{GEMINI_API_BASE}/{urllib.parse.quote(model_name, safe='')}:generateContent"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    # Keep original transport exceptions so retry logic can inspect status codes.
    with urllib.request.urlopen(
        request, timeout=timeout_seconds, context=ssl.create_default_context(),
    ) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise InvalidModelResponse("Gemini API response must be a JSON object.")
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def iter_metadata_files(metadata_dir: Path, limit: int | None, only_ids: set[str] | None) -> list[Path]:
    files = sorted(path for path in metadata_dir.iterdir() if path.is_file() and path.suffix == ".json")
    if only_ids:
        files = [path for path in files if path.stem in only_ids]
    return files if limit is None else files[:limit]


def iter_output_subdirs(output_dir: Path, selected_subdirs: list[str] | None) -> list[Path]:
    if selected_subdirs:
        return [output_dir / subdir for subdir in selected_subdirs]
    return sorted(path for path in output_dir.iterdir() if path.is_dir())


def is_retryable_api_error(exc: Exception, connection_errors: tuple[type[Exception], ...] = ()) -> bool:
    """Retry transient requests and malformed responses, not auth or input errors."""
    chain = []
    current = exc
    while isinstance(current, Exception) and current not in chain:
        reason = current.reason if isinstance(current, urllib.error.URLError) else current
        if isinstance(reason, ssl.SSLError):
            return False
        chain.append(current)
        current = current.__cause__
    for error in chain:
        status = error.code if isinstance(error, urllib.error.HTTPError) else getattr(error, "status_code", None)
        if isinstance(status, int):
            return status in {408, 409, 429} or 500 <= status < 600
        if isinstance(error, (InvalidModelResponse, json.JSONDecodeError, urllib.error.URLError,
                              TimeoutError, ConnectionError, *connection_errors)):
            return True
    return False


def retry_call(
    operation: Callable[[], T],
    *,
    attempts: int,
    wait_seconds: float,
    context: str,
    retryable: Callable[[Exception], bool],
) -> T:
    if attempts < 1:
        raise ValueError("--retry-attempts must be >= 1")
    if wait_seconds < 0:
        raise ValueError("--retry-wait-seconds must be >= 0")
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            # All other errors propagate immediately; interrupts are never caught.
            if attempt == attempts or not retryable(exc):
                raise
            delay = wait_seconds * attempt
            print(
                f"[retry] {context} attempt={attempt}/{attempts} "
                f"error={type(exc).__name__}; sleeping {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
