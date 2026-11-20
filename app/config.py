from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


def _bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    value = _int_env(name, default)
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True, slots=True)
class Settings:
    environment: str
    host: str
    port: int
    db_path: Path
    static_dir: Path
    framework_path: Path
    resources_path: Path
    calibration_path: Path | None
    max_file_bytes: int
    max_input_chars: int
    proxy_auth_header: str | None
    reviewer_auth_header: str | None
    reviewer_auth_value: str | None
    allowed_origin: str | None
    llm_api_key: str | None
    llm_base_url: str
    llm_model: str
    llm_timeout: int
    llm_temperature: float
    llm_max_tokens: int
    llm_max_concurrency: int

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)

    @classmethod
    def from_env(cls) -> "Settings":
        calibration_raw = os.getenv("EVIDENCE_LEDGER_CALIBRATION_FILE", "").strip()
        calibration = None
        if calibration_raw:
            candidate = Path(calibration_raw)
            calibration = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        return cls(
            environment=os.getenv("EVIDENCE_LEDGER_ENV", "development"),
            host=os.getenv("EVIDENCE_LEDGER_HOST", "127.0.0.1"),
            port=_int_env("EVIDENCE_LEDGER_PORT", 8080),
            db_path=_path_env(
                "EVIDENCE_LEDGER_DB", PROJECT_ROOT / "runtime" / "evidence_ledger.sqlite3"
            ),
            static_dir=PROJECT_ROOT / "app" / "static",
            framework_path=PROJECT_ROOT / "app" / "data" / "framework.json",
            resources_path=PROJECT_ROOT / "app" / "data" / "resources.json",
            calibration_path=calibration,
            max_file_bytes=_int_env("EVIDENCE_LEDGER_MAX_FILE_BYTES", 10 * 1024 * 1024),
            max_input_chars=_int_env("EVIDENCE_LEDGER_MAX_INPUT_CHARS", 180_000),
            proxy_auth_header=os.getenv("EVIDENCE_LEDGER_PROXY_AUTH_HEADER") or None,
            reviewer_auth_header=os.getenv("EVIDENCE_LEDGER_REVIEWER_AUTH_HEADER") or None,
            reviewer_auth_value=os.getenv("EVIDENCE_LEDGER_REVIEWER_AUTH_VALUE") or None,
            allowed_origin=os.getenv("EVIDENCE_LEDGER_ALLOWED_ORIGIN") or None,
            llm_api_key=os.getenv("EVIDENCE_LEDGER_LLM_API_KEY") or os.getenv("V_API_KEY"),
            llm_base_url=os.getenv("EVIDENCE_LEDGER_LLM_BASE_URL")
            or os.getenv("V_API_BASE_URL")
            or "https://run.v3.cm/v1",
            llm_model=os.getenv("EVIDENCE_LEDGER_LLM_MODEL", "qwen3.7-flash"),
            llm_timeout=_int_env("EVIDENCE_LEDGER_LLM_TIMEOUT", 120),
            llm_temperature=_float_env("EVIDENCE_LEDGER_LLM_TEMPERATURE", 0.1),
            llm_max_tokens=_int_env("EVIDENCE_LEDGER_LLM_MAX_TOKENS", 12_000),


            llm_max_concurrency=_bounded_int_env(
                "EVIDENCE_LEDGER_LLM_MAX_CONCURRENCY", 50, 1, 50
            ),
        )
