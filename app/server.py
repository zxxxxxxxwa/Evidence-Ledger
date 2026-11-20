from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import json
import mimetypes
import os
import re
import sys
import threading
import time
import uuid
from datetime import UTC, date, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import Settings
from .db import Database
from .document_parser import chunk_text, parse_document
from .errors import AppError
from .pipeline import AssessmentPipeline, load_json
from .retrieval import ResourceLibrary
from .safety import scan_text


MATERIAL_TYPES = {"paper", "research_project", "slides", "lesson_plan", "teaching_achievement", "other"}
CONTRIBUTION_LEVELS = {"sole", "lead", "substantial", "limited", "unclear"}
CAREER_STAGES = {"early", "mid", "senior", "other"}
DATE_PATTERN = re.compile(r"^\d{4}(?:-\d{2}(?:-\d{2})?)?$")
CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]")


class EvidenceLedgerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], settings: Settings):
        self.settings = settings
        self.database = Database(settings.db_path)
        self.pipeline = AssessmentPipeline(self.database, settings)
        self.framework = load_json(settings.framework_path)
        self.library = ResourceLibrary(settings.resources_path)
        self.rate_lock = threading.Lock()
        self.assessment_calls: dict[str, list[float]] = {}
        super().__init__(address, handler)

    def allow_assessment(self, workspace_id: str) -> bool:
        now = time.monotonic()
        with self.rate_lock:
            recent = [item for item in self.assessment_calls.get(workspace_id, []) if now - item < 60]
            if len(recent) >= 3:
                self.assessment_calls[workspace_id] = recent
                return False
            recent.append(now)
            self.assessment_calls[workspace_id] = recent
            return True


class Handler(BaseHTTPRequestHandler):
    server: EvidenceLedgerServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:

        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'",
        )
        self.send_header("Cache-Control", "no-store")

    def _trusted_subject(self) -> str | None:
        if self.server.settings.environment != "production":
            return None
        header = self.server.settings.proxy_auth_header
        value = self.headers.get(header, "").strip() if header else ""
        if not value or len(value) > 300 or CONTROL_PATTERN.search(value):
            raise AppError("AUTHENTICATION_REQUIRED", 401)
        return value

    def _authenticate(self) -> None:
        self._trusted_subject()

    def _authorize_reviewer(self) -> None:
        header = self.server.settings.reviewer_auth_header
        expected = self.server.settings.reviewer_auth_value





        if self.server.settings.environment != "production" and not header and not expected:
            return
        if not header or not expected:
            raise AppError("REVIEWER_AUTH_NOT_CONFIGURED", 503)
        supplied = self.headers.get(header, "")
        if not hmac.compare_digest(supplied, expected):
            raise AppError("REVIEWER_AUTH_REQUIRED", 403)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        static_root = self.server.settings.static_dir.resolve()
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            self._send_json(404, {"error": {"code": "NOT_FOUND"}})
            return
        if static_root not in resolved.parents and resolved != static_root:
            self._send_json(404, {"error": {"code": "NOT_FOUND"}})
            return
        body = resolved.read_bytes()
        content_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        if resolved.suffix == ".js":
            content_type = "text/javascript"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith(("text/", "application/json")) else content_type)
        self.send_header("Content-Length", str(len(body)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self, max_bytes: int | None = None) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise AppError("CONTENT_LENGTH_INVALID", 400) from exc
        limit = max_bytes or self.server.settings.max_file_bytes * 2
        if length <= 0:
            raise AppError("JSON_BODY_REQUIRED", 400)
        if length > limit:
            raise AppError("REQUEST_TOO_LARGE", 413)
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AppError("JSON_INVALID", 400) from exc
        if not isinstance(payload, dict):
            raise AppError("JSON_OBJECT_REQUIRED", 400)
        return payload

    def _workspace_id(self) -> str:
        value = self.headers.get("X-Workspace-ID", "")
        try:
            workspace_id = str(uuid.UUID(value))
        except (ValueError, AttributeError) as exc:
            raise AppError("WORKSPACE_ID_INVALID", 400) from exc
        subject = self._trusted_subject()
        if subject is not None and not self.server.database.bind_workspace_subject(
            workspace_id, subject
        ):
            raise AppError("WORKSPACE_SUBJECT_MISMATCH", 403)
        return workspace_id

    def _require_same_origin(self) -> None:
        origin = self.headers.get("Origin")
        if not origin:
            return
        configured = self.server.settings.allowed_origin
        if configured:
            if origin != configured:
                raise AppError("ORIGIN_REJECTED", 403)
            return
        expected_http = f"http://{self.headers.get('Host')}"
        expected_https = f"https://{self.headers.get('Host')}"
        if origin not in {expected_http, expected_https}:
            raise AppError("ORIGIN_REJECTED", 403)

    def _require_mutable_workspace(self, workspace_id: str) -> None:
        if self.server.database.has_running_assessment(workspace_id):
            raise AppError("ASSESSMENT_ALREADY_RUNNING", 409)

    @staticmethod
    def _clean(value: Any, max_length: int = 500) -> str:
        return str(value or "").strip()[:max_length]

    @staticmethod
    def _required_profile_text(
        payload: dict[str, Any], field: str, max_length: int, error_code: str
    ) -> str:
        value = payload.get(field)
        if not isinstance(value, str):
            raise AppError(error_code, 422)
        cleaned = value.strip()
        if (
            not cleaned
            or len(cleaned) > max_length
            or CONTROL_PATTERN.search(cleaned)
            or not re.search(r"\w", cleaned, re.UNICODE)
        ):
            raise AppError(error_code, 422)
        return cleaned

    @staticmethod
    def _source_date(value: Any) -> str | None:
        if value is None or value == "":
            return None
        if not isinstance(value, str) or not DATE_PATTERN.fullmatch(value):
            raise AppError("SOURCE_DATE_INVALID", 422)
        today = datetime.now(UTC).date()
        try:
            if len(value) == 4:
                parsed_year = int(value)
                if not 1900 <= parsed_year <= today.year:
                    raise ValueError
            elif len(value) == 7:
                parsed_year, parsed_month = (int(part) for part in value.split("-"))
                date(parsed_year, parsed_month, 1)
                if parsed_year < 1900 or (parsed_year, parsed_month) > (today.year, today.month):
                    raise ValueError
            else:
                parsed = date.fromisoformat(value)
                if parsed.year < 1900 or parsed > today:
                    raise ValueError
        except (TypeError, ValueError):
            raise AppError("SOURCE_DATE_INVALID", 422) from None
        return value

    def _profile_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        goals_raw = payload.get("goals")
        if not isinstance(goals_raw, list):
            raise AppError("GOALS_INVALID", 422)
        goals = [self._clean(item, 300) for item in goals_raw if self._clean(item, 300)]
        if not goals:
            raise AppError("GOAL_REQUIRED", 422)
        try:
            weekly_minutes = int(payload.get("weekly_minutes", 60))
        except (TypeError, ValueError) as exc:
            raise AppError("WEEKLY_MINUTES_INVALID", 422) from exc
        if not 15 <= weekly_minutes <= 600:
            raise AppError("WEEKLY_MINUTES_INVALID", 422)
        career_stage = payload.get("career_stage")
        if not isinstance(career_stage, str) or career_stage not in CAREER_STAGES:
            raise AppError("CAREER_STAGE_INVALID", 422)
        acknowledgement = payload.get("developmental_use_acknowledged")
        if not isinstance(acknowledgement, bool):
            raise AppError("DEVELOPMENTAL_USE_ACK_INVALID", 422)
        return {
            "display_name": self._clean(payload.get("display_name"), 80),
            "discipline": self._required_profile_text(
                payload, "discipline", 120, "DISCIPLINE_INVALID"
            ),
            "career_stage": career_stage,
            "role": self._required_profile_text(payload, "role", 120, "ROLE_INVALID"),
            "goals": goals[:5],
            "weekly_minutes": weekly_minutes,
            "constraints": self._clean(payload.get("constraints"), 1000),
            "developmental_use_acknowledged": acknowledgement,
        }

    def do_GET(self) -> None:
        try:
            self._authenticate()
            parsed = urlparse(self.path)
            if parsed.path == "/api/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "version": "0.1.0",
                        "llm_configured": self.server.settings.llm_configured,
                        "calibration": self.server.pipeline.ratability.metadata(),
                    },
                )
                return
            if parsed.path == "/api/bootstrap":
                workspace_id = self._workspace_id()
                self.server.database.ensure_workspace(workspace_id)
                self._send_json(
                    200,
                    {
                        "profile": self.server.database.get_profile(workspace_id),
                        "materials": self.server.database.list_materials(workspace_id),
                        "assessment": self.server.database.latest_assessment(workspace_id),
                        "reviews": self.server.database.list_reviews(workspace_id),
                        "framework": {
                            "version": self.server.framework["version"],
                            "dimensions": self.server.framework["dimensions"],
                            "indicators": [
                                {
                                    key: item[key]
                                    for key in ("id", "dimension", "title_en", "title_zh", "description_en", "description_zh")
                                }
                                for item in self.server.framework["indicators"]
                            ],
                        },
                        "resources": self.server.library.public_catalog(),
                        "system": {
                            "llm_configured": self.server.settings.llm_configured,
                            "model": self.server.settings.llm_model,
                            "calibration": self.server.pipeline.ratability.metadata(),
                            "developmental_only": True,
                            "original_files_retained": False,
                            "max_file_bytes": self.server.settings.max_file_bytes,
                            "max_input_chars": self.server.settings.max_input_chars,
                        },
                    },
                )
                return
            if parsed.path == "/api/resources":
                query = parse_qs(parsed.query).get("q", [""])[0][:500]
                resources = self.server.library.search(query, limit=10) if query else self.server.library.resources
                self._send_json(200, {"version": self.server.library.version, "resources": resources})
                return
            if parsed.path in {"/", "/index.html"}:
                self._send_file(self.server.settings.static_dir / "index.html")
                return
            if parsed.path.startswith("/static/"):
                relative = parsed.path.removeprefix("/static/")
                self._send_file(self.server.settings.static_dir / relative)
                return
            self._send_json(404, {"error": {"code": "NOT_FOUND"}})
        except AppError as exc:
            self._send_json(exc.status, exc.as_dict())
        except Exception:
            self._send_json(500, {"error": {"code": "INTERNAL_ERROR"}})

    def do_POST(self) -> None:
        assessment_id: str | None = None
        workspace_id: str | None = None
        try:
            self._authenticate()
            self._require_same_origin()
            parsed = urlparse(self.path)
            workspace_id = self._workspace_id()
            if parsed.path == "/api/profile":
                payload = self._read_json()
                self._require_mutable_workspace(workspace_id)
                profile = self._profile_payload(payload)
                if not profile["developmental_use_acknowledged"]:
                    raise AppError("DEVELOPMENTAL_USE_ACK_REQUIRED", 422)
                self.server.database.save_profile(workspace_id, profile)
                self._send_json(200, {"profile": self.server.database.get_profile(workspace_id)})
                return

            if parsed.path == "/api/materials":
                payload = self._read_json(self.server.settings.max_file_bytes * 2)
                self._require_mutable_workspace(workspace_id)
                privacy_confirmed = payload.get("privacy_confirmed")
                if not isinstance(privacy_confirmed, bool):
                    raise AppError("PRIVACY_CONFIRMATION_INVALID", 422)
                if not privacy_confirmed:
                    raise AppError("PRIVACY_CONFIRMATION_REQUIRED", 422)
                filename = Path(self._clean(payload.get("filename"), 240)).name
                material_type = self._clean(payload.get("material_type"), 40)
                contribution = self._clean(payload.get("contribution_level"), 40)
                if material_type not in MATERIAL_TYPES:
                    raise AppError("MATERIAL_TYPE_INVALID", 422)
                if contribution not in CONTRIBUTION_LEVELS:
                    raise AppError("CONTRIBUTION_LEVEL_INVALID", 422)
                source_date = self._source_date(payload.get("source_date"))
                encoded = payload.get("content_base64")
                if not isinstance(encoded, str):
                    raise AppError("FILE_CONTENT_REQUIRED", 422)
                try:
                    binary = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise AppError("FILE_BASE64_INVALID", 400) from exc
                if len(binary) > self.server.settings.max_file_bytes:
                    raise AppError("FILE_TOO_LARGE", 413)
                text, mime_type, parse_meta = parse_document(filename, binary)
                material_id = str(uuid.uuid4())
                flags = scan_text(text)
                existing = self.server.database.list_materials(workspace_id)
                if any(item["sha256"] == parse_meta["sha256"] for item in existing):
                    flags.append({"code": "DUPLICATE_FILE", "severity": "warning"})
                chunks = chunk_text(material_id, text)
                self.server.database.add_material(
                    workspace_id,
                    {
                        "id": material_id,
                        "title": self._clean(payload.get("title"), 180) or filename,
                        "filename": filename,
                        "material_type": material_type,
                        "mime_type": mime_type,
                        "source_date": source_date,
                        "contribution_level": contribution,
                        "contribution_note": self._clean(payload.get("contribution_note"), 500),
                        "source_ref": self._clean(payload.get("source_ref"), 200),
                        "sha256": parse_meta["sha256"],
                        "text_content": text,
                        "chunks": chunks,
                        "safety_flags": flags,
                        "parse_meta": parse_meta,
                        "bytes": len(binary),
                    },
                )
                material = next(item for item in self.server.database.list_materials(workspace_id) if item["id"] == material_id)
                self._send_json(201, {"material": material})
                return

            if parsed.path == "/api/assessments":
                payload = self._read_json()
                language = payload.get("language", "en")
                if not self.server.settings.llm_configured:
                    raise AppError("LLM_NOT_CONFIGURED", 503)
                if not self.server.allow_assessment(workspace_id):
                    raise AppError("RATE_LIMITED", 429)
                metadata = self.server.pipeline.metadata(language)
                assessment_id = self.server.database.create_assessment(workspace_id, metadata)
                if assessment_id is None:
                    raise AppError("ASSESSMENT_ALREADY_RUNNING", 409)
                try:
                    result = self.server.pipeline.run(workspace_id, assessment_id, language)
                except AppError as exc:
                    self.server.database.fail_assessment(assessment_id, workspace_id, exc.code)
                    raise
                self.server.database.complete_assessment(assessment_id, workspace_id, result)
                self._send_json(201, {"assessment": self.server.database.latest_assessment(workspace_id)})
                return

            review_match = re.fullmatch(r"/api/reviews/([0-9a-f-]+)/resolve", parsed.path)
            if review_match:
                self._authorize_reviewer()
                payload = self._read_json()
                outcome = self._clean(payload.get("outcome"), 50)
                if outcome not in {"clarified", "insufficient", "confirmed_conflict", "withdrawn"}:
                    raise AppError("REVIEW_OUTCOME_INVALID", 422)
                updated = self.server.database.resolve_review(
                    workspace_id,
                    review_match.group(1),
                    {
                        "outcome": outcome,
                        "note": self._clean(payload.get("note"), 1000),
                        "requires_reassessment": True,
                    },
                )
                if not updated:
                    raise AppError("REVIEW_NOT_OPEN", 404)
                self._send_json(200, {"reviews": self.server.database.list_reviews(workspace_id)})
                return

            self._send_json(404, {"error": {"code": "NOT_FOUND"}})
        except AppError as exc:
            self._send_json(exc.status, exc.as_dict())
        except Exception:
            if assessment_id and workspace_id:
                self.server.database.fail_assessment(assessment_id, workspace_id, "INTERNAL_ERROR")
            self._send_json(500, {"error": {"code": "INTERNAL_ERROR"}})

    def do_DELETE(self) -> None:
        try:
            self._authenticate()
            self._require_same_origin()
            parsed = urlparse(self.path)
            workspace_id = self._workspace_id()
            material_match = re.fullmatch(r"/api/materials/([0-9a-f-]+)", parsed.path)
            if material_match:
                self._require_mutable_workspace(workspace_id)
                if not self.server.database.delete_material(workspace_id, material_match.group(1)):
                    raise AppError("MATERIAL_NOT_FOUND", 404)
                self._send_json(200, {"deleted": True})
                return
            if parsed.path == "/api/workspace":
                payload = self._read_json()
                self._require_mutable_workspace(workspace_id)
                if payload.get("confirmation") != "DELETE":
                    raise AppError("DELETE_CONFIRMATION_REQUIRED", 422)
                deleted = self.server.database.delete_workspace(workspace_id)
                self._send_json(200, {"deleted": deleted})
                return
            self._send_json(404, {"error": {"code": "NOT_FOUND"}})
        except AppError as exc:
            self._send_json(exc.status, exc.as_dict())
        except Exception:
            self._send_json(500, {"error": {"code": "INTERNAL_ERROR"}})


def create_server(settings: Settings | None = None) -> EvidenceLedgerServer:
    settings = settings or Settings.from_env()
    if settings.environment == "production":
        if settings.host not in {"127.0.0.1", "::1", "localhost"}:
            raise RuntimeError("Production mode must bind to loopback behind an authenticated TLS reverse proxy")
        if not settings.proxy_auth_header or not settings.allowed_origin:
            raise RuntimeError("Production mode requires EVIDENCE_LEDGER_PROXY_AUTH_HEADER and EVIDENCE_LEDGER_ALLOWED_ORIGIN")
        if not settings.reviewer_auth_header or not settings.reviewer_auth_value:
            raise RuntimeError(
                "Production mode requires EVIDENCE_LEDGER_REVIEWER_AUTH_HEADER and "
                "EVIDENCE_LEDGER_REVIEWER_AUTH_VALUE"
            )
    return EvidenceLedgerServer((settings.host, settings.port), Handler, settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Evidence Ledger web application.")
    parser.add_argument("--host", help="Override EVIDENCE_LEDGER_HOST")
    parser.add_argument("--port", type=int, help="Override EVIDENCE_LEDGER_PORT")
    args = parser.parse_args()
    if args.host:
        os.environ["EVIDENCE_LEDGER_HOST"] = args.host
    if args.port:
        os.environ["EVIDENCE_LEDGER_PORT"] = str(args.port)
    server = create_server()
    print(f"Evidence Ledger listening on http://{server.server_address[0]}:{server.server_address[1]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
