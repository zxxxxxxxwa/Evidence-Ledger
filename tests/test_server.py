from __future__ import annotations

import base64
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.config import Settings
from app.llm_client import LLMClient
from app.server import EvidenceLedgerServer, Handler


class ServerEngineeringSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        base = Settings.from_env()
        self.workspace_id = str(uuid.uuid4())
        settings = replace(
            base,
            host="127.0.0.1",
            port=0,
            db_path=Path(self.temporary.name) / "smoke.sqlite3",
            llm_api_key=None,
            reviewer_auth_header="X-Reviewer-Key",
            reviewer_auth_value="engineering-reviewer-only",
        )
        self.server = EvidenceLedgerServer((settings.host, settings.port), Handler, settings)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temporary.cleanup()

    def request(
        self,
        path: str,
        method: str = "GET",
        payload: dict | None = None,
        *,
        headers: dict[str, str] | None = None,
        base_url: str | None = None,
        workspace_id: str | None = None,
    ) -> tuple[int, dict | str]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request_headers = {"X-Workspace-ID": workspace_id or self.workspace_id}
        request_headers.update(headers or {})
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            (base_url or self.base_url) + path,
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            response = urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read()
            decoded = json.loads(data) if "application/json" in content_type else data.decode("utf-8")
            return response.status, decoded

    @staticmethod
    def assessment_metadata() -> dict[str, str]:
        return {
            "language": "en",
            "model": "test-fixture-only",
            "prompt_version": "engineering",
            "framework_version": "engineering",
            "resource_version": "engineering",
        }

    @staticmethod
    def valid_profile() -> dict:
        return {
            "display_name": "Fixture fixture",
            "discipline": "Engineering test",
            "career_stage": "early",
            "role": "fictional teacher",
            "goals": ["Check the API chain"],
            "weekly_minutes": 60,
            "constraints": "not research data",
            "developmental_use_acknowledged": True,
        }

    @staticmethod
    def material_payload(**overrides: object) -> dict:
        fixture = (
            b"Fixture engineering fixture only. This checks document parsing and API storage; "
            b"it is not teacher data."
        )
        payload = {
            "filename": "test-fixture.txt",
            "title": "Fixture engineering input",
            "material_type": "other",
            "source_date": "2026-11-18",
            "contribution_level": "sole",
            "contribution_note": "fictional",
            "source_ref": "engineering-only",
            "privacy_confirmed": True,
            "content_base64": base64.b64encode(fixture).decode("ascii"),
        }
        payload.update(overrides)
        return payload

    def test_health_static_bootstrap_and_material_wiring(self) -> None:
        status, health = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertFalse(health["llm_configured"])
        status, html = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("Evidence Ledger", html)
        status, bootstrap = self.request("/api/bootstrap")
        self.assertEqual(status, 200)
        self.assertEqual(bootstrap["materials"], [])
        self.assertEqual(bootstrap["system"]["max_file_bytes"], self.server.settings.max_file_bytes)
        self.assertEqual(bootstrap["system"]["max_input_chars"], self.server.settings.max_input_chars)

        profile = self.valid_profile()
        status, _ = self.request("/api/profile", "POST", profile)
        self.assertEqual(status, 200)
        material = self.material_payload()
        status, uploaded = self.request("/api/materials", "POST", material)
        self.assertEqual(status, 201)
        self.assertGreater(uploaded["material"]["parse_meta"]["characters"], 20)
        status, bootstrap = self.request("/api/bootstrap")
        self.assertEqual(len(bootstrap["materials"]), 1)

    def test_strict_profile_privacy_and_date_validation(self) -> None:
        for field, value, code in (
            ("developmental_use_acknowledged", "true", "DEVELOPMENTAL_USE_ACK_INVALID"),
            ("career_stage", "unknown-stage", "CAREER_STAGE_INVALID"),
            ("discipline", "   ", "DISCIPLINE_INVALID"),
            ("role", "\n", "ROLE_INVALID"),
        ):
            with self.subTest(field=field):
                profile = self.valid_profile()
                profile[field] = value
                status, payload = self.request("/api/profile", "POST", profile)
                self.assertEqual(status, 422)
                self.assertEqual(payload["error"]["code"], code)

        status, payload = self.request(
            "/api/materials", "POST", self.material_payload(privacy_confirmed="true")
        )
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "PRIVACY_CONFIRMATION_INVALID")

        status, payload = self.request(
            "/api/materials", "POST", self.material_payload(source_date="2026-02-31")
        )
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "SOURCE_DATE_INVALID")

        status, payload = self.request(
            "/api/materials", "POST", self.material_payload(source_date="2999-01-01")
        )
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "SOURCE_DATE_INVALID")

    def test_material_deletion_removes_derived_assessments_and_reviews(self) -> None:
        status, uploaded = self.request("/api/materials", "POST", self.material_payload())
        self.assertEqual(status, 201)
        material_id = uploaded["material"]["id"]
        assessment_id = self.server.database.create_assessment(
            self.workspace_id, self.assessment_metadata()
        )
        self.assertIsNotNone(assessment_id)
        self.server.database.complete_assessment(assessment_id, self.workspace_id, {"indicators": []})
        self.server.database.add_review(
            self.workspace_id, assessment_id, "T_ALIGN", {"reason_code": "fixture"}
        )

        status, payload = self.request(f"/api/materials/{material_id}", "DELETE")
        self.assertEqual(status, 200)
        self.assertTrue(payload["deleted"])
        self.assertIsNone(self.server.database.latest_assessment(self.workspace_id))
        self.assertEqual(self.server.database.list_reviews(self.workspace_id), [])
        with self.server.database.connect() as connection:
            self.assertEqual(connection.execute("PRAGMA secure_delete").fetchone()[0], 1)

    def test_only_one_running_assessment_per_workspace(self) -> None:
        first = self.server.database.create_assessment(
            self.workspace_id, self.assessment_metadata()
        )
        second = self.server.database.create_assessment(
            self.workspace_id, self.assessment_metadata()
        )
        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_review_resolution_requires_reviewer_and_stales_assessment(self) -> None:
        assessment_id = self.server.database.create_assessment(
            self.workspace_id, self.assessment_metadata()
        )
        self.assertIsNotNone(assessment_id)
        self.server.database.complete_assessment(assessment_id, self.workspace_id, {"indicators": []})
        review_id = self.server.database.add_review(
            self.workspace_id, assessment_id, "A_RESP", {"reason_code": "fixture"}
        )
        endpoint = f"/api/reviews/{review_id}/resolve"

        status, payload = self.request(endpoint, "POST", {"outcome": "clarified"})
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"]["code"], "REVIEWER_AUTH_REQUIRED")

        status, _ = self.request(
            endpoint,
            "POST",
            {"outcome": "clarified", "note": "engineering state check"},
            headers={"X-Reviewer-Key": "engineering-reviewer-only"},
        )
        self.assertEqual(status, 200)
        assessment = self.server.database.latest_assessment(self.workspace_id)
        self.assertEqual(assessment["status"], "stale")
        review = self.server.database.list_reviews(self.workspace_id)[0]
        self.assertTrue(review["resolution"]["requires_reassessment"])

    def test_production_workspace_is_bound_to_proxy_subject(self) -> None:
        production_settings = replace(
            self.server.settings,
            environment="production",
            port=0,
            db_path=Path(self.temporary.name) / "production.sqlite3",
            proxy_auth_header="X-Authenticated-User",
            allowed_origin="https://teachers.example.edu",
        )
        production = EvidenceLedgerServer(
            (production_settings.host, production_settings.port), Handler, production_settings
        )
        thread = threading.Thread(target=production.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{production.server_address[1]}"
        try:
            status, _ = self.request(
                "/api/bootstrap",
                headers={"X-Authenticated-User": "teacher-a@example.edu"},
                base_url=base_url,
            )
            self.assertEqual(status, 200)
            self.assertEqual(
                production.database.workspace_owner(self.workspace_id),
                "teacher-a@example.edu",
            )
            status, payload = self.request(
                "/api/bootstrap",
                headers={"X-Authenticated-User": "teacher-b@example.edu"},
                base_url=base_url,
            )
            self.assertEqual(status, 403)
            self.assertEqual(payload["error"]["code"], "WORKSPACE_SUBJECT_MISMATCH")
        finally:
            production.shutdown()
            production.server_close()
            thread.join(timeout=3)

    def test_llm_client_uses_process_wide_configured_concurrency_gate(self) -> None:
        settings = replace(
            self.server.settings,
            llm_api_key="test-fixture-key",
            llm_max_concurrency=2,
            llm_timeout=2,
        )
        client = LLMClient(settings)
        lock = threading.Lock()
        active = 0
        maximum = 0
        errors: list[Exception] = []

        def fake_chat_completion(**_: object) -> str:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.04)
            with lock:
                active -= 1
            return "{}"

        def invoke() -> None:
            try:
                client.complete_json(
                    [{"role": "user", "content": "fixture engineering call"}],
                    purpose="engineering_concurrency_check",
                )
            except Exception as exc:
                errors.append(exc)

        with patch("app.llm_client.chat_completion", side_effect=fake_chat_completion):
            threads = [threading.Thread(target=invoke) for _ in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)

        self.assertEqual(errors, [])
        self.assertLessEqual(maximum, 2)


if __name__ == "__main__":
    unittest.main()
