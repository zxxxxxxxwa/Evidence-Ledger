from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA secure_delete = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    owner_subject TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS profiles (
                    workspace_id TEXT PRIMARY KEY REFERENCES workspaces(id) ON DELETE CASCADE,
                    data_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS materials (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    material_type TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    source_date TEXT,
                    contribution_level TEXT NOT NULL,
                    contribution_note TEXT,
                    source_ref TEXT,
                    sha256 TEXT NOT NULL,
                    text_content TEXT NOT NULL,
                    chunks_json TEXT NOT NULL,
                    safety_flags_json TEXT NOT NULL,
                    parse_meta_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS materials_workspace_idx ON materials(workspace_id);
                CREATE INDEX IF NOT EXISTS materials_hash_idx ON materials(workspace_id, sha256);

                CREATE TABLE IF NOT EXISTS assessments (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    language TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    framework_version TEXT NOT NULL,
                    resource_version TEXT NOT NULL,
                    result_json TEXT,
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS assessments_workspace_idx ON assessments(workspace_id, created_at);

                CREATE TABLE IF NOT EXISTS reviews (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    assessment_id TEXT NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
                    indicator_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason_json TEXT NOT NULL,
                    resolution_json TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                );
                CREATE INDEX IF NOT EXISTS reviews_workspace_idx ON reviews(workspace_id, created_at);

                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    action TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )


            workspace_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(workspaces)").fetchall()
            }
            if "owner_subject" not in workspace_columns:
                db.execute("ALTER TABLE workspaces ADD COLUMN owner_subject TEXT")



            db.execute(
                """UPDATE assessments
                   SET status='failed', error_code='INTERRUPTED_ON_RESTART', completed_at=?
                   WHERE status='running'""",
                (utc_now(),),
            )
            db.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS assessments_one_running_per_workspace
                   ON assessments(workspace_id) WHERE status='running'"""
            )

    @staticmethod
    def _dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _loads(value: str | None, fallback: Any = None) -> Any:
        if not value:
            return fallback
        return json.loads(value)

    def ensure_workspace(self, workspace_id: str) -> None:
        now = utc_now()
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO workspaces(id, created_at, updated_at) VALUES (?, ?, ?)",
                (workspace_id, now, now),
            )
            db.execute("UPDATE workspaces SET updated_at = ? WHERE id = ?", (now, workspace_id))

    def bind_workspace_subject(self, workspace_id: str, subject: str) -> bool:

        now = utc_now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT owner_subject FROM workspaces WHERE id=?", (workspace_id,)
            ).fetchone()
            if row is None:
                db.execute(
                    """INSERT INTO workspaces(id, owner_subject, created_at, updated_at)
                       VALUES (?, ?, ?, ?)""",
                    (workspace_id, subject, now, now),
                )
                return True
            owner = row["owner_subject"]
            if owner == subject:
                db.execute(
                    "UPDATE workspaces SET updated_at=? WHERE id=?", (now, workspace_id)
                )
                return True
            if owner is not None:
                return False
            has_data = any(
                db.execute(
                    f"SELECT 1 FROM {table} WHERE workspace_id=? LIMIT 1", (workspace_id,)
                ).fetchone()
                is not None
                for table in ("profiles", "materials", "assessments", "reviews")
            )
            if has_data:
                return False
            db.execute(
                "UPDATE workspaces SET owner_subject=?, updated_at=? WHERE id=?",
                (subject, now, workspace_id),
            )
            return True

    def workspace_owner(self, workspace_id: str) -> str | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT owner_subject FROM workspaces WHERE id=?", (workspace_id,)
            ).fetchone()
        return row["owner_subject"] if row else None

    def save_profile(self, workspace_id: str, profile: dict[str, Any]) -> None:
        self.ensure_workspace(workspace_id)
        now = utc_now()
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO profiles(workspace_id, data_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(workspace_id) DO UPDATE SET data_json=excluded.data_json, updated_at=excluded.updated_at
                """,
                (workspace_id, self._dumps(profile), now),
            )
            db.execute(
                "UPDATE assessments SET status='stale' WHERE workspace_id=? AND status='completed'",
                (workspace_id,),
            )
        self.audit(workspace_id, "profile.saved", "profile", workspace_id, {"fields": sorted(profile)})

    def get_profile(self, workspace_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT data_json, updated_at FROM profiles WHERE workspace_id = ?", (workspace_id,)
            ).fetchone()
        if not row:
            return None
        profile = self._loads(row["data_json"], {})
        profile["updated_at"] = row["updated_at"]
        return profile

    def add_material(self, workspace_id: str, material: dict[str, Any]) -> str:
        self.ensure_workspace(workspace_id)
        material_id = material.get("id") or str(uuid.uuid4())
        now = utc_now()
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO materials(
                    id, workspace_id, title, filename, material_type, mime_type, source_date,
                    contribution_level, contribution_note, source_ref, sha256, text_content,
                    chunks_json, safety_flags_json, parse_meta_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    material_id,
                    workspace_id,
                    material["title"],
                    material["filename"],
                    material["material_type"],
                    material["mime_type"],
                    material.get("source_date"),
                    material["contribution_level"],
                    material.get("contribution_note"),
                    material.get("source_ref"),
                    material["sha256"],
                    material["text_content"],
                    self._dumps(material["chunks"]),
                    self._dumps(material.get("safety_flags", [])),
                    self._dumps(material.get("parse_meta", {})),
                    now,
                ),
            )
            db.execute(
                "UPDATE assessments SET status='stale' WHERE workspace_id=? AND status='completed'",
                (workspace_id,),
            )
        self.audit(
            workspace_id,
            "material.added",
            "material",
            material_id,
            {"type": material["material_type"], "bytes": material.get("bytes", 0)},
        )
        return material_id

    def list_materials(self, workspace_id: str, include_text: bool = False) -> list[dict[str, Any]]:
        columns = "*" if include_text else "id, title, filename, material_type, mime_type, source_date, contribution_level, contribution_note, source_ref, sha256, chunks_json, safety_flags_json, parse_meta_json, created_at"
        with self.connect() as db:
            rows = db.execute(
                f"SELECT {columns} FROM materials WHERE workspace_id = ? ORDER BY created_at DESC",
                (workspace_id,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["chunks"] = self._loads(item.pop("chunks_json"), [])
            item["safety_flags"] = self._loads(item.pop("safety_flags_json"), [])
            item["parse_meta"] = self._loads(item.pop("parse_meta_json"), {})
            if not include_text:
                item["chunk_count"] = len(item.pop("chunks"))
            result.append(item)
        return result

    def delete_material(self, workspace_id: str, material_id: str) -> bool:
        with self.connect() as db:
            cursor = db.execute(
                "DELETE FROM materials WHERE workspace_id = ? AND id = ?", (workspace_id, material_id)
            )
            if cursor.rowcount > 0:




                db.execute("DELETE FROM assessments WHERE workspace_id=?", (workspace_id,))
        deleted = cursor.rowcount > 0
        if deleted:
            self.audit(workspace_id, "material.deleted", "material", material_id, {})
        return deleted

    def create_assessment(self, workspace_id: str, metadata: dict[str, str]) -> str | None:
        self.ensure_workspace(workspace_id)
        assessment_id = str(uuid.uuid4())
        try:
            with self.connect() as db:
                db.execute(
                    """
                    INSERT INTO assessments(id, workspace_id, status, language, model, prompt_version,
                        framework_version, resource_version, created_at)
                    VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        assessment_id,
                        workspace_id,
                        metadata["language"],
                        metadata["model"],
                        metadata["prompt_version"],
                        metadata["framework_version"],
                        metadata["resource_version"],
                        utc_now(),
                    ),
                )
        except sqlite3.IntegrityError as exc:

            if "assessments.workspace_id" in str(exc):
                return None
            raise
        self.audit(workspace_id, "assessment.started", "assessment", assessment_id, metadata)
        return assessment_id

    def complete_assessment(self, assessment_id: str, workspace_id: str, result: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                """UPDATE assessments SET status='completed', result_json=?, completed_at=?
                   WHERE id=? AND workspace_id=?""",
                (self._dumps(result), utc_now(), assessment_id, workspace_id),
            )
        self.audit(workspace_id, "assessment.completed", "assessment", assessment_id, {
            "indicator_count": len(result.get("indicators", [])),
            "review_count": len(result.get("review_ids", [])),
        })

    def fail_assessment(self, assessment_id: str, workspace_id: str, error_code: str) -> None:
        with self.connect() as db:
            db.execute(
                """UPDATE assessments SET status='failed', error_code=?, completed_at=?
                   WHERE id=? AND workspace_id=?""",
                (error_code, utc_now(), assessment_id, workspace_id),
            )
        self.audit(workspace_id, "assessment.failed", "assessment", assessment_id, {"code": error_code})

    def latest_assessment(self, workspace_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM assessments WHERE workspace_id=? ORDER BY created_at DESC LIMIT 1",
                (workspace_id,),
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["result"] = self._loads(item.pop("result_json"), None)
        return item

    def has_running_assessment(self, workspace_id: str) -> bool:
        with self.connect() as db:
            row = db.execute(
                "SELECT 1 FROM assessments WHERE workspace_id=? AND status='running' LIMIT 1",
                (workspace_id,),
            ).fetchone()
        return row is not None

    def add_review(
        self, workspace_id: str, assessment_id: str, indicator_id: str, reason: dict[str, Any]
    ) -> str:
        review_id = str(uuid.uuid4())
        with self.connect() as db:
            db.execute(
                """INSERT INTO reviews(id, workspace_id, assessment_id, indicator_id, status,
                   reason_json, created_at) VALUES (?, ?, ?, ?, 'open', ?, ?)""",
                (review_id, workspace_id, assessment_id, indicator_id, self._dumps(reason), utc_now()),
            )
        self.audit(workspace_id, "review.opened", "review", review_id, {"indicator_id": indicator_id})
        return review_id

    def list_reviews(self, workspace_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM reviews WHERE workspace_id=? ORDER BY created_at DESC", (workspace_id,)
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["reason"] = self._loads(item.pop("reason_json"), {})
            item["resolution"] = self._loads(item.pop("resolution_json"), None)
            result.append(item)
        return result

    def resolve_review(self, workspace_id: str, review_id: str, resolution: dict[str, Any]) -> bool:
        with self.connect() as db:
            review = db.execute(
                """SELECT assessment_id FROM reviews
                   WHERE workspace_id=? AND id=? AND status='open'""",
                (workspace_id, review_id),
            ).fetchone()
            if not review:
                return False
            cursor = db.execute(
                """UPDATE reviews SET status='resolved', resolution_json=?, resolved_at=?
                   WHERE workspace_id=? AND id=? AND status='open'""",
                (self._dumps(resolution), utc_now(), workspace_id, review_id),
            )
            if cursor.rowcount > 0:
                db.execute(
                    """UPDATE assessments SET status='stale'
                       WHERE workspace_id=? AND id=? AND status='completed'""",
                    (workspace_id, review["assessment_id"]),
                )
        updated = cursor.rowcount > 0
        if updated:
            self.audit(workspace_id, "review.resolved", "review", review_id, {"outcome": resolution.get("outcome")})
        return updated

    def audit(
        self,
        workspace_id: str,
        action: str,
        entity_type: str,
        entity_id: str | None,
        metadata: dict[str, Any],
    ) -> None:

        with self.connect() as db:
            db.execute(
                "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    workspace_id,
                    action,
                    entity_type,
                    entity_id,
                    self._dumps(metadata),
                    utc_now(),
                ),
            )

    def delete_workspace(self, workspace_id: str) -> bool:
        with self.connect() as db:
            cursor = db.execute("DELETE FROM workspaces WHERE id=?", (workspace_id,))
        return cursor.rowcount > 0
