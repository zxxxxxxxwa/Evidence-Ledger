from __future__ import annotations

import hashlib
import re
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any


def _normalize_key(value: Any) -> str:
    if not value:
        return ""
    if not isinstance(value, (str, int, float)):
        return ""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value).casefold())


def _tokens(text: str) -> set[str]:
    english = re.findall(r"[a-z0-9]{2,}", text.casefold())
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    bigrams = ["".join(chinese[index : index + 2]) for index in range(max(0, len(chinese) - 1))]
    return set(english + bigrams)


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _time_metadata(value: Any) -> tuple[int | None, str]:
    match = re.search(r"(?:19|20)\d{2}", str(value or ""))
    if not match:
        return None, "unknown"
    year = int(match.group(0))
    current_year = datetime.now(UTC).year
    if year > current_year:
        return None, "invalid_future"
    age = current_year - year
    if age <= 2:
        return age, "current"
    if age <= 5:
        return age, "recent"
    return age, "historical"


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def validate_evidence(
    raw_items: list[dict[str, Any]],
    materials: list[dict[str, Any]],
    indicators: set[str] | dict[str, set[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    material_map = {item["id"]: item for item in materials}
    chunk_map: dict[str, dict[str, Any]] = {}
    for material in materials:
        for chunk in material.get("chunks", []):
            chunk_map[chunk["id"]] = chunk

    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    indicator_ids = set(indicators)
    for index, raw in enumerate(raw_items):
        material = material_map.get(str(raw.get("material_id", "")))
        chunk = chunk_map.get(str(raw.get("chunk_id", "")))
        quote = raw.get("quote")
        indicator_id = str(raw.get("indicator_id", ""))
        observable = raw.get("observable") if isinstance(raw.get("observable"), str) else ""
        reason = None
        if not material or not chunk or chunk.get("material_id") != material.get("id"):
            reason = "unknown_source"
        elif indicator_id not in indicator_ids:
            reason = "unknown_indicator"
        elif isinstance(indicators, dict) and observable not in indicators.get(indicator_id, set()):
            reason = "unknown_observable"
        elif not isinstance(quote, str) or len(quote.strip()) < 5 or quote not in chunk.get("text", ""):
            reason = "quote_not_verbatim"
        if reason:
            rejected.append({"index": index, "reason": reason})
            continue

        polarity = raw.get("polarity")
        if polarity not in {"support", "contradict", "context"}:
            polarity = "context"
        contribution = raw.get("contribution_clarity")
        if contribution not in {"clear", "partial", "unclear"}:
            contribution = "unclear"
        source_ref = _normalize_key(material.get("source_ref"))
        proposed_achievement_key = _normalize_key(raw.get("achievement_key"))
        family_hint = _normalize_key(raw.get("family_hint"))
        searchable_source = _normalize_key(
            " ".join([material.get("source_ref") or "", chunk.get("text", "")])
        )
        achievement_key = (
            proposed_achievement_key
            if proposed_achievement_key and proposed_achievement_key in searchable_source
            else ""
        )
        source_key = source_ref or f"material:{material['id']}"
        family_identity = achievement_key or source_ref or f"material:{material['id']}"
        security_restricted = any(
            item.get("severity") == "high" for item in material.get("safety_flags", [])
        )


        event_date = material.get("source_date")
        age_years, recency_band = _time_metadata(event_date)
        contribution_factor = {
            "sole": 1.0,
            "lead": 1.0,
            "substantial": 0.8,
            "limited": 0.4,
            "unclear": 0.0,
        }.get(material.get("contribution_level", "unclear"), 0.0)
        valid.append(
            {
                "id": f"ev-{uuid.uuid4().hex[:12]}",
                "indicator_id": indicator_id,
                "material_id": material["id"],
                "chunk_id": chunk["id"],
                "quote": quote.strip(),
                "locator": chunk.get("locator"),
                "observable": observable.strip(),
                "polarity": polarity,
                "claim": str(raw.get("claim", "")).strip(),
                "event_date": event_date,
                "age_years": age_years,
                "recency_band": recency_band,
                "achievement_key": achievement_key,
                "family_hint": family_hint,
                "source_key": source_key,
                "family_identity": family_identity,
                "material_sha256": material.get("sha256"),
                "contribution_clarity": contribution,
                "contribution_level": material.get("contribution_level", "unclear"),
                "contribution_note": material.get("contribution_note", ""),
                "contribution_factor": contribution_factor,
                "security_restricted": security_restricted,
            }
        )
    return valid, rejected


def build_evidence_families(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:

    finder = _UnionFind(len(evidence))
    merge_reasons: dict[tuple[int, int], str] = {}
    for left in range(len(evidence)):
        for right in range(left + 1, len(evidence)):
            a, b = evidence[left], evidence[right]
            reason = ""
            if a.get("family_identity") and a["family_identity"] == b.get("family_identity"):
                reason = "same_underlying_achievement"
            elif a.get("achievement_key") and a["achievement_key"] == b.get("achievement_key"):
                reason = "same_achievement_key"
            elif (
                a.get("material_sha256")
                and a["material_sha256"] == b.get("material_sha256")
                and a.get("material_id") != b.get("material_id")
                and a.get("indicator_id") == b.get("indicator_id")
                and a.get("observable") == b.get("observable")
                and a.get("quote") == b.get("quote")
            ):




                reason = "duplicate_located_claim"
            elif (
                a.get("indicator_id") == b.get("indicator_id")
                and len(a.get("quote", "")) >= 50
                and len(b.get("quote", "")) >= 50
                and _jaccard(a.get("quote", ""), b.get("quote", "")) >= 0.88
            ):
                reason = "near_duplicate_quote"
            if reason:
                finder.union(left, right)
                merge_reasons[(left, right)] = reason

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(evidence)):
        grouped[finder.find(index)].append(index)

    families: list[dict[str, Any]] = []
    for member_indices in grouped.values():
        members = [evidence[index] for index in member_indices]
        canonical = "|".join(sorted({member.get("family_identity", "") for member in members}))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
        reasons = sorted(
            {
                reason
                for (left, right), reason in merge_reasons.items()
                if left in member_indices and right in member_indices
            }
        ) or ["single_evidence"]
        families.append(
            {
                "id": f"fam-{digest}",
                "canonical_identity": canonical,
                "indicator_ids": sorted({member["indicator_id"] for member in members}),
                "member_ids": [member["id"] for member in members],
                "members": members,
                "independent_source_keys": sorted({member["source_key"] for member in members}),
                "event_dates": sorted({str(member["event_date"]) for member in members if member.get("event_date")}),
                "recency_bands": sorted({member["recency_band"] for member in members}),
                "contribution_levels": sorted({member["contribution_level"] for member in members}),
                "contribution_notes": sorted({member["contribution_note"] for member in members if member.get("contribution_note")}),
                "max_contribution_factor": max(member["contribution_factor"] for member in members),
                "all_security_restricted": all(member["security_restricted"] for member in members),
                "merge_reasons": reasons,
                "family_hints": sorted({member["family_hint"] for member in members if member.get("family_hint")}),
                "requires_review": False,
                "candidate_related_family_ids": [],
                "member_count": len(members),
            }
        )
    families = sorted(families, key=lambda item: item["id"])
    for left_index, left in enumerate(families):
        for right in families[left_index + 1 :]:
            if set(left["family_hints"]) & set(right["family_hints"]):
                left["requires_review"] = True
                right["requires_review"] = True
                left["candidate_related_family_ids"].append(right["id"])
                right["candidate_related_family_ids"].append(left["id"])
    return families
