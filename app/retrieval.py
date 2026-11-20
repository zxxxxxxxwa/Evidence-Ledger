from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Callable


AUTHORITY_SCORES = {
    "intergovernmental": 3.0,
    "government": 2.6,
    "international research network": 2.2,
    "university": 2.0,
    "nonprofit research organization": 1.8,
}
DIFFICULTY_ORDER = {"foundation": 0, "intermediate": 1, "advanced": 2}


def tokenize(text: str) -> list[str]:
    text = text.casefold()
    words = re.findall(r"[a-z0-9][a-z0-9_-]{1,}", text)
    han = re.findall(r"[\u4e00-\u9fff]", text)
    bigrams = ["".join(han[index : index + 2]) for index in range(max(0, len(han) - 1))]
    return words + bigrams


class ResourceLibrary:
    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.version = payload["version"]
        self.notice = payload.get("notice", "")
        self.resources: list[dict[str, Any]] = payload["resources"]
        self.by_id = {item["id"]: item for item in self.resources}
        self.documents = [self._document(resource) for resource in self.resources]
        self.term_counts = [Counter(tokenize(document)) for document in self.documents]
        document_frequency: Counter[str] = Counter()
        for counts in self.term_counts:
            document_frequency.update(counts.keys())
        total = len(self.resources)
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    @staticmethod
    def _document(resource: dict[str, Any]) -> str:
        fields = [
            resource.get("title", ""),
            resource.get("title_zh", ""),
            resource.get("summary_en", ""),
            resource.get("summary_zh", ""),
            " ".join(resource.get("topics", [])),
            " ".join(resource.get("dimensions", [])),
            " ".join(resource.get("indicator_ids", [])),
        ]
        return " ".join(fields)

    def search(
        self,
        query: str,
        *,
        indicator_id: str | None = None,
        dimension: str | None = None,
        safe_when_unknown: bool = False,
        language: str | None = None,
        target_difficulty: str | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        query_counts = Counter(tokenize(query))
        scored: list[tuple[float, float, dict[str, Any], list[str]]] = []
        preferred_language = language if language in {"en", "zh"} else None
        target_rank = DIFFICULTY_ORDER.get(target_difficulty or "")
        for resource, counts in zip(self.resources, self.term_counts, strict=True):
            if not resource.get("accessible", False):
                continue
            if not str(resource.get("license") or "").strip():
                continue
            if safe_when_unknown and not resource.get("safe_when_unknown", False):
                continue
            relevance_score = 0.0
            for term, frequency in query_counts.items():
                if term in counts:
                    relevance_score += self.idf.get(term, 0.0) * min(frequency, 2) * (1 + math.log1p(counts[term]))
            if indicator_id and indicator_id in resource.get("indicator_ids", []):
                relevance_score += 5.0
            if dimension and dimension in resource.get("dimensions", []):
                relevance_score += 2.0
            if safe_when_unknown and resource.get("diagnostic_task"):
                relevance_score += 1.0

            governance_score = AUTHORITY_SCORES.get(str(resource.get("authority", "")).casefold(), 1.0)
            reasons = [f"authority:{resource.get('authority', 'unspecified')}"]

            languages = {str(item).casefold() for item in resource.get("languages", [])}
            if preferred_language:
                if preferred_language in languages:
                    governance_score += 1.5
                    reasons.append(f"language:{preferred_language}")
                elif preferred_language == "zh" and "en" in languages:

                    governance_score += 0.2
                    reasons.append("language:english_fallback")
                else:
                    governance_score -= 0.5
                    reasons.append("language:mismatch")

            resource_rank = DIFFICULTY_ORDER.get(str(resource.get("difficulty", "")))
            if target_rank is not None and resource_rank is not None:
                distance = abs(target_rank - resource_rank)
                governance_score += max(0.0, 1.2 - 0.6 * distance)
                reasons.append(f"difficulty_distance:{distance}")

            checked_at = str(resource.get("checked_at") or "")
            try:
                checked_date = date.fromisoformat(checked_at)
                age_days = max(0, (date.today() - checked_date).days)
                freshness = max(0.0, 0.8 - age_days / 730)
                governance_score += freshness
                reasons.append(f"checked_days_ago:{age_days}")
            except ValueError:
                reasons.append("checked_at:unverified")

            if relevance_score > 0:
                total_score = relevance_score + governance_score
                scored.append((total_score, governance_score, resource, reasons))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]["id"]))
        return [
            {
                **resource,
                "retrieval_score": round(total_score, 4),
                "governance_score": round(governance_score, 4),
                "governance_reasons": reasons,
            }
            for total_score, governance_score, resource, reasons in scored[:limit]
        ]

    def public_catalog(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "notice": self.notice,
            "resources": self.resources,
        }

    def prerequisite_closure(
        self,
        resource_ids: list[str],
        *,
        allowed: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[str]:

        ordered: list[str] = []
        state: dict[str, int] = {}
        stack: list[str] = []

        def visit(resource_id: str) -> None:
            if resource_id not in self.by_id:
                raise ValueError(f"unknown_prerequisite:{resource_id}")
            marker = state.get(resource_id, 0)
            if marker == 2:
                return
            if marker == 1:
                cycle_start = stack.index(resource_id) if resource_id in stack else 0
                cycle = " -> ".join(stack[cycle_start:] + [resource_id])
                raise ValueError(f"prerequisite_cycle:{cycle}")
            resource = self.by_id[resource_id]
            if allowed is not None and not allowed(resource):
                raise ValueError(f"prerequisite_not_allowed:{resource_id}")
            state[resource_id] = 1
            stack.append(resource_id)
            for prerequisite in resource.get("prerequisites", []):
                visit(str(prerequisite))
            stack.pop()
            state[resource_id] = 2
            ordered.append(resource_id)

        for resource_id in resource_ids:
            visit(resource_id)
        return ordered

    def order_with_prerequisites(self, resource_ids: list[str]) -> list[str]:

        return self.prerequisite_closure(resource_ids)
