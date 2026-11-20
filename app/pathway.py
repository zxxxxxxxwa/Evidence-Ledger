from __future__ import annotations

from typing import Any

from .errors import AppError
from .llm_client import StructuredLLM
from .prompts import pathway_messages
from .retrieval import ResourceLibrary, tokenize


def localized(value: Any, fallback_en: str = "", fallback_zh: str = "") -> dict[str, str]:
    if isinstance(value, dict):
        return {
            "en": str(value.get("en") or fallback_en).strip(),
            "zh": str(value.get("zh") or fallback_zh).strip(),
        }
    text = str(value or "").strip()
    return {"en": text or fallback_en, "zh": text or fallback_zh}


class PathwayPlanner:
    def __init__(self, library: ResourceLibrary, llm: StructuredLLM):
        self.library = library
        self.llm = llm

    @staticmethod
    def _goal_match(profile: dict[str, Any], decision: dict[str, Any]) -> float:
        goals = " ".join(str(item) for item in profile.get("goals", []))
        target = " ".join(
            [
                decision.get("title_en", ""),
                decision.get("title_zh", ""),
                decision.get("description_en", ""),
                decision.get("description_zh", ""),
                " ".join(str(item) for item in decision.get("covered_observables", [])),
                decision.get("dimension", ""),
            ]
        )
        goal_tokens, target_tokens = set(tokenize(goals)), set(tokenize(target))
        if not goal_tokens or not target_tokens:
            return 0.0
        return len(goal_tokens & target_tokens) / max(1, len(target_tokens))

    def build_routes(
        self, profile: dict[str, Any], decisions: list[dict[str, Any]], max_active: int = 4
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        all_routes: list[dict[str, Any]] = []
        candidates: list[tuple[float, dict[str, Any]]] = []
        for decision in decisions:
            state = decision["state"]
            goal_match = self._goal_match(profile, decision)
            if state == "estimable":
                level = decision.get("level") or 1
                if goal_match <= 0:
                    route = {
                        "indicator_id": decision["indicator_id"],
                        "state": state,
                        "route": "no_action",
                        "development_mode": "goal_not_matched",
                        "level": level,
                        "goal_match": 0.0,
                        "priority": 0,
                        "rationale": decision["rationale"],
                    }
                    all_routes.append(route)
                    continue
                if level <= 2:
                    mode = "strengthen"
                elif level == 3:
                    mode = "deepen"
                else:
                    mode = "integrate"
                route = {
                    "indicator_id": decision["indicator_id"],
                    "state": state,
                    "route": "develop",
                    "development_mode": mode,
                    "level": level,
                    "goal_match": round(goal_match, 3),
                    "priority": round(goal_match * 10 + (5 - level), 3),
                    "rationale": decision["rationale"],
                }
                candidates.append((route["priority"], route))
            elif state == "insufficient":
                route = {
                    "indicator_id": decision["indicator_id"],
                    "state": state,
                    "route": "request_evidence",
                    "development_mode": "diagnostic",
                    "missing_evidence": decision.get("missing_evidence", []),
                    "goal_match": round(goal_match, 3),
                    "priority": round(goal_match * 10 + 2, 3),
                    "rationale": decision["rationale"],
                }


                if goal_match > 0:
                    candidates.append((route["priority"], route))
            elif state == "conflict":
                route = {
                    "indicator_id": decision["indicator_id"],
                    "state": state,
                    "route": "human_review",
                    "development_mode": "paused",
                    "priority": 99,
                    "rationale": decision["rationale"],
                }
            else:
                route = {
                    "indicator_id": decision["indicator_id"],
                    "state": state,
                    "route": "no_action",
                    "development_mode": "not_applicable",
                    "priority": 0,
                    "rationale": decision["rationale"],
                }
            all_routes.append(route)

        active_ids = {
            route["indicator_id"]
            for _, route in sorted(candidates, key=lambda pair: (-pair[0], pair[1]["indicator_id"]))[:max_active]
        }
        active_routes = [route for route in all_routes if route["indicator_id"] in active_ids]
        return all_routes, active_routes

    @staticmethod
    def _target_difficulty(route: dict[str, Any]) -> str:
        if route["route"] == "request_evidence":
            return "foundation"
        level = int(route.get("level") or 1)
        if level <= 2:
            return "foundation"
        if level == 3:
            return "intermediate"
        return "advanced"

    def _resources_for_route(
        self,
        profile: dict[str, Any],
        decision: dict[str, Any],
        route: dict[str, Any],
        language: str,
    ) -> list[dict[str, Any]]:
        query = " ".join(
            [
                " ".join(str(item) for item in profile.get("goals", [])),
                decision.get("title_en", ""),
                decision.get("title_zh", ""),
                decision.get("dimension", ""),
                route.get("development_mode", ""),
            ]
        )
        found = self.library.search(
            query,
            indicator_id=decision["indicator_id"],
            dimension=decision["dimension"],
            safe_when_unknown=route["route"] == "request_evidence",
            language=language,
            target_difficulty=self._target_difficulty(route),
            limit=4,
        )
        unknown_safe = route["route"] == "request_evidence"

        def governed(resource: dict[str, Any]) -> bool:
            return bool(
                resource.get("accessible")
                and str(resource.get("license") or "").strip()
                and (not unknown_safe or resource.get("safe_when_unknown"))
            )

        roots: list[str] = []
        ordered_ids: list[str] = []
        enriched = {item["id"]: item for item in found}
        for item in found:
            try:
                candidate_order = self.library.prerequisite_closure(
                    roots + [item["id"]], allowed=governed
                )
            except ValueError:
                continue

            if len(candidate_order) > 5:
                continue
            roots.append(item["id"])
            ordered_ids = candidate_order
            if len(ordered_ids) >= 4:
                break
        return [{**self.library.by_id[item], **enriched.get(item, {})} for item in ordered_ids]

    @staticmethod
    def _bilingual_complete(value: Any) -> bool:
        return bool(
            isinstance(value, dict)
            and str(value.get("en") or "").strip()
            and str(value.get("zh") or "").strip()
        )

    def _fallback_path(
        self,
        route: dict[str, Any],
        resources: list[dict[str, Any]],
        weekly_minutes: int,
        reason: str,
    ) -> dict[str, Any]:
        if route["route"] == "request_evidence":
            summary = {
                "en": "Clarify the evidence gap before interpreting this capability.",
                "zh": "先澄清证据缺口，再解释该项能力。",
            }
        else:
            summary = {
                "en": "Use the retrieved sources to complete one goal-linked practice cycle.",
                "zh": "利用检索到的资源完成一个与目标相连的实践循环。",
            }
        steps: list[dict[str, Any]] = []
        remaining = max(30, weekly_minutes * 4)
        resource_map = {item["id"]: item for item in resources}
        try:
            ordered_ids = self.library.prerequisite_closure(
                list(resource_map), allowed=lambda item: item["id"] in resource_map
            )
        except ValueError:
            ordered_ids = []
        completed: set[str] = set()
        for resource_id in ordered_ids:
            resource = resource_map[resource_id]
            try:
                closure = self.library.prerequisite_closure(
                    [resource_id], allowed=lambda item: item["id"] in resource_map
                )
            except ValueError:
                continue
            if any(item not in completed for item in closure[:-1]):
                continue
            duration = min(int(resource.get("duration_minutes", 30)), remaining)
            if duration < 10:
                break
            if route["route"] == "request_evidence":
                objective = {
                    "en": "Use a low-assumption task to produce observable evidence.",
                    "zh": "通过低假设诊断任务形成可观察证据。",
                }
                task = {
                    "en": "Apply one checklist or prompt to a current work sample; retain the before/after version and a short rationale.",
                    "zh": "将一项检查表或提示应用于当前工作样例，保留修改前后版本和简短理由。",
                }
            else:
                objective = {
                    "en": "Connect the source to the stated development goal.",
                    "zh": "将该资源与已陈述的发展目标建立联系。",
                }
                task = {
                    "en": "Apply one idea to a current teaching or research task and record the decision, evidence and reflection.",
                    "zh": "把一个要点应用到当前教学或科研任务，并记录决策、证据与反思。",
                }
            steps.append(
                {
                    "resource_id": resource["id"],
                    "objective": objective,
                    "reason": {
                        "en": "Retrieved for this indicator, goal and evidence state.",
                        "zh": "依据该指标、目标和证据状态检索。",
                    },
                    "duration_minutes": duration,
                    "practice_task": task,
                    "completion_evidence": {
                        "en": "A dated work sample and a short reflection; submission remains optional.",
                        "zh": "带日期的工作样例与简短反思；是否提交由教师自主决定。",
                    },
                }
            )
            remaining -= duration
            completed.add(resource_id)
        return {
            "indicator_id": route["indicator_id"],
            "route": route["route"],
            "summary": summary,
            "steps": steps,
            "generation_mode": "catalog_rule_fallback",
            "fallback_reason": reason,
        }

    def _checked_llm_path(
        self,
        candidate: Any,
        route: dict[str, Any],
        allowed: dict[str, dict[str, Any]],
        weekly_minutes: int,
    ) -> dict[str, Any] | None:
        if not isinstance(candidate, dict) or candidate.get("route") != route["route"]:
            return None
        if not self._bilingual_complete(candidate.get("summary")):
            return None
        raw_steps = candidate.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps or len(raw_steps) > 5:
            return None

        budget = weekly_minutes * 4
        seen: set[str] = set()
        valid_steps: list[dict[str, Any]] = []
        for step in raw_steps:
            if not isinstance(step, dict):
                return None
            resource_id = step.get("resource_id")
            if resource_id not in allowed or resource_id in seen:
                return None
            if not all(
                self._bilingual_complete(step.get(field))
                for field in ("objective", "reason", "practice_task", "completion_evidence")
            ):
                return None
            try:
                duration = int(step.get("duration_minutes", 0))
            except (TypeError, ValueError):
                return None
            if duration < 10 or duration > 240 or duration > budget:
                return None
            try:
                closure = self.library.prerequisite_closure(
                    [resource_id], allowed=lambda item: item["id"] in allowed
                )
            except ValueError:
                return None
            if any(prerequisite not in seen for prerequisite in closure[:-1]):
                return None
            valid_steps.append(
                {
                    "resource_id": resource_id,
                    "objective": localized(step["objective"]),
                    "reason": localized(step["reason"]),
                    "duration_minutes": duration,
                    "practice_task": localized(step["practice_task"]),
                    "completion_evidence": localized(step["completion_evidence"]),
                }
            )
            seen.add(resource_id)
            budget -= duration

        return {
            "indicator_id": route["indicator_id"],
            "route": route["route"],
            "summary": localized(candidate["summary"]),
            "steps": valid_steps,
            "generation_mode": "llm_rule_checked",
        }

    def plan(
        self,
        profile: dict[str, Any],
        decisions: list[dict[str, Any]],
        language: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        all_routes, active_routes = self.build_routes(profile, decisions)
        decision_map = {item["indicator_id"]: item for item in decisions}
        route_resources: dict[str, list[dict[str, Any]]] = {}
        whitelist: dict[str, dict[str, Any]] = {}
        for route in active_routes:
            resources = self._resources_for_route(
                profile, decision_map[route["indicator_id"]], route, language
            )
            route_resources[route["indicator_id"]] = resources
            whitelist.update({item["id"]: item for item in resources})

        if not active_routes:
            return all_routes, [], []

        llm_resources = [
            {
                "id": item["id"],
                "title": item["title"],
                "title_zh": item["title_zh"],
                "summary_en": item["summary_en"],
                "summary_zh": item["summary_zh"],
                "difficulty": item["difficulty"],
                "duration_minutes": item["duration_minutes"],
                "prerequisites": item["prerequisites"],
                "license": item["license"],
            }
            for item in whitelist.values()
        ]
        llm_failure_reason: str | None = None
        raw_paths: list[Any] = []
        if llm_resources:
            try:
                raw = self.llm.complete_json(
                    pathway_messages(profile, active_routes, llm_resources, language),
                    purpose="pathway_generation",
                )
                if isinstance(raw, dict) and isinstance(raw.get("paths"), list):
                    raw_paths = raw["paths"]
                else:
                    llm_failure_reason = "llm_output_schema_rejected"
            except AppError as exc:
                llm_failure_reason = f"llm_error:{exc.code}"
        else:
            llm_failure_reason = "no_governed_resources"
        raw_by_indicator = {
            item.get("indicator_id"): item for item in raw_paths if isinstance(item, dict) and item.get("indicator_id")
        }

        weekly_minutes = max(15, min(int(profile.get("weekly_minutes") or 60), 600))
        paths: list[dict[str, Any]] = []
        for route in active_routes:
            indicator_id = route["indicator_id"]
            allowed = {item["id"]: item for item in route_resources.get(indicator_id, [])}
            candidate = raw_by_indicator.get(indicator_id)
            path = self._checked_llm_path(candidate, route, allowed, weekly_minutes)
            if path is None:
                reason = llm_failure_reason or "llm_output_rule_rejected"
                path = self._fallback_path(
                    route, list(allowed.values()), weekly_minutes, reason
                )
            path["citations"] = [
                {
                    "resource_id": step["resource_id"],
                    "title": allowed[step["resource_id"]]["title"],
                    "title_zh": allowed[step["resource_id"]]["title_zh"],
                    "url": allowed[step["resource_id"]]["url"],
                    "publisher": allowed[step["resource_id"]]["publisher"],
                    "license": allowed[step["resource_id"]]["license"],
                    "checked_at": allowed[step["resource_id"]]["checked_at"],
                }
                for step in path["steps"]
            ]
            path["total_minutes"] = sum(item["duration_minutes"] for item in path["steps"])
            paths.append(path)

        return all_routes, paths, list(whitelist.values())
