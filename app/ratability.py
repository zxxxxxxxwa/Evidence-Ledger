from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any


VALID_STATES = {"estimable", "insufficient", "conflict", "not_applicable"}


def _localized(value: Any, fallback_en: str = "", fallback_zh: str = "") -> dict[str, str]:
    if isinstance(value, dict):
        return {
            "en": str(value.get("en") or fallback_en).strip(),
            "zh": str(value.get("zh") or fallback_zh).strip(),
        }
    text = str(value or "").strip()
    return {"en": text or fallback_en, "zh": text or fallback_zh}


class RatabilityEngine:

    def __init__(
        self,
        calibration_path: Path | None = None,
        *,
        framework_version: str | None = None,
        valid_indicator_ids: set[str] | None = None,
    ):
        self.artifact: dict[str, Any] | None = None
        self.status = "provisional_rules"
        self.version = "provisional-2026-11-18"
        if calibration_path:
            artifact = json.loads(calibration_path.read_text(encoding="utf-8"))
            required = {"artifact_type", "version", "expert_calibrated", "source_manifest", "indicator_policies"}
            if (
                not isinstance(artifact, dict)
                or set(artifact) != required
                or artifact.get("artifact_type") != "ratability_calibration"
                or not isinstance(artifact.get("version"), str)
                or not artifact["version"].strip()
            ):
                raise RuntimeError("Invalid ratability calibration artifact")
            if artifact.get("expert_calibrated") is not True:
                raise RuntimeError("Calibration artifact must explicitly declare expert_calibrated=true")
            manifest = artifact.get("source_manifest")
            manifest_required = {"created_at", "framework_version", "method", "data_fingerprint"}
            if not isinstance(manifest, dict) or set(manifest) != manifest_required:
                raise RuntimeError("Calibration source_manifest is incomplete")
            if not all(str(manifest.get(item, "")).strip() for item in manifest_required):
                raise RuntimeError("Calibration source_manifest fields must be non-empty")
            if framework_version and manifest["framework_version"] != framework_version:
                raise RuntimeError("Calibration framework_version does not match the loaded framework")
            if not isinstance(artifact.get("indicator_policies"), dict):
                raise RuntimeError("Calibration indicator_policies must be an object")
            allowed_features = {
                "family_count",
                "log1p_family_count",
                "critical_coverage",
                "attributable_family_fraction",
                "temporally_relevant_family_fraction",
                "conflict_pair_count",
            }
            for indicator_id, policy in artifact["indicator_policies"].items():
                if valid_indicator_ids is not None and indicator_id not in valid_indicator_ids:
                    raise RuntimeError(f"Unknown calibration indicator: {indicator_id}")
                if not isinstance(policy, dict):
                    raise RuntimeError(f"Calibration policy for {indicator_id} must be an object")
                if not set(policy).issubset(
                    {"min_independent_families", "contribution_required", "ratability_model"}
                ):
                    raise RuntimeError(f"Unknown calibration policy field for {indicator_id}")
                if "min_independent_families" in policy and (
                    not isinstance(policy["min_independent_families"], int)
                    or isinstance(policy["min_independent_families"], bool)
                    or policy["min_independent_families"] < 1
                ):
                    raise RuntimeError(f"Invalid min_independent_families for {indicator_id}")
                if "contribution_required" in policy and not isinstance(
                    policy["contribution_required"], bool
                ):
                    raise RuntimeError(f"Invalid contribution_required for {indicator_id}")
                model = policy.get("ratability_model")
                if model is None:
                    continue
                if (
                    not isinstance(model, dict)
                    or set(model) != {"type", "intercept", "coefficients", "threshold"}
                    or model.get("type") != "logistic"
                ):
                    raise RuntimeError(f"Unsupported calibration model for {indicator_id}")
                intercept = model.get("intercept")
                threshold_value = model.get("threshold")
                coefficients = model.get("coefficients")
                if (
                    not isinstance(intercept, (int, float))
                    or isinstance(intercept, bool)
                    or not isinstance(threshold_value, (int, float))
                    or isinstance(threshold_value, bool)
                    or not 0 <= float(threshold_value) <= 1
                    or not isinstance(coefficients, dict)
                    or not set(coefficients).issubset(allowed_features)
                    or any(
                        not isinstance(value, (int, float)) or isinstance(value, bool)
                        for value in coefficients.values()
                    )
                ):
                    raise RuntimeError(f"Invalid calibration model for {indicator_id}")
            self.artifact = artifact
            self.status = "calibrated_model"
            self.version = str(artifact["version"])

    def metadata(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "version": self.version,
            "expert_calibrated": self.status == "calibrated_model",
        }
        if self.artifact:
            payload["sha256"] = hashlib.sha256(
                json.dumps(self.artifact, sort_keys=True).encode("utf-8")
            ).hexdigest()
        return payload

    def _policy(self, indicator: dict[str, Any]) -> dict[str, Any]:
        policy = {
            "min_independent_families": int(indicator.get("min_independent_families", 1)),
            "contribution_required": bool(indicator.get("contribution_required", False)),
        }
        if self.artifact:
            policy.update(self.artifact.get("indicator_policies", {}).get(indicator["id"], {}))
        return policy

    @staticmethod
    def _features(
        indicator: dict[str, Any],
        selected_ids: list[str],
        family_map: dict[str, dict[str, Any]],
        covered: set[str],
        conflict_pair_count: int,
    ) -> dict[str, float]:
        selected = [family_map[item] for item in selected_ids if item in family_map]
        count = len(selected)
        critical = set(indicator.get("critical_observables", []))
        critical_coverage = len(critical & covered) / len(critical) if critical else 1.0
        attributable = sum(item.get("max_contribution_factor", 0) >= 0.8 for item in selected)
        temporally_relevant = sum(
            any(band in {"current", "recent"} for band in item.get("recency_bands", []))
            for item in selected
        )
        return {
            "family_count": float(count),
            "log1p_family_count": math.log1p(count),
            "critical_coverage": float(critical_coverage),
            "attributable_family_fraction": attributable / count if count else 0.0,
            "temporally_relevant_family_fraction": temporally_relevant / count if count else 0.0,
            "conflict_pair_count": float(conflict_pair_count),
        }

    @staticmethod
    def _calibrated_probability(policy: dict[str, Any], features: dict[str, float]) -> float | None:
        model = policy.get("ratability_model")
        if not isinstance(model, dict) or model.get("type") != "logistic":
            return None
        coefficients = model.get("coefficients")
        if not isinstance(coefficients, dict):
            return None
        logit = float(model.get("intercept", 0.0))
        for feature, coefficient in coefficients.items():
            if feature not in features:
                raise RuntimeError(f"Unknown calibration feature: {feature}")
            logit += float(coefficient) * features[feature]
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exponential = math.exp(logit)
        return exponential / (1.0 + exponential)

    def decide(
        self,
        indicator: dict[str, Any],
        raw: dict[str, Any],
        families: list[dict[str, Any]],
    ) -> dict[str, Any]:
        family_map = {family["id"]: family for family in families if indicator["id"] in family["indicator_ids"]}
        raw_family_ids = raw.get("family_ids") if isinstance(raw.get("family_ids"), list) else []
        proposed_family_ids = list(dict.fromkeys(item for item in raw_family_ids if item in family_map))




        selected_ids = proposed_family_ids
        usable_selected_ids = [
            item for item in selected_ids if not family_map[item].get("all_security_restricted", False)
        ]
        all_usable_family_ids = [
            item
            for item in sorted(family_map)
            if not family_map[item].get("all_security_restricted", False)
        ]
        conflict_pairs: list[list[str]] = []
        temporal_pair_requires_review = False
        conflict_observables = set(indicator.get("critical_observables", [])) or set(
            indicator.get("observables", [])
        )

        def polarity_by_observable(family_id: str) -> dict[str, set[str]]:
            result: dict[str, set[str]] = {}
            for member in family_map[family_id].get("members", []):
                observable = member.get("observable")
                if (
                    member.get("indicator_id") == indicator["id"]
                    and observable in conflict_observables
                    and member.get("polarity") in {"support", "contradict"}
                ):
                    result.setdefault(str(observable), set()).add(str(member["polarity"]))
            return result

        polarities = {
            family_id: polarity_by_observable(family_id) for family_id in all_usable_family_ids
        }
        within_family_mixed = any(
            {"support", "contradict"}.issubset(polarity_set)
            for by_observable in polarities.values()
            for polarity_set in by_observable.values()
        )
        for left_id, right_id in combinations(all_usable_family_ids, 2):
            left_sources = set(family_map[left_id].get("independent_source_keys", []))
            right_sources = set(family_map[right_id].get("independent_source_keys", []))
            independent_sources = bool(
                left_sources and right_sources and left_sources.isdisjoint(right_sources)
            )
            shared_observables = set(polarities[left_id]) & set(polarities[right_id])
            opposite_polarity = any(
                (
                    "support" in polarities[left_id][observable]
                    and "contradict" in polarities[right_id][observable]
                )
                or (
                    "contradict" in polarities[left_id][observable]
                    and "support" in polarities[right_id][observable]
                )
                for observable in shared_observables
            )
            left_years = {
                int(str(member.get("event_date"))[:4])
                for member in family_map[left_id].get("members", [])
                if str(member.get("event_date", ""))[:4].isdigit()
            }
            right_years = {
                int(str(member.get("event_date"))[:4])
                for member in family_map[right_id].get("members", [])
                if str(member.get("event_date", ""))[:4].isdigit()
            }
            temporally_separated = bool(
                left_years
                and right_years
                and min(abs(left - right) for left in left_years for right in right_years) >= 2
            )
            if independent_sources and opposite_polarity and temporally_separated:
                temporal_pair_requires_review = True
            elif independent_sources and opposite_polarity:
                conflict_pairs.append([left_id, right_id])

        policy = self._policy(indicator)
        allowed_observables = set(indicator.get("observables", []))
        covered = {
            str(member.get("observable"))
            for family_id in usable_selected_ids
            for member in family_map[family_id].get("members", [])
            if member.get("indicator_id") == indicator["id"]
            and member.get("polarity") == "support"
            and member.get("observable") in allowed_observables
        }
        features = self._features(indicator, usable_selected_ids, family_map, covered, len(conflict_pairs))
        calibrated_probability = self._calibrated_probability(policy, features)
        model_config = policy.get("ratability_model") if calibrated_probability is not None else None
        calibrated_threshold = float(model_config.get("threshold", 0.5)) if model_config else None

        proposed_state = raw.get("proposed_state")
        if proposed_state not in VALID_STATES:
            proposed_state = "insufficient"
        if raw.get("applicable") is False or proposed_state == "not_applicable":
            state = "not_applicable"
            reason_codes = ["not_applicable_to_context"]
        elif conflict_pairs:
            state = "conflict"
            reason_codes = ["independent_evidence_conflict"]
        else:
            critical = set(indicator.get("critical_observables", []))
            critical_present = critical.issubset(covered)
            attributable_family = any(
                family_map[family_id].get("max_contribution_factor", 0) >= 0.8
                for family_id in usable_selected_ids
            )
            any_attribution = any(
                family_map[family_id].get("max_contribution_factor", 0) > 0
                for family_id in usable_selected_ids
            )
            insufficient_reasons: list[str] = []
            if len(set(usable_selected_ids)) < int(policy["min_independent_families"]):
                insufficient_reasons.append("too_few_independent_families")
            if len(usable_selected_ids) < len(selected_ids):
                insufficient_reasons.append("security_flagged_evidence_excluded")
            if temporal_pair_requires_review:
                insufficient_reasons.append("temporal_change_requires_review")
            if within_family_mixed:
                insufficient_reasons.append("within_family_mixed_evidence")
            if not critical_present:
                insufficient_reasons.append("critical_observable_missing")
            if not any_attribution:
                insufficient_reasons.append("personal_contribution_unclear")
            elif policy.get("contribution_required") and not attributable_family:
                insufficient_reasons.append("personal_contribution_unclear")
            if proposed_state in {"insufficient", "conflict"} and not conflict_pairs:
                insufficient_reasons.append("model_requested_abstention")
            if calibrated_probability is not None and calibrated_probability < calibrated_threshold:
                insufficient_reasons.append("expert_calibrated_ratability_below_threshold")
            state = "insufficient" if insufficient_reasons else "estimable"
            reason_codes = insufficient_reasons

        level: int | None = None
        interval: list[int] | None = None
        if state == "estimable":
            candidate = raw.get("level")
            try:
                level = int(candidate)
            except (TypeError, ValueError):
                level = None
            if level not in {1, 2, 3, 4}:
                state = "insufficient"
                reason_codes.append("conditional_level_missing")
                level = None
            raw_interval = raw.get("level_interval")
            if level is not None and isinstance(raw_interval, list) and len(raw_interval) == 2:
                try:
                    lower, upper = int(raw_interval[0]), int(raw_interval[1])
                except (TypeError, ValueError):
                    lower = upper = level
                interval = [max(1, min(lower, level)), min(4, max(upper, level))]
            elif level is not None:
                interval = [level, level]

        if state != "estimable":
            level = None
            interval = None

        missing: list[dict[str, str]] = []
        raw_missing = raw.get("missing_evidence") if isinstance(raw.get("missing_evidence"), list) else []
        for item in raw_missing:




            if not isinstance(item, dict):
                continue
            localized_item = _localized(
                item,
                fallback_en="Provide additional observable evidence for this indicator.",
                fallback_zh="请为该指标补充更多可观察证据。",
            )
            if localized_item["en"] and localized_item["zh"]:
                missing.append(localized_item)
        if state == "insufficient" and not missing:
            observables = ", ".join(indicator.get("critical_observables", []))
            missing = [
                {
                    "en": f"Add observable evidence for: {observables}",
                    "zh": f"请补充以下方面的可观察证据：{observables}",
                }
            ]
        if "security_flagged_evidence_excluded" in reason_codes:
            missing.insert(
                0,
                {
                    "en": "Review the flagged document locally and provide a clean, instruction-free copy before using it as evidence.",
                    "zh": "请先在本地核查被标记文档，并提供不含指令式内容的干净副本，再将其用作证据。",
                },
            )
        if "temporal_change_requires_review" in reason_codes:
            missing.insert(
                0,
                {
                    "en": "Clarify whether the dated evidence reflects development over time or a current contradiction.",
                    "zh": "请澄清带日期证据反映的是随时间发展，还是当前仍存在的矛盾。",
                },
            )

        return {
            "indicator_id": indicator["id"],
            "dimension": indicator["dimension"],
            "title_en": indicator["title_en"],
            "title_zh": indicator["title_zh"],
            "description_en": indicator.get("description_en", ""),
            "description_zh": indicator.get("description_zh", ""),
            "state": state,
            "level": level,
            "level_interval": interval,



            "family_ids": list(
                dict.fromkeys(
                    usable_selected_ids
                    + [family_id for pair in conflict_pairs for family_id in pair]
                )
            ),
            "proposed_family_ids": proposed_family_ids,
            "conflict_pairs": conflict_pairs,
            "covered_observables": sorted(covered),
            "missing_evidence": missing,
            "reason_codes": reason_codes,
            "ratability_probability": round(calibrated_probability, 6) if calibrated_probability is not None else None,
            "ratability_features": features,
            "rationale": _localized(
                raw.get("rationale"),
                fallback_en="The decision is limited to the submitted evidence.",
                fallback_zh="该判断仅限于已提交证据。",
            ),
            "temporal_note": _localized(
                raw.get("temporal_note"),
                fallback_en="The dated evidence requires cautious interpretation.",
                fallback_zh="带日期的证据需要谨慎解释。",
            )
            if raw.get("temporal_note")
            else None,
            "calibration": {
                **self.metadata(),
                "applied_to_indicator": calibrated_probability is not None,
                "threshold": calibrated_threshold,
            },
        }
