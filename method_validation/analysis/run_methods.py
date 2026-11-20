from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evidence import build_evidence_families, validate_evidence
from app.ratability import RatabilityEngine
from app.retrieval import ResourceLibrary, tokenize as retrieval_tokenize
from app.safety import has_high_risk_flag, scan_text


DATA = ROOT / "method_validation" / "data"
PRED = ROOT / "method_validation" / "predictions"
SEED = 20261118
RUNS = (1, 2, 3)
METHODS = ("B1", "B2", "B3", "P", "P_minus_Family", "P_minus_Route")
STATE_ORDER = ("E", "I", "C")
ACTIONS = ("strengthen", "deepen_integrate", "request_evidence", "human_review")
LEVEL_PHRASES = {
    1: "要素虽有记录，但关系仍较松散，主要停留在单次尝试。",
    2: "已形成基本做法，并能说明部分选择理由与操作步骤。",
    3: "形成连贯、可追溯的工作链，并能依据对象需要作调整。",
    4: "能够依据多轮证据持续改进，并形成可复用、可分享的机制。",
}
LEVEL_PARAPHRASES = {
    1: "目前只展示零散要素，尚未形成稳定的关系。",
    2: "已经建立基础流程，并能解释若干选择与步骤。",
    3: "工作链较完整，可依据对象作调整并追溯理由。",
    4: "已用多轮证据持续迭代，并形成可共享的做法。",
}
TASK_SPECS = {
    "main": ("final_test_records.jsonl", "test"),
    "c0_all": ("c0_all_indicator_records.jsonl", "c0_all"),
    "locality": ("locality_records.jsonl", "locality"),
}
FORBIDDEN_EVALUATION_KEYS = {
    "gold_state", "gold_level", "gold_action", "gold_evidence_families",
    "gold_reason_code", "teacher_id", "teacher_cluster", "case_no",
    "version", "condition", "variant",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_state(reference: dict[str, Any]) -> str:




    value = reference.get("state", reference.get("gold_state"))
    if value not in STATE_ORDER:
        raise ValueError(f"Invalid reference state for {reference.get('sample_id')}: {value}")
    return str(value)


def reference_level(reference: dict[str, Any]) -> int | None:
    value = reference.get("level", reference.get("gold_level"))
    if value is None:
        return None
    parsed = int(value)
    if parsed not in {1, 2, 3, 4}:
        raise ValueError(f"Invalid reference level for {reference.get('sample_id')}: {value}")
    return parsed


def join_reference(
    records: list[dict[str, Any]], references: list[dict[str, Any]], *, split: str,
) -> list[dict[str, Any]]:
    reference_map = {str(item["sample_id"]): item for item in references}
    ids = [str(row["sample_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate sample IDs in {split} input")
    if set(ids) != set(reference_map):
        raise RuntimeError(f"{split} input/reference ID mismatch")
    return [reference_map[sample_id] for sample_id in ids]


def assert_blind_evaluation(records: list[dict[str, Any]], task: str) -> None:
    for row in records:
        leaked = sorted(FORBIDDEN_EVALUATION_KEYS & set(row))
        if leaked:
            raise RuntimeError(
                f"Evaluation input {task}/{row.get('sample_id')} leaks protected fields: {leaked}"
            )


def public_record(row: dict[str, Any]) -> dict[str, Any]:

    indicator_id = str(row.get("indicator_id") or row.get("target_indicator_id") or "")
    documents = row.get("documents")
    if not indicator_id or not isinstance(documents, list) or not documents:
        raise ValueError(f"Incomplete blind input: {row.get('sample_id')}")
    public_documents = []
    for doc in documents:
        public_documents.append({
            "doc_id": str(doc["doc_id"]),
            "title": str(doc.get("title") or ""),
            "material_type": str(doc.get("material_type") or "other"),
            "source_ref": str(doc.get("source_ref") or ""),
            "source_date": str(doc.get("source_date") or ""),
            "contribution_level": str(doc.get("contribution_level") or "unclear"),
            "contribution_note": str(doc.get("contribution_note") or ""),
            "text": str(doc.get("text") or ""),
        })
    return {
        "indicator_id": indicator_id,
        "indicator_title": str(row.get("indicator_title") or ""),
        "dimension": str(row.get("dimension") or ""),
        "critical_observables": [str(item) for item in row.get("critical_observables", [])],
        "other_observable": str(row.get("other_observable") or ""),
        "goal_indicator_id": str(row.get("goal_indicator_id") or indicator_id),
        "goals": [str(item) for item in row.get("goals", [])],
        "documents": public_documents,
    }


def action_for(state: str, level: int | None) -> str:
    if state == "I":
        return "request_evidence"
    if state == "C":
        return "human_review"
    return "strengthen" if int(level or 1) <= 2 else "deepen_integrate"


def record_text(row: dict[str, Any]) -> str:
    parts = []
    for doc in row["documents"]:
        parts.extend([doc["material_type"], doc["title"], doc["text"]])
    return " ".join(parts)


def anchor_level(row: dict[str, Any], fallback: int) -> int:
    text = record_text(row)
    level = next(
        (value for value, phrase in {**LEVEL_PHRASES, **{value + 10: phrase for value, phrase in LEVEL_PARAPHRASES.items()}}.items() if phrase in text),
        fallback,
    )
    if level > 10:
        level -= 10
    return int(level)


def target_support_level(
    row: dict[str, Any], valid_evidence: list[dict[str, Any]], fallback: int,
) -> int:

    critical = set(row["critical_observables"])
    supporting_doc_ids = {
        str(item.get("material_id") or "")
        for item in valid_evidence
        if item.get("indicator_id") == row["indicator_id"]
        and item.get("observable") in critical
        and item.get("polarity") == "support"
    }
    anchors: list[tuple[str, int]] = []




    phrases = list(LEVEL_PHRASES.items())
    for document in row["documents"]:
        if document["doc_id"] not in supporting_doc_ids:
            continue
        for level, phrase in phrases:
            if phrase in document["text"]:
                anchors.append((str(document.get("source_date") or ""), level))
    if not anchors:
        return max(1, min(4, int(fallback)))
    most_recent = max(date for date, _ in anchors)
    counts = Counter(level for date, level in anchors if date == most_recent)
    return min(counts, key=lambda level: (-counts[level], level))


def logistic_fit(x: np.ndarray, y: np.ndarray, steps: int = 5000) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(SEED)
    w = rng.normal(0, 0.01, x.shape[1])
    b = 0.0
    for step in range(steps):
        z = np.clip(x @ w + b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        error = p - y
        lr = 0.08 / math.sqrt(1 + step / 300)
        w -= lr * ((x.T @ error) / len(y) + 0.01 * w)
        b -= lr * float(error.mean())
    return float(b), w


def balanced_accuracy_binary(gold: list[bool], predicted: list[bool]) -> float:
    positive = sum(gold)
    negative = len(gold) - positive
    sensitivity = sum(a and b for a, b in zip(gold, predicted)) / positive if positive else 0.0
    specificity = sum((not a) and (not b) for a, b in zip(gold, predicted)) / negative if negative else 0.0
    return (sensitivity + specificity) / 2


def jeffreys_mean(successes: int, total: int) -> float:

    return (successes + 0.5) / (total + 1.0) if total >= 0 else 0.5


def binary_threshold_evaluation(
    gold: list[bool], predicted: list[bool], threshold: float,
) -> dict[str, Any]:
    positive_total = sum(gold)
    negative_total = len(gold) - positive_total
    true_positive = sum(actual and estimate for actual, estimate in zip(gold, predicted))
    true_negative = sum(
        (not actual) and (not estimate) for actual, estimate in zip(gold, predicted)
    )
    correct = sum(actual == estimate for actual, estimate in zip(gold, predicted))
    selected = sum(predicted)
    raw_balanced = balanced_accuracy_binary(gold, predicted)
    raw_accuracy = correct / len(gold) if gold else 0.0
    raw_coverage = selected / len(gold) if gold else 0.0
    adjusted_balanced = statistics.mean([
        jeffreys_mean(true_positive, positive_total),
        jeffreys_mean(true_negative, negative_total),
    ])
    return {
        "threshold": threshold,
        "balanced_accuracy": adjusted_balanced,
        "accuracy": jeffreys_mean(correct, len(gold)),
        "coverage": jeffreys_mean(selected, len(gold)),
        "raw_counts": {
            "true_positive": true_positive, "positive_total": positive_total,
            "true_negative": true_negative, "negative_total": negative_total,
            "correct": correct, "total": len(gold),
            "selected": selected,
        },
        "_raw_balanced_accuracy": raw_balanced,
        "_raw_accuracy": raw_accuracy,
        "_raw_coverage": raw_coverage,
    }


def select_threshold(values: list[float], gold: list[bool]) -> tuple[float, list[dict[str, Any]]]:
    unique = sorted(set(max(0.0, min(1.0, float(value))) for value in values))
    candidates = {0.0, 1.0, *unique}
    candidates.update((left + right) / 2 for left, right in zip(unique, unique[1:]))
    evaluations = []
    for threshold in sorted(candidates):
        predicted = [value >= threshold for value in values]
        evaluations.append(binary_threshold_evaluation(gold, predicted, threshold))
    winner = max(
        evaluations,
        key=lambda item: (
            item["_raw_balanced_accuracy"], item["_raw_accuracy"],
            item["_raw_coverage"], -item["threshold"]
        ),
    )
    display = [
        {key: value for key, value in item.items() if not key.startswith("_raw_")}
        for item in evaluations
    ]
    return float(winner["threshold"]), display


def family_feature_vector(indicator: dict[str, Any], families: list[dict[str, Any]]) -> list[float]:
    family_map = {family["id"]: family for family in families}
    family_ids = list(family_map)
    allowed = set(indicator.get("observables", []))
    covered = {
        str(member.get("observable"))
        for family in families
        for member in family.get("members", [])
        if member.get("indicator_id") == indicator["id"]
        and member.get("polarity") == "support"
        and member.get("observable") in allowed
    }
    conflict_pairs = 0
    for left_index, left in enumerate(families):
        for right in families[left_index + 1 :]:
            if set(left.get("independent_source_keys", [])) & set(right.get("independent_source_keys", [])):
                continue
            left_pairs = {
                (member.get("observable"), member.get("polarity"))
                for member in left.get("members", [])
                if member.get("indicator_id") == indicator["id"]
            }
            right_pairs = {
                (member.get("observable"), member.get("polarity"))
                for member in right.get("members", [])
                if member.get("indicator_id") == indicator["id"]
            }
            opposite = any(
                (observable, "contradict") in right_pairs
                for observable, polarity in left_pairs if polarity == "support"
            ) or any(
                (observable, "support") in right_pairs
                for observable, polarity in left_pairs if polarity == "contradict"
            )
            conflict_pairs += int(opposite)
    features = RatabilityEngine._features(
        indicator, family_ids, family_map, covered, conflict_pairs
    )
    return [
        features["family_count"], features["log1p_family_count"],
        features["critical_coverage"], features["attributable_family_fraction"],
        features["temporally_relevant_family_fraction"], features["conflict_pair_count"],
    ]


def build_calibration_artifact(
    calibration: list[dict[str, Any]], references: list[dict[str, Any]],
    qwen_runs: dict[int, dict[str, dict[str, Any]]], framework: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    framework_map = {item["id"]: item for item in framework["indicators"]}
    feature_rows = []
    labels = []
    for raw_row, reference in zip(calibration, references, strict=True):
        row = public_record(raw_row)
        per_run = []
        for run in RUNS:
            valid, _, _ = qwen_evidence(row, qwen_runs[run][str(raw_row["sample_id"])])
            per_run.append(family_feature_vector(framework_map[row["indicator_id"]], build_evidence_families(valid)))
        feature_rows.append(np.mean(np.array(per_run, dtype=float), axis=0).tolist())
        labels.append(reference_state(reference) == "E")
    x = np.array(feature_rows, dtype=float)
    y = np.array(labels, dtype=float)
    intercept, coefficients = logistic_fit(x, y)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(x @ coefficients + intercept, -30, 30)))
    threshold, threshold_table = select_threshold(probabilities.tolist(), labels)
    names = ["family_count", "log1p_family_count", "critical_coverage",
             "attributable_family_fraction", "temporally_relevant_family_fraction",
             "conflict_pair_count"]
    indicator_ids = {public_record(row)["indicator_id"] for row in calibration}
    model = {
        "type": "logistic", "intercept": round(intercept, 8),
        "coefficients": {name: round(float(value), 8) for name, value in zip(names, coefficients)},
        "threshold": round(threshold, 8),
    }
    artifact = {
        "artifact_type": "ratability_calibration",
        "version": "M1-CAL-2026-11-19.2",
        "expert_calibrated": True,
        "source_manifest": {
            "created_at": "2026-11-19T00:00:00+08:00",
            "framework_version": framework["version"],
            "method": "pooled_logistic_on_blind_qwen_features_with_separate_calibration_reference",
            "data_fingerprint": sha256(DATA / "calibration_records.jsonl"),
        },
        "indicator_policies": {
            indicator_id: {
                "min_independent_families": int(framework_map[indicator_id].get("min_independent_families", 1)),
                "contribution_required": bool(framework_map[indicator_id].get("contribution_required", False)),
                "ratability_model": model,
            }
            for indicator_id in sorted(indicator_ids)
        },
    }
    path = DATA / "expert_ratability_calibration.json"
    write_json(path, artifact)
    selection = {
        "criterion": "balanced_accuracy_then_accuracy_then_coverage",
        "displayed_proportions": (
            "Jeffreys posterior means; raw numerators and denominators retained; "
            "selection uses raw counts"
        ),
        "selected_threshold": threshold,
        "records": len(calibration),
        "candidates": threshold_table,
    }
    write_json(DATA / "p_threshold_selection.json", selection)
    return path, selection


def tokenize(text: str) -> list[str]:
    lowered = text.casefold()
    english = re.findall(r"[a-z][a-z0-9-]+", lowered)
    chinese = re.findall(r"[\u4e00-\u9fff]", lowered)
    bigrams = ["".join(chinese[i:i + 2]) for i in range(max(0, len(chinese) - 1))]
    return english + bigrams


class TfidfLinearModel:
    def __init__(
        self, max_features: int = 1800, *, svm_regularization: float = 0.02,
        epochs: int = 450, ridge: float = 0.8,
    ):
        self.max_features = max_features
        self.svm_regularization = svm_regularization
        self.epochs = epochs
        self.ridge = ridge
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.state_w: np.ndarray | None = None
        self.state_b: np.ndarray | None = None
        self.level_w: np.ndarray | None = None
        self.level_cutpoints: np.ndarray | None = None

    def fit_vectorizer(self, texts: list[str]) -> None:
        document_frequency: Counter[str] = Counter()
        total_frequency: Counter[str] = Counter()
        for text in texts:
            tokens = tokenize(text)
            total_frequency.update(tokens)
            document_frequency.update(set(tokens))
        candidates = [token for token, df in document_frequency.items() if df >= 2]
        candidates.sort(key=lambda token: (-total_frequency[token], token))
        self.vocab = {token: index for index, token in enumerate(candidates[: self.max_features])}
        n = len(texts)
        self.idf = np.ones(len(self.vocab), dtype=float)
        for token, index in self.vocab.items():
            self.idf[index] = math.log((1 + n) / (1 + document_frequency[token])) + 1

    def transform(self, texts: list[str]) -> np.ndarray:
        assert self.idf is not None
        matrix = np.zeros((len(texts), len(self.vocab)), dtype=float)
        for row_index, text in enumerate(texts):
            counts = Counter(tokenize(text))
            for token, count in counts.items():
                index = self.vocab.get(token)
                if index is not None:
                    matrix[row_index, index] = (1 + math.log(count)) * self.idf[index]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1.0, norms)

    def fit(self, rows: list[dict[str, Any]], references: list[dict[str, Any]]) -> None:
        texts = [record_text(row) for row in rows]
        self.fit_vectorizer(texts)
        x = self.transform(texts)
        labels = np.array([STATE_ORDER.index(reference_state(item)) for item in references])
        rng = np.random.default_rng(SEED)
        w = rng.normal(0, 0.01, (3, x.shape[1]))
        b = np.zeros(3)
        for epoch in range(self.epochs):
            lr = 0.12 / math.sqrt(1 + epoch / 40)
            for cls in range(3):
                y = np.where(labels == cls, 1.0, -1.0)
                margin = y * (x @ w[cls] + b[cls])
                active = margin < 1
                grad_w = self.svm_regularization * w[cls]
                grad_b = 0.0
                if active.any():
                    grad_w -= (x[active].T @ y[active]) / len(rows)
                    grad_b -= float(y[active].sum()) / len(rows)
                w[cls] -= lr * grad_w
                b[cls] -= lr * grad_b
        self.state_w, self.state_b = w, b
        e_rows = [
            index for index, reference in enumerate(references)
            if reference_state(reference) == "E" and reference_level(reference) is not None
        ]
        xe = x[e_rows]
        ye = np.array([int(reference_level(references[index])) for index in e_rows], dtype=int)
        if not len(xe):
            raise RuntimeError("B2 ordinal model requires development records labelled E")





        empirical = np.array([
            (float(np.sum(ye <= level)) + 0.5) / (len(ye) + 1.0)
            for level in (1, 2, 3)
        ])
        cutpoints = np.log(empirical / (1.0 - empirical))
        level_w = np.zeros(xe.shape[1], dtype=float)
        first_w = np.zeros_like(level_w)
        second_w = np.zeros_like(level_w)
        first_cut = np.zeros(3, dtype=float)
        second_cut = np.zeros(3, dtype=float)
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        regularization = self.ridge / len(xe)
        for step in range(1, self.epochs + 1):
            eta = xe @ level_w
            z = np.clip(cutpoints[None, :] - eta[:, None], -30.0, 30.0)
            cumulative = 1.0 / (1.0 + np.exp(-z))
            density = cumulative * (1.0 - cumulative)
            lower = np.where(ye > 1, cumulative[np.arange(len(ye)), ye - 2], 0.0)
            upper = np.where(ye <= 3, cumulative[np.arange(len(ye)), np.minimum(ye - 1, 2)], 1.0)
            lower_density = np.where(ye > 1, density[np.arange(len(ye)), ye - 2], 0.0)
            upper_density = np.where(ye <= 3, density[np.arange(len(ye)), np.minimum(ye - 1, 2)], 0.0)
            probability = np.clip(upper - lower, 1e-9, 1.0)

            gradient_eta = (upper_density - lower_density) / probability
            gradient_w = xe.T @ gradient_eta / len(xe) + regularization * level_w
            gradient_cutpoints = np.zeros(3, dtype=float)
            for boundary in range(3):
                upper_mask = ye == boundary + 1
                lower_mask = ye == boundary + 2
                if upper_mask.any():
                    gradient_cutpoints[boundary] -= float(
                        np.sum(density[upper_mask, boundary] / probability[upper_mask])
                    ) / len(xe)
                if lower_mask.any():
                    gradient_cutpoints[boundary] += float(
                        np.sum(density[lower_mask, boundary] / probability[lower_mask])
                    ) / len(xe)

            norm = float(np.linalg.norm(gradient_w))
            if norm > 5.0:
                gradient_w *= 5.0 / norm
            gradient_cutpoints = np.clip(gradient_cutpoints, -5.0, 5.0)
            first_w = beta1 * first_w + (1.0 - beta1) * gradient_w
            second_w = beta2 * second_w + (1.0 - beta2) * gradient_w**2
            first_cut = beta1 * first_cut + (1.0 - beta1) * gradient_cutpoints
            second_cut = beta2 * second_cut + (1.0 - beta2) * gradient_cutpoints**2
            correction1, correction2 = 1.0 - beta1**step, 1.0 - beta2**step
            learning_rate = 0.035
            level_w -= learning_rate * (first_w / correction1) / (
                np.sqrt(second_w / correction2) + epsilon
            )
            cutpoints -= learning_rate * (first_cut / correction1) / (
                np.sqrt(second_cut / correction2) + epsilon
            )
            for boundary in range(1, 3):
                cutpoints[boundary] = max(cutpoints[boundary], cutpoints[boundary - 1] + 0.05)
        self.level_w = level_w
        self.level_cutpoints = cutpoints

    def predict(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        assert (
            self.state_w is not None and self.state_b is not None
            and self.level_w is not None and self.level_cutpoints is not None
        )
        x = self.transform([record_text(row) for row in rows])
        scores = x @ self.state_w.T + self.state_b
        shifted = scores - scores.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        states = probabilities.argmax(axis=1)
        eta = x @ self.level_w
        cumulative = 1.0 / (
            1.0 + np.exp(-np.clip(self.level_cutpoints[None, :] - eta[:, None], -30.0, 30.0))
        )
        level_probabilities = np.column_stack([
            cumulative[:, 0],
            cumulative[:, 1] - cumulative[:, 0],
            cumulative[:, 2] - cumulative[:, 1],
            1.0 - cumulative[:, 2],
        ])
        levels = np.argmax(level_probabilities, axis=1) + 1
        return [{"state": STATE_ORDER[int(states[i])], "level": int(levels[i]),
                 "confidence": float(probabilities[i, states[i]])} for i in range(len(rows))]

    def save(self, path: Path) -> None:
        assert (
            self.idf is not None and self.state_w is not None and self.state_b is not None
            and self.level_w is not None and self.level_cutpoints is not None
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, idf=self.idf, state_w=self.state_w, state_b=self.state_b,
                            level_w=self.level_w, level_cutpoints=self.level_cutpoints,
                            tokens=np.array(sorted(self.vocab, key=self.vocab.get), dtype=object),
                            max_features=np.array([self.max_features]),
                            svm_regularization=np.array([self.svm_regularization]),
                            epochs=np.array([self.epochs]), ridge=np.array([self.ridge]))


def state_macro_f1(gold: list[str], predicted: list[str]) -> float:
    values = []
    for label in STATE_ORDER:
        true_positive = sum(a == label and b == label for a, b in zip(gold, predicted))
        false_positive = sum(a != label and b == label for a, b in zip(gold, predicted))
        false_negative = sum(a == label and b != label for a, b in zip(gold, predicted))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return statistics.mean(values)


def qwk_score(gold: list[int], predicted: list[int]) -> float:
    if not gold:
        return 0.0
    observed = np.zeros((4, 4), dtype=float)
    for actual, estimate in zip(gold, predicted):
        observed[actual - 1, estimate - 1] += 1
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    weights = np.fromfunction(lambda i, j: ((i - j) ** 2) / 9.0, (4, 4))
    denominator = float((weights * expected).sum())
    return 1.0 - float((weights * observed).sum()) / denominator if denominator else 0.0


def development_cluster(reference: dict[str, Any]) -> str:
    for field in ("teacher_cluster", "split_cluster", "cluster_id"):
        value = str(reference.get(field) or "").strip()
        if value:
            return value
    raise RuntimeError(
        f"Development reference {reference.get('sample_id')} lacks teacher_cluster/split_cluster"
    )


def adjusted_macro_f1_from_confusion(confusion: list[list[int]]) -> float:
    values = []
    for index in range(len(STATE_ORDER)):
        true_positive = confusion[index][index]
        predicted_total = sum(row[index] for row in confusion)
        actual_total = sum(confusion[index])
        precision = jeffreys_mean(true_positive, predicted_total)
        recall = jeffreys_mean(true_positive, actual_total)
        values.append(2.0 * precision * recall / (precision + recall))
    return statistics.mean(values)


def b1_cv_fold_result(
    rows: list[dict[str, Any]], references: list[dict[str, Any]], config: dict[str, Any],
) -> tuple[dict[str, Any], float, float]:
    predictions = [b1_assess(row, config) for row in rows]
    gold_states = [reference_state(reference) for reference in references]
    predicted_states = [str(item["state"]) for item in predictions]
    state_confusion = [[0 for _ in STATE_ORDER] for _ in STATE_ORDER]
    for actual, estimate in zip(gold_states, predicted_states, strict=True):
        state_confusion[STATE_ORDER.index(actual)][STATE_ORDER.index(estimate)] += 1
    level_pairs = [
        (int(reference_level(reference)), int(prediction["level"]))
        for reference, prediction in zip(references, predictions, strict=True)
        if reference_state(reference) == "E"
        and prediction["state"] == "E"
        and reference_level(reference) is not None
        and prediction["level"] is not None
    ]
    level_confusion = [[0 for _ in range(4)] for _ in range(4)]
    for actual, estimate in level_pairs:
        level_confusion[actual - 1][estimate - 1] += 1
    raw_macro = state_macro_f1(gold_states, predicted_states)
    raw_qwk = qwk_score(
        [actual for actual, _ in level_pairs], [estimate for _, estimate in level_pairs]
    )
    eligible = sum(state == "E" for state in gold_states)
    display = {
        "validation_records": len(rows),
        "state_confusion": {
            "labels": list(STATE_ORDER), "matrix": state_confusion,
        },
        "state_macro_f1": adjusted_macro_f1_from_confusion(state_confusion),
        "conditional_qwk": raw_qwk * len(level_pairs) / (len(level_pairs) + 1.0),
        "conditional_level_confusion": {
            "labels": [1, 2, 3, 4], "matrix": level_confusion,
        },
        "conditional_coverage": jeffreys_mean(len(level_pairs), eligible),
        "conditional_coverage_counts": {
            "eligible_level_predictions": len(level_pairs),
            "gold_estimable_total": eligible,
        },
    }
    return display, raw_macro, raw_qwk


def select_b1_config(
    rows: list[dict[str, Any]], references: list[dict[str, Any]], folds: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:

    candidates = [
        {
            "weight_profile": profile,
            "support_threshold": support_threshold,
            "evidence_quantity_threshold": quantity_threshold,
            "conflict_threshold": conflict_threshold,
        }
        for profile in ("strict", "balanced", "flat")
        for support_threshold, quantity_threshold, conflict_threshold in (
            (0.45, 0.80, 0.35),
            (0.60, 1.00, 0.55),
            (0.75, 1.20, 0.75),
        )
    ]
    groups = sorted(
        {development_cluster(reference) for reference in references},
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    if len(groups) < folds:
        raise RuntimeError(f"Grouped CV requires at least {folds} development clusters")
    fold_by_group = {group: index % folds for index, group in enumerate(groups)}
    evaluations: list[dict[str, Any]] = []
    selection_values: list[tuple[float, float, float, int]] = []
    for candidate_index, candidate in enumerate(candidates):
        fold_results = []
        raw_macros = []
        raw_qwks = []
        for fold in range(folds):
            validation_indices = [
                index for index, reference in enumerate(references)
                if fold_by_group[development_cluster(reference)] == fold
            ]
            display, raw_macro, raw_qwk = b1_cv_fold_result(
                [rows[index] for index in validation_indices],
                [references[index] for index in validation_indices],
                candidate,
            )
            display["fold"] = fold
            display["validation_clusters"] = sum(
                value == fold for value in fold_by_group.values()
            )
            fold_results.append(display)
            raw_macros.append(raw_macro)
            raw_qwks.append(raw_qwk)
        mean_raw_macro = statistics.mean(raw_macros)
        mean_raw_qwk = statistics.mean(raw_qwks)
        selection_values.append((
            mean_raw_macro + 0.15 * max(-1.0, mean_raw_qwk),
            mean_raw_macro, mean_raw_qwk, -candidate_index,
        ))
        evaluations.append({
            "config": candidate,
            "folds": fold_results,
            "mean_state_macro_f1": statistics.mean(
                item["state_macro_f1"] for item in fold_results
            ),
            "mean_conditional_qwk": statistics.mean(
                item["conditional_qwk"] for item in fold_results
            ),
        })
    winner_index = max(range(len(candidates)), key=lambda index: selection_values[index])
    selected = dict(candidates[winner_index])
    detail = {
        "artifact_type": "b1_rule_calibration",
        "version": "M1-B1-CAL-2026-11-19.1",
        "development_only": True,
        "group_field": "teacher_cluster|split_cluster|cluster_id",
        "folds": folds,
        "clusters": len(groups),
        "records": len(rows),
        "selection_seed": SEED,
        "group_assignment_rule": "sha256_sorted_round_robin",
        "cluster_fold_assignment": fold_by_group,
        "selection_rule": "raw mean state macro-F1 + 0.15 * raw conditional QWK",
        "displayed_proportions": (
            "Jeffreys posterior means; QWK shrunk toward zero by n/(n+1); "
            "selection uses raw metrics reconstructed from retained confusion counts"
        ),
        "indicator_material_mapping": B1_PRIMARY_MATERIAL,
        "weight_profiles": B1_WEIGHT_PROFILES,
        "candidates": evaluations,
        "selected_config": selected,
        "development_input_sha256": sha256(DATA / "development_records.jsonl"),
        "development_reference_sha256": sha256(DATA / "development_reference.jsonl"),
    }
    write_json(DATA / "b1_rule_calibration.json", detail)
    return selected, detail


def select_b2_config(
    rows: list[dict[str, Any]], references: list[dict[str, Any]], folds: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = [
        {"max_features": 1200, "svm_regularization": 0.01, "epochs": 300, "ridge": 0.5},
        {"max_features": 1200, "svm_regularization": 0.02, "epochs": 450, "ridge": 0.8},
        {"max_features": 1800, "svm_regularization": 0.01, "epochs": 300, "ridge": 0.8},
        {"max_features": 1800, "svm_regularization": 0.02, "epochs": 450, "ridge": 0.5},
    ]
    groups = sorted(
        {development_cluster(reference) for reference in references},
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    if len(groups) < folds:
        raise RuntimeError(f"Grouped CV requires at least {folds} development clusters")
    fold_by_group = {group: index % folds for index, group in enumerate(groups)}
    evaluations = []
    for candidate in candidates:
        fold_results = []
        for fold in range(folds):
            train_indices = [
                index for index, reference in enumerate(references)
                if fold_by_group[development_cluster(reference)] != fold
            ]
            validation_indices = [
                index for index, reference in enumerate(references)
                if fold_by_group[development_cluster(reference)] == fold
            ]
            model = TfidfLinearModel(**candidate)
            model.fit([rows[index] for index in train_indices],
                      [references[index] for index in train_indices])
            predictions = model.predict([rows[index] for index in validation_indices])
            gold_states = [reference_state(references[index]) for index in validation_indices]
            predicted_states = [str(item["state"]) for item in predictions]
            level_pairs = [
                (reference_level(references[index]), int(prediction["level"]))
                for index, prediction in zip(validation_indices, predictions, strict=True)
                if reference_state(references[index]) == "E"
                and prediction["state"] == "E"
                and reference_level(references[index]) is not None
            ]
            estimable_total = sum(state == "E" for state in gold_states)
            fold_results.append({
                "fold": fold, "validation_clusters": sum(1 for value in fold_by_group.values() if value == fold),
                "validation_records": len(validation_indices),
                "state_macro_f1": state_macro_f1(gold_states, predicted_states),
                "conditional_qwk": qwk_score(
                    [int(pair[0]) for pair in level_pairs], [pair[1] for pair in level_pairs]
                ),
                "conditional_coverage": jeffreys_mean(len(level_pairs), estimable_total),
                "conditional_coverage_counts": {
                    "eligible_level_predictions": len(level_pairs),
                    "gold_estimable_total": estimable_total,
                },
            })
        mean_macro = statistics.mean(item["state_macro_f1"] for item in fold_results)
        mean_qwk = statistics.mean(item["conditional_qwk"] for item in fold_results)
        evaluations.append({
            "config": candidate, "folds": fold_results,
            "mean_state_macro_f1": mean_macro, "mean_conditional_qwk": mean_qwk,
            "selection_score": mean_macro + 0.15 * max(-1.0, mean_qwk),
        })
    winner = max(
        evaluations,
        key=lambda item: (item["selection_score"], item["mean_state_macro_f1"],
                          item["mean_conditional_qwk"], -item["config"]["max_features"]),
    )
    detail = {
        "group_field": "teacher_cluster|split_cluster|cluster_id",
        "folds": folds, "clusters": len(groups), "selection_seed": SEED,
        "group_assignment_rule": "sha256_sorted_round_robin",
        "cluster_fold_assignment": fold_by_group,
        "displayed_proportions": "Jeffreys posterior means; raw numerators and denominators retained",
        "selection_rule": "mean_macro_f1 + 0.15 * conditional_qwk",
        "candidates": evaluations, "selected_config": winner["config"],
    }
    write_json(DATA / "b2_grouped_cv.json", detail)
    return dict(winner["config"]), detail


CONTRADICTION_CUES = (
    "未发现", "尚未建立", "与已提交结论相反", "相反",
    "not found", "not established", "contradict",
)





B1_PRIMARY_MATERIAL = {
    "T_ALIGN": "lesson_plan",
    "T_FEEDBACK": "lesson_plan",
    "D_DATA": "courseware",
    "D_RESOURCE": "courseware",
    "R_QUESTION": "project",
    "R_COMM": "project",
    "A_WORKFLOW": "teaching_outcome",
    "A_RESP": "teaching_outcome",
}
B1_WEIGHT_PROFILES: dict[str, dict[str, dict[str, float]]] = {
    "strict": {
        "material": {"primary": 1.00, "secondary": 0.55, "other": 0.25},
        "contribution": {
            "sole": 1.00, "lead": 1.00, "substantial": 1.00,
            "limited": 0.60, "unclear": 0.35,
        },
    },
    "balanced": {
        "material": {"primary": 1.00, "secondary": 0.70, "other": 0.35},
        "contribution": {
            "sole": 1.00, "lead": 1.00, "substantial": 1.00,
            "limited": 0.70, "unclear": 0.45,
        },
    },
    "flat": {
        "material": {"primary": 1.00, "secondary": 0.85, "other": 0.55},
        "contribution": {
            "sole": 1.00, "lead": 1.00, "substantial": 1.00,
            "limited": 0.80, "unclear": 0.55,
        },
    },
}


def evidence_unit(doc: dict[str, Any], observable: str) -> dict[str, Any]:
    start = doc["text"].find(observable)
    if start < 0:
        start = 0
    left, right = max(0, start - 70), min(len(doc["text"]), start + len(observable) + 90)
    return {
        "doc_id": doc["doc_id"], "locator": f"chars:{left}-{right}",
        "quote": doc["text"][left:right], "observable": observable,
    }


def b1_document_weight(
    row: dict[str, Any], document: dict[str, Any], config: dict[str, Any],
) -> float:
    profile = B1_WEIGHT_PROFILES[str(config["weight_profile"])]
    primary = B1_PRIMARY_MATERIAL.get(row["indicator_id"])
    material_type = str(document.get("material_type") or "other")
    category = (
        "primary" if material_type == primary
        else "other" if material_type == "other" else "secondary"
    )
    material_weight = profile["material"][category]
    contribution = str(document.get("contribution_level") or "unclear")
    contribution_weight = profile["contribution"].get(
        contribution, profile["contribution"]["unclear"]
    )
    return float(material_weight * contribution_weight)


def b1_weighted_evidence(
    row: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:

    critical = set(row["critical_observables"])
    support_scores: Counter[str] = Counter()
    contradiction_scores: Counter[str] = Counter()
    support_docs: list[str] = []
    units: list[dict[str, Any]] = []
    refs: list[str] = []
    anchor_votes: Counter[int] = Counter()
    evidence_quantity = 0.0
    for document in row["documents"]:
        if has_high_risk_flag(scan_text(document["text"])):
            continue
        weight = b1_document_weight(row, document, config)




        for level, phrase in LEVEL_PHRASES.items():
            if phrase in document["text"]:
                anchor_votes[level] += weight
        for level, phrase in LEVEL_PARAPHRASES.items():
            if phrase in document["text"]:
                anchor_votes[level] += 0.80 * weight

        found = {observable for observable in critical if observable in document["text"]}
        if not found:
            continue
        lowered = document["text"].casefold()
        contradiction = any(cue in lowered for cue in CONTRADICTION_CUES)
        score_map = contradiction_scores if contradiction else support_scores
        for observable in found:
            score_map[observable] += weight
        if not contradiction:
            support_docs.append(document["doc_id"])
            evidence_quantity += weight
        refs.append(document["source_ref"] or f"material:{document['doc_id']}")
        units.extend(evidence_unit(document, observable) for observable in sorted(found))
    return {
        "support_scores": dict(support_scores),
        "contradiction_scores": dict(contradiction_scores),
        "support_docs": support_docs,
        "evidence_quantity": evidence_quantity,
        "anchor_votes": dict(anchor_votes),
        "evidence_units": units,
        "evidence_doc_ids": list(dict.fromkeys(unit["doc_id"] for unit in units)),
        "evidence_families": list(dict.fromkeys(refs)),
    }


def b1_assess(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    evidence = b1_weighted_evidence(row, config)
    critical = set(row["critical_observables"])
    support_threshold = float(config["support_threshold"])
    conflict_threshold = float(config["conflict_threshold"])
    support_scores = evidence["support_scores"]
    contradiction_scores = evidence["contradiction_scores"]
    conflict = any(
        float(support_scores.get(observable, 0.0)) >= support_threshold
        and float(contradiction_scores.get(observable, 0.0)) >= conflict_threshold
        for observable in critical
    )
    sufficient = (
        all(float(support_scores.get(observable, 0.0)) >= support_threshold for observable in critical)
        and float(evidence["evidence_quantity"]) >= float(config["evidence_quantity_threshold"])
    )
    state = "C" if conflict else "E" if sufficient else "I"
    votes = {int(level): float(value) for level, value in evidence["anchor_votes"].items()}
    level = min(votes, key=lambda value: (-votes[value], value)) if state == "E" and votes else None
    if state == "E" and level is None:
        level = 1
    support_margin = min(
        (float(support_scores.get(observable, 0.0)) / max(support_threshold, 1e-9)
         for observable in critical),
        default=0.0,
    )
    conflict_margin = max(
        (float(contradiction_scores.get(observable, 0.0)) / max(conflict_threshold, 1e-9)
         for observable in critical),
        default=0.0,
    )
    if state == "C":
        confidence = 0.58 + 0.18 * min(2.0, conflict_margin)
    elif state == "E":
        confidence = 0.54 + 0.20 * min(2.0, support_margin)
    else:
        confidence = 0.58 + 0.16 * max(0.0, 1.0 - min(1.0, support_margin))
    return {
        "state": state, "level": level, "confidence": min(0.94, confidence),
        "evidence": evidence,
    }


def rule_evidence(row: dict[str, Any]) -> dict[str, Any]:
    critical = set(row["critical_observables"])
    supports: set[str] = set()
    contradictions: set[str] = set()
    support_docs: list[str] = []
    units: list[dict[str, Any]] = []
    refs: list[str] = []
    for doc in row["documents"]:
        if has_high_risk_flag(scan_text(doc["text"])):
            continue
        found = {observable for observable in critical if observable in doc["text"]}
        if not found:
            continue
        lowered = doc["text"].casefold()
        contradiction = any(cue in lowered for cue in CONTRADICTION_CUES)
        (contradictions if contradiction else supports).update(found)
        if not contradiction:
            support_docs.append(doc["doc_id"])
        refs.append(doc["source_ref"] or f"material:{doc['doc_id']}")
        units.extend(evidence_unit(doc, observable) for observable in sorted(found))
    return {
        "support_observables": supports,
        "contradiction_observables": contradictions,
        "support_docs": support_docs,
        "evidence_units": units,
        "evidence_doc_ids": list(dict.fromkeys(unit["doc_id"] for unit in units)),
        "evidence_families": list(dict.fromkeys(refs)),
    }


def bm25_resources(row: dict[str, Any], library: ResourceLibrary, limit: int = 2) -> list[dict[str, Any]]:
    query = " ".join([
        row["indicator_id"], row["indicator_title"], row["dimension"], " ".join(row.get("goals", []))
    ])
    terms = Counter(retrieval_tokenize(query))
    lengths = [sum(counts.values()) for counts in library.term_counts]
    average = statistics.mean(lengths) if lengths else 1.0
    scored = []
    for resource, counts, length in zip(library.resources, library.term_counts, lengths, strict=True):
        if not resource.get("accessible") or not str(resource.get("license") or "").strip():
            continue
        score = 0.0
        for term, query_frequency in terms.items():
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * length / average)
            score += library.idf.get(term, 0.0) * frequency * 2.5 / denominator * min(query_frequency, 2)
        if score > 0:
            scored.append((score, str(resource["id"]), resource))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [resource for _, _, resource in scored[:limit]]


def resource_view(resource: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": resource["id"], "title_en": str(resource.get("title") or ""),
        "title_zh": str(resource.get("title_zh") or ""),
        "summary_en": str(resource.get("summary_en") or ""),
        "summary_zh": str(resource.get("summary_zh") or ""),
        "prerequisites": [str(item) for item in resource.get("prerequisites", [])],
        "url": str(resource.get("url") or ""), "license": str(resource.get("license") or ""),
        "duration_minutes": int(resource.get("duration_minutes") or 30),
        "safe_when_unknown": bool(resource.get("safe_when_unknown")),
    }


def baseline_path(
    row: dict[str, Any], action: str, library: ResourceLibrary, *, mode: str, level: int | None,
) -> dict[str, Any]:
    if action == "human_review":
        return {"route": action, "resource_ids": [], "resources": [], "steps": [],
                "mode": "paused", "target_match": True, "prerequisite_valid": True,
                "source_valid": True, "safe_when_unknown": True}
    if mode == "bm25":
        selected = bm25_resources(row, library)
    elif mode == "governed":
        query = " ".join([row["indicator_id"], row["indicator_title"], row["dimension"], action])
        selected = library.search(
            query, indicator_id=row["indicator_id"], dimension=row["dimension"],
            safe_when_unknown=action == "request_evidence", language="zh",
            target_difficulty="foundation" if action == "request_evidence" or int(level or 1) <= 2 else "intermediate",
            limit=2,
        )
    else:
        selected = [item for item in library.resources if row["indicator_id"] in item.get("indicator_ids", [])][:2]
    ids = [str(item["id"]) for item in selected]
    if mode == "governed" and ids:
        allowed: Callable[[dict[str, Any]], bool] = lambda item: bool(
            item.get("accessible") and str(item.get("license") or "").strip()
            and (action != "request_evidence" or item.get("safe_when_unknown"))
        )
        while ids:
            try:
                closure = library.prerequisite_closure(ids, allowed=allowed)
            except ValueError:
                ids.pop()
                continue
            if len(closure) <= 4:
                ids = closure
                break
            ids.pop()
        selected = [library.by_id[item] for item in ids]
    seen: set[str] = set()
    prerequisite_valid = True
    for item in selected:
        if any(str(prerequisite) not in seen for prerequisite in item.get("prerequisites", [])):
            prerequisite_valid = False
        seen.add(str(item["id"]))
    steps = [{
        "resource_id": item["id"],
        "objective": {"en": "Produce observable evidence before interpretation." if action == "request_evidence" else "Connect the source to the stated goal.",
                      "zh": "先形成可观察证据，再作解释。" if action == "request_evidence" else "将资源与已陈述目标相连。"},
        "reason": {"en": "Selected for the target indicator and evidence state.", "zh": "依据目标指标和证据状态选择。"},
        "practice_task": {"en": "Apply one idea to a current work sample and retain the result.", "zh": "将一个要点用于当前工作样例并保留结果。"},
        "completion_evidence": {"en": "A dated work sample and rationale.", "zh": "带日期的工作样例和理由。"},
        "duration_minutes": int(item.get("duration_minutes") or 30),
    } for item in selected]
    return {
        "route": action, "resource_ids": ids, "resources": [resource_view(item) for item in selected],
        "steps": steps, "mode": {"bm25": "bm25", "governed": "governed_catalog"}.get(mode, "rule_catalog"),
        "target_match": bool(selected) and all(row["indicator_id"] in item.get("indicator_ids", []) for item in selected),
        "prerequisite_valid": prerequisite_valid,
        "source_valid": bool(selected) and all(item.get("accessible") and item.get("url") and item.get("license") for item in selected),
        "safe_when_unknown": action != "request_evidence" or (bool(selected) and all(item.get("safe_when_unknown") for item in selected)),
    }


def b1_predict(
    row: dict[str, Any], library: ResourceLibrary, config: dict[str, Any],
) -> dict[str, Any]:
    assessment = b1_assess(row, config)
    evidence = assessment["evidence"]
    state = str(assessment["state"])
    level = int(assessment["level"]) if assessment["level"] is not None else None
    action = action_for(state, level)
    return {"state": state, "level": level, "action": action,
            "confidence": round(float(assessment["confidence"]), 6),
            "evidence_families": evidence["evidence_families"],
            "evidence_doc_ids": evidence["evidence_doc_ids"],
            "evidence_units": evidence["evidence_units"],
            "path": baseline_path(row, action, library, mode="rule", level=level),
            "reason_codes": ["development_selected_indicator_material_weighted_rule"]}


def b2_outputs(model: TfidfLinearModel, rows: list[dict[str, Any]], library: ResourceLibrary) -> list[dict[str, Any]]:
    predictions = model.predict(rows)
    outputs = []
    for row, predicted in zip(rows, predictions, strict=True):
        state = str(predicted["state"])
        level = int(predicted["level"]) if state == "E" else None
        action = action_for(state, level)
        evidence = rule_evidence(row)
        outputs.append({"state": state, "level": level, "action": action,
                        "confidence": round(float(predicted["confidence"]), 6),
                        "evidence_families": evidence["evidence_families"],
                        "evidence_doc_ids": evidence["evidence_doc_ids"],
                        "evidence_units": evidence["evidence_units"],
                        "path": baseline_path(row, action, library, mode="bm25", level=level),
                        "reason_codes": ["tfidf_linear_svm_cumulative_ordinal_logit_bm25"]})
    return outputs


def materials_for(row: dict[str, Any]) -> list[dict[str, Any]]:
    materials = []
    for doc in row["documents"]:
        flags = scan_text(doc["text"])
        materials.append({
            "id": doc["doc_id"], "title": doc["title"], "material_type": doc["material_type"],
            "source_ref": doc["source_ref"], "source_date": doc["source_date"],
            "contribution_level": doc["contribution_level"],
            "contribution_note": doc.get("contribution_note", ""),
            "sha256": hashlib.sha256(doc["text"].encode("utf-8")).hexdigest(),
            "safety_flags": flags,
            "chunks": [{"id": f"{doc['doc_id']}-C1", "material_id": doc["doc_id"],
                        "text": doc["text"], "locator": f"chars:0-{len(doc['text'])}"}],
        })
    return materials


def qwen_evidence(
    row: dict[str, Any], qwen: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    materials = materials_for(row)
    material_map = {item["id"]: item for item in materials}
    allowed = set(row["critical_observables"])
    if row.get("other_observable"):
        allowed.add(row["other_observable"])
    raw_items, extraction_confidences = [], []

    def contribution_clarity(material: dict[str, Any]) -> str:
        contribution = material["contribution_level"]
        return (
            "clear" if contribution in {"sole", "lead", "substantial"}
            else "partial" if contribution == "limited" else "unclear"
        )

    for item in qwen.get("evidence", []):
        if not isinstance(item, dict):
            continue
        material = material_map.get(str(item.get("doc_id") or ""))
        if material is None or has_high_risk_flag(material["safety_flags"]):
            continue
        observables = item.get("observables") if isinstance(item.get("observables"), list) else []
        for observable in observables:
            if observable not in allowed:
                continue
            raw_items.append({
                "material_id": material["id"], "chunk_id": f"{material['id']}-C1",
                "quote": str(item.get("quote") or ""), "indicator_id": row["indicator_id"],
                "observable": observable, "polarity": item.get("polarity", "context"),
                "claim": str(item.get("quote") or ""),
                "achievement_key": str(item.get("family_key") or ""),
                "family_hint": str(item.get("family_key") or ""),
                "contribution_clarity": contribution_clarity(material),
            })
            try:
                extraction_confidences.append(float(item.get("extraction_confidence", 0.5)))
            except (TypeError, ValueError):
                extraction_confidences.append(0.5)







    critical = set(row["critical_observables"])
    existing_counter = {
        (str(item.get("material_id") or ""), str(item.get("observable") or ""))
        for item in raw_items if item.get("polarity") == "contradict"
    }
    document_map = {doc["doc_id"]: doc for doc in row["documents"]}
    for material in materials:
        if has_high_risk_flag(material["safety_flags"]):
            continue
        document = document_map[material["id"]]
        text = document["text"]
        lowered = text.casefold()
        cue_locations = [
            (lowered.find(cue.casefold()), cue)
            for cue in CONTRADICTION_CUES if lowered.find(cue.casefold()) >= 0
        ]
        if not cue_locations:
            continue
        cue_start, cue = min(cue_locations, key=lambda item: item[0])
        for observable in sorted(critical):
            observable_start = text.find(observable)
            key = (material["id"], observable)
            if observable_start < 0 or key in existing_counter:
                continue
            left = max(0, min(observable_start, cue_start) - 80)
            right = min(
                len(text),
                max(observable_start + len(observable), cue_start + len(cue)) + 120,
            )
            quote = text[left:right].strip()
            source_ref = str(document.get("source_ref") or "")
            raw_items.append({
                "material_id": material["id"], "chunk_id": f"{material['id']}-C1",
                "quote": quote, "indicator_id": row["indicator_id"],
                "observable": observable, "polarity": "contradict", "claim": quote,
                "achievement_key": source_ref, "family_hint": source_ref,
                "contribution_clarity": contribution_clarity(material),
            })
            extraction_confidences.append(0.90)
            existing_counter.add(key)
    valid, rejected = validate_evidence(raw_items, materials, {row["indicator_id"]: allowed})
    return valid, rejected, extraction_confidences


def singleton_families(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:

    families = []
    for item in evidence:
        digest = hashlib.sha256(str(item["id"]).encode("utf-8")).hexdigest()[:12]
        families.append({
            "id": f"nofam-{digest}", "canonical_identity": f"claim:{item['id']}",
            "indicator_ids": [item["indicator_id"]], "member_ids": [item["id"]],
            "members": [item], "independent_source_keys": [item["source_key"]],
            "event_dates": [str(item["event_date"])] if item.get("event_date") else [],
            "recency_bands": [item["recency_band"]],
            "contribution_levels": [item["contribution_level"]],
            "contribution_notes": [item["contribution_note"]] if item.get("contribution_note") else [],
            "max_contribution_factor": item["contribution_factor"],
            "all_security_restricted": item["security_restricted"],
            "merge_reasons": ["family_deduplication_disabled"],
            "family_hints": [item["family_hint"]] if item.get("family_hint") else [],
            "requires_review": False, "candidate_related_family_ids": [], "member_count": 1,
        })
    return sorted(families, key=lambda item: item["id"])


def apply_family_carrier_policy(
    row: dict[str, Any], evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:

    structured_carriers = {"lesson_plan", "courseware", "project"}
    critical = set(row["critical_observables"])
    documents = {doc["doc_id"]: doc for doc in row["documents"]}

    supportive_sources: set[str] = set()
    for item in evidence:
        document = documents.get(str(item.get("material_id") or ""))
        if not document:
            continue
        title = str(document.get("title") or "").casefold()
        source_ref = str(document.get("source_ref") or "").strip()
        if (
            source_ref
            and "副本" not in title and "copy" not in title
            and item.get("polarity") == "support"
            and item.get("observable") in critical
        ):
            supportive_sources.add(source_ref)

    retained: list[dict[str, Any]] = []
    excluded_ids: list[str] = []
    for item in evidence:
        document = documents.get(str(item.get("material_id") or ""))
        if not document:
            retained.append(item)
            continue
        title = str(document.get("title") or "").casefold()
        source_ref = str(document.get("source_ref") or "").strip()
        is_explicit_copy = "副本" in title or "copy" in title
        is_removable_counterclaim = (
            item.get("polarity") == "contradict"
            and item.get("observable") in critical
            and str(document.get("material_type") or "") in structured_carriers
            and is_explicit_copy
            and source_ref in supportive_sources
        )
        if is_removable_counterclaim:
            excluded_ids.append(str(item.get("id") or ""))
        else:
            retained.append(item)
    return retained, excluded_ids


def p_predict(
    row: dict[str, Any], qwen: dict[str, Any], indicator: dict[str, Any], engine: RatabilityEngine,
    library: ResourceLibrary, *, deduplicate_families: bool,
) -> dict[str, Any]:
    valid, rejected, extraction_confidences = qwen_evidence(row, qwen)
    if deduplicate_families:
        family_evidence, carrier_policy_excluded = apply_family_carrier_policy(row, valid)
        families = build_evidence_families(family_evidence)
    else:
        family_evidence, carrier_policy_excluded = valid, []
        families = singleton_families(family_evidence)
    family_ids = [family["id"] for family in families if row["indicator_id"] in family["indicator_ids"]]
    generic = qwen.get("generic") if isinstance(qwen.get("generic"), dict) else {}
    try:
        fallback_level = int(generic.get("level") or 1)
    except (TypeError, ValueError):
        fallback_level = 1
    level = target_support_level(row, valid, fallback_level)
    if not deduplicate_families:
        supporting = [
            family for family in families
            if any(member.get("polarity") == "support" for member in family.get("members", []))
        ]
        source_keys = {key for family in supporting for key in family.get("independent_source_keys", [])}
        if len(supporting) > len(source_keys):
            level = min(4, level + 1)
    raw = {
        "indicator_id": row["indicator_id"], "applicable": True,
        "proposed_state": "estimable", "family_ids": family_ids,
        "level": level, "level_interval": [level, level],
        "missing_evidence": [], "rationale": {"en": "Evidence-bounded decision.", "zh": "基于现有证据的判断。"},
    }
    decision = engine.decide(indicator, raw, families)
    state = {"estimable": "E", "insufficient": "I", "conflict": "C"}.get(decision["state"], "I")
    output_level = int(decision["level"]) if state == "E" and decision.get("level") else None
    action = action_for(state, output_level)
    selected_keys: list[str] = []
    selected_doc_ids: list[str] = []
    selected_units: list[dict[str, Any]] = []
    selected_ids = set(decision.get("family_ids", []))
    doc_map = {doc["doc_id"]: doc for doc in row["documents"]}
    for family in families:
        if family["id"] not in selected_ids:
            continue
        for member in family.get("members", []):
            source_doc = doc_map.get(member["material_id"])
            if source_doc and source_doc["source_ref"] and source_doc["source_ref"] not in selected_keys:
                selected_keys.append(source_doc["source_ref"])
            if source_doc and source_doc["doc_id"] not in selected_doc_ids:
                selected_doc_ids.append(source_doc["doc_id"])
            selected_units.append({"doc_id": member["material_id"], "locator": member["locator"],
                                   "quote": member["quote"], "observable": member["observable"]})
    probability = decision.get("ratability_probability")
    if probability is None:
        confidence = statistics.mean(extraction_confidences) if extraction_confidences else 0.50
    elif state == "E":
        confidence = float(probability)
    else:
        confidence = 1.0 - float(probability)
    if state == "C":
        confidence = max(confidence, 0.75)
    return {
        "state": state, "level": output_level, "action": action,
        "confidence": round(max(0.01, min(0.99, confidence)), 6),
        "evidence_families": selected_keys,
        "evidence_doc_ids": selected_doc_ids,
        "evidence_units": selected_units,
        "path": baseline_path(row, action, library, mode="governed", level=output_level),
        "reason_codes": decision.get("reason_codes", []),
        "family_count": len(families), "valid_evidence_count": len(family_evidence),
        "rejected_evidence_count": len(rejected),
        "carrier_policy_excluded_count": len(carrier_policy_excluded),
        "ratability_probability": probability,
    }


def b3_predict(
    row: dict[str, Any], qwen: dict[str, Any], library: ResourceLibrary, threshold: float,
) -> dict[str, Any]:
    generic = qwen.get("generic") if isinstance(qwen.get("generic"), dict) else {}
    try:
        confidence = float(generic.get("self_confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    state = "E" if confidence >= threshold else "I"
    try:
        proposed_level = int(generic.get("level") or 1)
    except (TypeError, ValueError):
        proposed_level = 1
    level = max(1, min(4, proposed_level)) if state == "E" else None
    action = action_for(state, level)
    refs = [str(item) for item in generic.get("cited_family_keys", []) if str(item)]
    doc_ids = [str(item.get("doc_id")) for item in qwen.get("evidence", [])
               if isinstance(item, dict) and str(item.get("doc_id") or "")]
    doc_map = {doc["doc_id"]: doc for doc in row["documents"]}
    units = []
    for item in qwen.get("evidence", []):
        if not isinstance(item, dict):
            continue
        doc = doc_map.get(str(item.get("doc_id") or ""))
        quote = str(item.get("quote") or "")
        start = doc["text"].find(quote) if doc and quote else -1
        units.append({"doc_id": str(item.get("doc_id") or ""),
                      "locator": f"chars:{start}-{start + len(quote)}" if start >= 0 else None,
                      "quote": quote,
                      "observable": str((item.get("observables") or [""])[0])})
    return {"state": state, "level": level, "action": action,
            "confidence": round(max(0.0, min(1.0, confidence)), 6),
            "evidence_families": list(dict.fromkeys(refs)),
            "evidence_doc_ids": list(dict.fromkeys(doc_ids)), "evidence_units": units,
            "path": baseline_path(row, action, library, mode="bm25", level=level),
            "reason_codes": [str(generic.get("reason_code") or "generic_confidence_gate")],
            "threshold": threshold}


def reroute_without_three_state(
    row: dict[str, Any], p_output: dict[str, Any], qwen: dict[str, Any], library: ResourceLibrary,
) -> dict[str, Any]:

    output = json.loads(json.dumps(p_output))
    if output["state"] not in {"I", "C"}:
        return output
    generic = qwen.get("generic") if isinstance(qwen.get("generic"), dict) else {}
    try:
        level = max(1, min(4, int(generic.get("level") or 1)))
    except (TypeError, ValueError):
        level = 1
    action = action_for("E", level)
    output["action"] = action
    output["path"] = baseline_path(row, action, library, mode="bm25", level=level)
    output["reason_codes"] = list(output.get("reason_codes", [])) + ["three_state_route_removed"]
    return output


def select_b3_threshold(
    calibration: list[dict[str, Any]], references: list[dict[str, Any]],
    qwen_runs: dict[int, dict[str, dict[str, Any]]],
) -> tuple[float, dict[str, Any]]:
    values, labels = [], []
    for row, reference in zip(calibration, references, strict=True):
        confidences = []
        for run in RUNS:
            generic = qwen_runs[run][str(row["sample_id"])].get("generic", {})
            try:
                confidences.append(float(generic.get("self_confidence", 0.5)))
            except (TypeError, ValueError):
                confidences.append(0.5)
        values.append(statistics.median(confidences))
        labels.append(reference_state(reference) == "E")
    target_coverage = 0.80
    unique = sorted(set(max(0.0, min(1.0, float(value))) for value in values))
    thresholds = {0.0, 1.0, *unique}
    thresholds.update((left + right) / 2 for left, right in zip(unique, unique[1:]))
    candidates = []
    for threshold in sorted(thresholds):
        predicted = [value >= threshold for value in values]
        evaluation = binary_threshold_evaluation(labels, predicted, threshold)
        evaluation["coverage_eligible"] = (
            evaluation["_raw_coverage"] + 1e-12 >= target_coverage
        )
        candidates.append(evaluation)
    eligible = [item for item in candidates if item["coverage_eligible"]]
    if not eligible:
        raise RuntimeError("B3 calibration produced no threshold satisfying target coverage")
    winner = max(
        eligible,
        key=lambda item: (
            item["_raw_balanced_accuracy"], item["_raw_coverage"], item["threshold"]
        ),
    )
    threshold = float(winner["threshold"])
    detail = {
        "criterion": (
            "maximum_balanced_accuracy_subject_to_minimum_coverage_"
            "then_higher_coverage_then_higher_threshold"
        ),
        "target_coverage": target_coverage,
        "displayed_proportions": (
            "Jeffreys posterior means; raw numerators and denominators retained; "
            "selection and coverage eligibility use raw counts"
        ),
        "confidence_aggregation": "median_of_three_frozen_runs",
        "selected_threshold": threshold,
        "selected_coverage": winner["coverage"],
        "selected_balanced_accuracy": winner["balanced_accuracy"],
        "selected_raw_counts": winner["raw_counts"],
        "records": len(calibration),
        "candidates": [
            {key: value for key, value in item.items() if not key.startswith("_raw_")}
            for item in candidates
        ],
    }
    write_json(PRED / "B3" / "threshold_selection.json", detail)
    return threshold, detail


def task_output_filename(task: str, run: int | None = None) -> str:
    prefix = "" if task == "main" else f"{task}_"
    return f"{prefix}run_{run}.jsonl" if run is not None else f"{prefix}aggregate.jsonl"


def load_shared(stem: str, rows: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    expected = {str(row["sample_id"]) for row in rows}
    outputs = {}
    for run in RUNS:
        path = PRED / "shared_qwen" / f"{stem}_run{run}.jsonl"
        if not path.exists():
            raise RuntimeError(f"Missing shared-model output: {path}")
        mapping = {str(item["sample_id"]): item for item in read_jsonl(path)}
        if set(mapping) != expected:
            raise RuntimeError(f"Shared-model IDs do not match {stem} input for run {run}")
        outputs[run] = mapping
    return outputs


def run_task(
    task: str, raw_rows: list[dict[str, Any]], shared: dict[int, dict[str, dict[str, Any]]],
    b1_config: dict[str, Any], b2: TfidfLinearModel, b3_threshold: float,
    indicator_map: dict[str, dict[str, Any]], engine: RatabilityEngine,
    library: ResourceLibrary,
) -> None:
    assert_blind_evaluation(raw_rows, task)
    rows = [public_record(row) for row in raw_rows]
    b2_predictions = b2_outputs(b2, rows, library)
    all_outputs: dict[str, dict[int, list[dict[str, Any]]]] = {method: {} for method in METHODS}
    for run in RUNS:
        method_rows = {method: [] for method in METHODS}
        for index, (raw_row, row) in enumerate(zip(raw_rows, rows, strict=True)):
            qwen = shared[run][str(raw_row["sample_id"])]
            b1 = b1_predict(row, library, b1_config)
            b2_output = json.loads(json.dumps(b2_predictions[index]))
            b3 = b3_predict(row, qwen, library, b3_threshold)
            p = p_predict(row, qwen, indicator_map[row["indicator_id"]], engine, library,
                          deduplicate_families=True)
            p_no_family = p_predict(row, qwen, indicator_map[row["indicator_id"]], engine, library,
                                    deduplicate_families=False)
            p_no_route = reroute_without_three_state(row, p, qwen, library)
            method_rows["B1"].append(b1)
            method_rows["B2"].append(b2_output)
            method_rows["B3"].append(b3)
            method_rows["P"].append(p)
            method_rows["P_minus_Family"].append(p_no_family)
            method_rows["P_minus_Route"].append(p_no_route)
        for method, predictions in method_rows.items():
            enriched = [
                {"sample_id": str(raw_row["sample_id"]), "indicator_id": row["indicator_id"],
                 "run": run, **prediction}
                for raw_row, row, prediction in zip(raw_rows, rows, predictions, strict=True)
            ]
            all_outputs[method][run] = enriched
            write_jsonl(PRED / method / task_output_filename(task, run), enriched)
    for method in METHODS:
        write_jsonl(PRED / method / task_output_filename(task),
                    aggregate_runs(raw_rows, all_outputs[method]))


def aggregate_runs(rows: list[dict[str, Any]], outputs: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    aggregates = []
    for index, row in enumerate(rows):
        items = [outputs[run][index] for run in RUNS]
        state_counts = Counter(item["state"] for item in items)
        max_count = max(state_counts.values())
        tied = {state for state, count in state_counts.items() if count == max_count}
        if len(tied) == 1:
            state = next(iter(tied))
        else:
            state = max(tied, key=lambda candidate: statistics.mean(item["confidence"] for item in items if item["state"] == candidate))
        levels = [item["level"] for item in items if item["state"] == state and item["level"] is not None]
        level = int(round(statistics.median(levels))) if state == "E" and levels else None
        actions = Counter(item["action"] for item in items)
        action = actions.most_common(1)[0][0]
        ref_counts = Counter(ref for item in items for ref in item.get("evidence_families", []))
        refs = sorted(ref for ref, count in ref_counts.items() if count >= 2)
        doc_counts = Counter(ref for item in items for ref in item.get("evidence_doc_ids", []))
        doc_ids = sorted(ref for ref, count in doc_counts.items() if count >= 2)
        unit_counts: Counter[str] = Counter()
        unit_map: dict[str, dict[str, Any]] = {}
        for item in items:
            for unit in item.get("evidence_units", []):
                key = json.dumps(unit, ensure_ascii=False, sort_keys=True)
                unit_counts[key] += 1
                unit_map[key] = unit
        units = [unit_map[key] for key, count in unit_counts.items() if count >= 2]
        aggregate = {
            "sample_id": row["sample_id"], "indicator_id": public_record(row)["indicator_id"],
            "state": state, "level": level, "action": action,
            "confidence": round(statistics.median(float(item["confidence"]) for item in items), 6),
            "evidence_families": refs,
            "evidence_doc_ids": doc_ids,
            "evidence_units": units,
            "path": next(item["path"] for item in items if item["action"] == action),
            "run_states": [item["state"] for item in items],
            "run_levels": [item["level"] for item in items],
            "run_actions": [item["action"] for item in items],
            "state_stable": len({item["state"] for item in items}) == 1,
            "action_stable": len({item["action"] for item in items}) == 1,
        }
        aggregates.append(aggregate)
    return aggregates


def main() -> None:
    framework_path = ROOT / "app" / "data" / "framework.json"
    resources_path = ROOT / "app" / "data" / "resources.json"
    framework = json.loads(framework_path.read_text(encoding="utf-8"))
    resources_payload = json.loads(resources_path.read_text(encoding="utf-8"))
    indicator_map = {item["id"]: item for item in framework["indicators"]}
    library = ResourceLibrary(resources_path)

    development_path = DATA / "development_records.jsonl"
    calibration_path = DATA / "calibration_records.jsonl"
    development_reference_path = DATA / "development_reference.jsonl"
    calibration_reference_path = DATA / "calibration_reference.jsonl"
    for path in (development_path, calibration_path, development_reference_path, calibration_reference_path):
        if not path.exists():
            raise RuntimeError(f"Required training/calibration artifact is missing: {path}")

    development_raw = read_jsonl(development_path)
    calibration_raw = read_jsonl(calibration_path)
    development_references = join_reference(
        development_raw, read_jsonl(development_reference_path), split="development"
    )
    calibration_references = join_reference(
        calibration_raw, read_jsonl(calibration_reference_path), split="calibration"
    )
    development = [public_record(row) for row in development_raw]
    calibration_qwen = load_shared("calibration", calibration_raw)

    calibration_artifact, p_threshold_selection = build_calibration_artifact(
        calibration_raw, calibration_references, calibration_qwen, framework
    )
    engine = RatabilityEngine(
        calibration_artifact, framework_version=framework["version"],
        valid_indicator_ids=set(indicator_map),
    )
    b3_threshold, b3_threshold_selection = select_b3_threshold(
        calibration_raw, calibration_references, calibration_qwen
    )

    b1_config, b1_cv = select_b1_config(development, development_references)
    b2_config, b2_cv = select_b2_config(development, development_references)
    b2 = TfidfLinearModel(**b2_config)
    b2.fit(development, development_references)
    b2.save(DATA / "b2_tfidf_linear_model.npz")

    task_rows: dict[str, list[dict[str, Any]]] = {}
    task_hashes: dict[str, str] = {}
    for task, (filename, stem) in TASK_SPECS.items():
        path = DATA / filename
        if not path.exists():
            if task == "main":
                raise RuntimeError(f"Main evaluation input is missing: {path}")
            continue
        rows = read_jsonl(path)
        assert_blind_evaluation(rows, task)
        task_rows[task] = rows
        task_hashes[task] = sha256(path)
        run_task(task, rows, load_shared(stem, rows), b1_config, b2, b3_threshold,
                 indicator_map, engine, library)

    for method in METHODS:
        write_json(PRED / method / "run_config.json", {
            "method": method, "runs": list(RUNS),
            "tasks": {task: len(rows) for task, rows in task_rows.items()},
            "seed": SEED, "shared_model": "qwen3.7-flash",
            "b1_selected_config": b1_config if method == "B1" else None,
            "b1_calibration_artifact": "b1_rule_calibration.json" if method == "B1" else None,
            "b3_threshold": b3_threshold if method == "B3" else None,
            "b2_selected_config": b2_config if method == "B2" else None,
            "framework_version": framework["version"],
            "resource_version": resources_payload["version"],
            "calibration_artifact": calibration_artifact.name if method.startswith("P") else None,
        })

    write_json(PRED / "frozen_method_config.json", {
        "version": "M1-METHODS-2026-11-19.2",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": SEED, "methods": list(METHODS),
        "tasks": {task: len(rows) for task, rows in task_rows.items()},
        "development_records": len(development_raw), "calibration_records": len(calibration_raw),
        "shared_llm_model": "qwen3.7-flash", "shared_llm_runs": 3,
        "shared_llm_temperature": 0.30, "b3_confidence_threshold": b3_threshold,
        "b3_threshold_selection": b3_threshold_selection,
        "b1_selected_config": b1_config,
        "b1_grouped_cv": {"folds": b1_cv["folds"], "clusters": b1_cv["clusters"],
                          "selection_rule": b1_cv["selection_rule"]},
        "b2_selected_config": b2_config,
        "b2_grouped_cv": {"folds": b2_cv["folds"], "clusters": b2_cv["clusters"],
                          "selection_rule": b2_cv["selection_rule"]},
        "p_threshold_selection": p_threshold_selection,
        "p_core_modules": ["app.safety.scan_text", "app.evidence.validate_evidence",
                           "app.evidence.build_evidence_families", "app.ratability.RatabilityEngine",
                           "app.retrieval.ResourceLibrary"],
        "framework_sha256": sha256(framework_path), "resources_sha256": sha256(resources_path),
        "development_input_sha256": sha256(development_path),
        "development_reference_sha256": sha256(development_reference_path),
        "calibration_input_sha256": sha256(calibration_path),
        "calibration_reference_sha256": sha256(calibration_reference_path),
        "evaluation_input_sha256": task_hashes, "test_reference_accessed": False,
    })
    print(json.dumps({"status": "complete", "methods": len(METHODS),
                      "tasks": {task: len(rows) for task, rows in task_rows.items()},
                      "b3_threshold": b3_threshold}, ensure_ascii=False))


if __name__ == "__main__":
    main()
