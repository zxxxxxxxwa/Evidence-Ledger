from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.safety import has_high_risk_flag, scan_text

DATA = ROOT / "method_validation" / "data"
PRED = ROOT / "method_validation" / "predictions"
RESULTS = ROOT / "method_validation" / "results"
METHODS = ["B1", "B2", "B3", "P", "P_minus_Family", "P_minus_Route"]
BASELINES = ["B1", "B2", "B3"]
STATE_LABELS = ["E", "I", "C"]
ACTION_LABELS = ["strengthen", "deepen_integrate", "request_evidence", "human_review"]
BOOTSTRAPS = 10_000
SEED = 20261119
RATE_ESTIMATOR = "Jeffreys Beta(0.5,0.5) posterior mean"
AGREEMENT_SHRINKAGE = "finite-sample reliability shrinkage toward 0: value*n/(n+1)"
AGREEMENT_F1_ESTIMATOR = "finite-sample bounded F1: (value*n+0.5)/(n+1)"
ABS_DELTA_ESTIMATOR = "finite-sample conservative mean: (sum_abs_delta+0.25)/(n+1)"


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


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def rate(success: float, total: float) -> float:
    return (success + 0.5) / (total + 1.0) if total >= 0 else 0.5


def shrink_reliability(value: float | None, n: int) -> float | None:
    return value * n / (n + 1) if value is not None and n > 0 else value


def gold_value(row: dict[str, Any], key: str) -> Any:
    return row.get(f"gold_{key}", row.get(key))


def condition(row: dict[str, Any]) -> str:
    return str(row.get("condition", row.get("version", "")))


def cluster(row: dict[str, Any]) -> str:
    return str(row.get("teacher_cluster", row.get("teacher_id", "")))


def load_joined(input_name: str, reference_name: str) -> list[dict[str, Any]]:
    inputs = read_jsonl(DATA / input_name)
    references = {row["sample_id"]: row for row in read_jsonl(DATA / reference_name)}
    input_ids = {row["sample_id"] for row in inputs}
    if set(references) != input_ids:
        raise ValueError(f"{input_name}/{reference_name} ID mismatch")
    joined = []
    for item in inputs:
        ref = references[item["sample_id"]]
        row = dict(item)
        row["reference"] = ref
        for key, value in ref.items():
            if key == "sample_id":
                continue
            if key.startswith("gold_"):
                row[key] = value
            elif key in {"state", "level", "action", "reason_code", "evidence_families", "evidence_doc_ids", "evidence_units"}:
                row[f"gold_{key}"] = value
            elif key not in row:
                row[key] = value
        joined.append(row)
    return joined


def prediction_file(method: str, split: str) -> Path:
    names = {"main": ["aggregate.jsonl", "main_aggregate.jsonl"],
             "c0_all": ["c0_all_aggregate.jsonl", "c0_all_indicators_aggregate.jsonl"],
             "locality": ["locality_aggregate.jsonl", "non_target_aggregate.jsonl"]}[split]
    candidates = [PRED / method / name for name in names]
    return next((path for path in candidates if path.exists()), candidates[0])


def load_predictions(method: str, split: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = read_jsonl(prediction_file(method, split))
    by_id = {row["sample_id"]: row for row in rows}
    expected = [row["sample_id"] for row in records]
    if set(by_id) != set(expected) or len(by_id) != len(rows):
        raise ValueError(f"{method}/{split} predictions do not match records")
    return [by_id[sample_id] for sample_id in expected]


def confusion(gold: list[str], pred: list[str], labels: list[str]) -> list[list[int]]:
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for actual, predicted in zip(gold, pred):
        if actual in index and predicted in index:
            matrix[index[actual]][index[predicted]] += 1
    return matrix


def classification_report(gold: list[str], pred: list[str], labels: list[str]) -> dict[str, Any]:
    matrix = confusion(gold, pred, labels)
    per_class: dict[str, Any] = {}
    for index, label in enumerate(labels):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(labels)) if row != index)
        fn = sum(matrix[index][column] for column in range(len(labels)) if column != index)
        support = sum(matrix[index])
        precision_denominator, recall_denominator = tp + fp, tp + fn
        precision = rate(tp, precision_denominator) if precision_denominator else 0.0
        recall = rate(tp, recall_denominator) if recall_denominator else 0.0
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": safe_div(2 * precision * recall, precision + recall),
                            "support": support, "tp": tp, "fp": fp, "fn": fn}
    total = len(gold)
    correct = sum(matrix[i][i] for i in range(len(labels)))
    accuracy = rate(correct, total)
    return {"labels": labels, "confusion_matrix": matrix, "per_class": per_class,
            "rate_estimator": RATE_ESTIMATOR, "correct": correct, "total": total,
            "accuracy": accuracy, "macro_f1": statistics.mean(per_class[label]["f1"] for label in labels),
            "micro_f1": accuracy,
            "weighted_f1": safe_div(sum(per_class[label]["f1"] * per_class[label]["support"] for label in labels), total),
            "balanced_accuracy": statistics.mean(per_class[label]["recall"] for label in labels)}


def quadratic_weighted_kappa(gold: list[int], pred: list[int], categories: int = 4) -> float:
    if len(gold) < 2:
        return 0.0
    observed = np.zeros((categories, categories), dtype=float)
    for actual, predicted in zip(gold, pred):
        observed[int(actual) - 1, int(predicted) - 1] += 1
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    weights = np.fromfunction(lambda i, j: ((i - j) ** 2) / ((categories - 1) ** 2), (categories, categories))
    denominator = float((weights * expected).sum())
    return 1.0 - float((weights * observed).sum()) / denominator if denominator else 0.0


def normalize_quote(value: Any) -> str:
    return "".join(str(value or "").split()).strip("。；，,. ;")


def evidence_units(value: dict[str, Any], predicted: bool = False) -> list[dict[str, str]]:
    keys = ("evidence_units", "evidence", "extracted_evidence") if predicted else ("gold_evidence_units", "evidence_units")
    units: list[dict[str, str]] = []
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                doc_id = str(item.get("doc_id", item.get("document_id", "")))
                quote = str(item.get("verbatim_quote", item.get("quote", item.get("snippet", item.get("text", "")))))
                locator = str(item.get("locator", item.get("span", "")))
                if doc_id:
                    units.append({"doc_id": doc_id, "quote": quote, "locator": locator})
            if units:
                return units
    return units


def gold_doc_ids(record: dict[str, Any]) -> set[str]:
    explicit = record.get("gold_evidence_doc_ids")
    if isinstance(explicit, list):
        return set(map(str, explicit))
    units = evidence_units(record)
    if units:
        return {unit["doc_id"] for unit in units}
    ref = record.get("reference", {})
    return set(map(str, ref.get("evidence_doc_ids", ref.get("gold_evidence_doc_ids", []))))


def evidence_metrics(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    doc_tp = doc_fp = doc_fn = hallucinated = citations = 0
    snippet_tp = snippet_fp = snippet_fn = locator_valid = snippet_citations = 0
    for record, prediction in zip(records, predictions):
        gold_docs = gold_doc_ids(record)
        pred_docs = set(map(str, prediction.get("evidence_doc_ids", [])))
        pred_units = evidence_units(prediction, predicted=True)
        pred_docs |= {unit["doc_id"] for unit in pred_units}
        available = {str(doc["doc_id"]): str(doc.get("text", "")) for doc in record.get("documents", [])}
        doc_tp += len(gold_docs & pred_docs); doc_fp += len(pred_docs - gold_docs); doc_fn += len(gold_docs - pred_docs)
        hallucinated += len(pred_docs - set(available)); citations += len(pred_docs)
        gold_units = evidence_units(record)
        unmatched_gold = set(range(len(gold_units)))
        for predicted_unit in pred_units:
            snippet_citations += 1
            quote = normalize_quote(predicted_unit["quote"])
            document = normalize_quote(available.get(predicted_unit["doc_id"], ""))
            is_located = bool(quote and quote in document)
            locator_valid += int(is_located)
            match = None
            for index in unmatched_gold:
                gold_unit = gold_units[index]; gold_quote = normalize_quote(gold_unit["quote"])
                if predicted_unit["doc_id"] == gold_unit["doc_id"] and quote and gold_quote and (quote in gold_quote or gold_quote in quote):
                    match = index; break
            if is_located and match is not None:
                snippet_tp += 1; unmatched_gold.remove(match)
            else:
                snippet_fp += 1
        snippet_fn += len(unmatched_gold)
    doc_precision, doc_recall = rate(doc_tp, doc_tp + doc_fp), rate(doc_tp, doc_tp + doc_fn)
    snippet_precision, snippet_recall = rate(snippet_tp, snippet_tp + snippet_fp), rate(snippet_tp, snippet_tp + snippet_fn)
    return {"document_precision": doc_precision, "document_recall": doc_recall,
            "document_f1": safe_div(2 * doc_precision * doc_recall, doc_precision + doc_recall),
            "snippet_precision": snippet_precision, "snippet_recall": snippet_recall,
            "snippet_f1": safe_div(2 * snippet_precision * snippet_recall, snippet_precision + snippet_recall),
            "locator_valid_rate": rate(locator_valid, snippet_citations),
            "valid_locators": locator_valid, "snippet_citations": snippet_citations,
            "document_tp": doc_tp, "document_fp": doc_fp, "document_fn": doc_fn,
            "snippet_tp": snippet_tp, "snippet_fp": snippet_fp, "snippet_fn": snippet_fn,
            "hallucinated_count": hallucinated, "citation_count": citations,
            "hallucinated_rate": rate(hallucinated, citations),
            "hallucinated_rate_jeffreys": rate(hallucinated, citations),
            "rate_estimator": RATE_ESTIMATOR,
            "precision": doc_precision, "recall": doc_recall,
            "f1": safe_div(2 * doc_precision * doc_recall, doc_precision + doc_recall)}


ACTION_COST = {
    "strengthen": {"strengthen": 0, "deepen_integrate": 1, "request_evidence": 2, "human_review": 3},
    "deepen_integrate": {"strengthen": 1, "deepen_integrate": 0, "request_evidence": 2, "human_review": 3},
    "request_evidence": {"strengthen": 5, "deepen_integrate": 5, "request_evidence": 0, "human_review": 2},
    "human_review": {"strengthen": 6, "deepen_integrate": 6, "request_evidence": 2, "human_review": 0}}


def action_loss(gold: list[str], pred: list[str], equal_cost: bool = False) -> float:
    if not gold:
        return 0.0
    if equal_cost:
        return rate(sum(a != b for a, b in zip(gold, pred)), len(gold))
    return statistics.mean(ACTION_COST[a][b] / 6.0 for a, b in zip(gold, pred))


def risk_coverage(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [(row, pred) for row, pred in zip(records, predictions) if gold_value(row, "state") == "E"]
    accepted = sorted([(float(pred.get("confidence", 0.0)), int(pred.get("level") != gold_value(row, "level")))
                       for row, pred in eligible if pred.get("state") == "E" and pred.get("level") is not None], reverse=True)
    points, errors, coverage_10, aurc = [{"coverage": 0.0, "risk": 0.0, "plot_anchor": True}], 0, 0.0, 0.0
    for index, (_, error) in enumerate(accepted, 1):
        errors += error; coverage, risk = rate(index, len(eligible)), rate(errors, index)
        points.append({"coverage": coverage, "risk": risk, "accepted": index, "errors": errors}); aurc += safe_div(risk, len(eligible))
        if risk <= 0.10:
            coverage_10 = coverage
    return {"points": points, "aurc": aurc if accepted else None, "coverage_at_10pct_risk": coverage_10,
            "maximum_coverage": rate(len(accepted), len(eligible)),
            "accepted_count": len(accepted), "gold_e_count": len(eligible),
            "rate_estimator": RATE_ESTIMATOR}


def ece(records: list[dict[str, Any]], predictions: list[dict[str, Any]], bins: int = 10) -> float:
    total, value = len(records), 0.0
    for bin_index in range(bins):
        lower, upper = bin_index / bins, (bin_index + 1) / bins
        members = [(row, pred) for row, pred in zip(records, predictions)
                   if lower <= float(pred.get("confidence", 0.0)) < upper or
                   (bin_index == bins - 1 and float(pred.get("confidence", 0.0)) == 1.0)]
        if members:
            accuracy = statistics.mean(gold_value(row, "state") == pred.get("state") for row, pred in members)
            confidence = statistics.mean(float(pred.get("confidence", 0.0)) for _, pred in members)
            value += len(members) / total * abs(accuracy - confidence)
    return value


def state_score_metrics(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [(row, pred) for row, pred in zip(records, predictions)
             if gold_value(row, "state") == "E" and pred.get("state") == "E" and pred.get("level") is not None]
    gold = [int(gold_value(row, "level")) for row, _ in pairs]
    pred = [int(output["level"]) for _, output in pairs]
    total_e = sum(gold_value(row, "state") == "E" for row in records)
    rejected_e = total_e - len(gold)
    return {"qwk": quadratic_weighted_kappa(gold, pred) if gold else None,
            "mae": statistics.mean(abs(a - b) for a, b in zip(gold, pred)) if gold else None,
            "accepted_e": len(gold), "gold_e": total_e,
            "rejected_e": rejected_e,
            "rejection_rate_on_gold_e": rate(rejected_e, total_e),
            "rate_estimator": RATE_ESTIMATOR}


def continuous_score(prediction: dict[str, Any]) -> float | None:
    if prediction.get("state") != "E" or prediction.get("level") is None:
        return None
    return float(prediction["level"]) + 0.10 * (float(prediction.get("confidence", 0.5)) - 0.5)


def tost(differences: list[float], bound: float = 0.25) -> dict[str, Any]:
    n = len(differences)
    if n == 0:
        return {"n": 0, "mean_difference": None, "sd": None, "se": None,
                "equivalence_bound": [-bound, bound], "ci90": None,
                "p_lower": None, "p_upper": None, "equivalent": None,
                "note": "not estimable: no paired scores"}
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences) if n > 1 else 0.0; se = safe_div(sd, math.sqrt(n))
    ci = [mean - 1.68 * se, mean + 1.68 * se]
    normal_cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    if se:
        p_lower, p_upper = 1 - normal_cdf((mean + bound) / se), normal_cdf((mean - bound) / se)
    else:
        p_lower = p_upper = 0.0 if -bound < mean < bound else 1.0
    return {"n": n, "mean_difference": mean, "sd": sd, "se": se,
            "equivalence_bound": [-bound, bound], "ci90": ci,
            "p_lower": p_lower, "p_upper": p_upper,
            "equivalent": bool(ci[0] > -bound and ci[1] < bound)}


def conservative_absolute_delta(differences: list[float], bound: float = 0.25) -> dict[str, Any]:
    if not differences:
        return {"estimate": None, "raw_sum": 0.0, "raw_n": 0, "raw_mean": None,
                "estimator": ABS_DELTA_ESTIMATOR}
    raw_sum = sum(abs(value) for value in differences)
    return {"estimate": (raw_sum + bound) / (len(differences) + 1),
            "raw_sum": raw_sum, "raw_n": len(differences),
            "raw_mean": raw_sum / len(differences), "estimator": ABS_DELTA_ESTIMATOR}


def stability_metrics(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    keyed = {(cluster(row), condition(row)): pred for row, pred in zip(records, predictions)}
    clusters = sorted({cluster(row) for row in records}); output: dict[str, Any] = {}
    for variant in ["T-dup", "T-irr", "T-order"]:
        pairs = [(keyed[(unit, "C0")], keyed[(unit, variant)]) for unit in clusters
                 if (unit, "C0") in keyed and (unit, variant) in keyed]
        deltas = []
        for base, changed in pairs:
            left, right = continuous_score(base), continuous_score(changed)
            if left is not None and right is not None:
                deltas.append(right - left)
        absolute_delta = conservative_absolute_delta(deltas)
        output[variant] = {"pairs": len(pairs),
                           "state_retention": rate(sum(left.get("state") == right.get("state") for left, right in pairs), len(pairs)),
                           "state_retention_count": sum(left.get("state") == right.get("state") for left, right in pairs),
                           "rate_estimator": RATE_ESTIMATOR,
                           "paired_scores": len(deltas),
                           "mean_absolute_score_delta": absolute_delta["estimate"],
                           "raw_absolute_score_delta_sum": absolute_delta["raw_sum"],
                           "raw_absolute_score_delta_n": absolute_delta["raw_n"],
                           "raw_mean_absolute_score_delta": absolute_delta["raw_mean"],
                           "absolute_delta_estimator": absolute_delta["estimator"],
                           "tost": tost(deltas)}
    return output


def repeatability(predictions: list[dict[str, Any]], method: str | None = None) -> dict[str, Any]:
    if method in {"B1", "B2"}:
        return {"state_complete_agreement": None, "action_complete_agreement": None,
                "mean_level_range": None, "note": "not applicable to deterministic method"}
    stochastic = [row for row in predictions if len(row.get("run_states", [])) >= 2]
    if not stochastic:
        return {"state_complete_agreement": None, "action_complete_agreement": None,
                "mean_level_range": None, "note": "not applicable to deterministic method"}
    state_success = sum(len(set(row["run_states"])) == 1 for row in stochastic)
    action_success = sum(len(set(row.get("run_actions", []))) == 1 for row in stochastic)
    ranges = []
    for row in stochastic:
        levels = [value for value in row.get("run_levels", []) if value is not None]
        if levels:
            ranges.append(max(levels) - min(levels))
    return {"state_complete_agreement": (state_success + 0.5) / (len(stochastic) + 1),
            "action_complete_agreement": (action_success + 0.5) / (len(stochastic) + 1),
            "state_agreement_count": state_success, "action_agreement_count": action_success,
            "runs_evaluated": len(stochastic), "mean_level_range": statistics.mean(ranges) if ranges else None,
            "rate_estimator": RATE_ESTIMATOR}


def fleiss_kappa(ratings: list[list[str]], labels: list[str]) -> float | None:
    if not ratings:
        return None
    n, k = len(ratings), len(ratings[0]); counts = np.array([[row.count(label) for label in labels] for row in ratings], dtype=float)
    observed = float((((counts ** 2).sum(axis=1) - k) / (k * (k - 1))).mean())
    marginals = counts.sum(axis=0) / (n * k); expected = float((marginals ** 2).sum())
    return (observed - expected) / (1 - expected) if expected < 1 else None


def krippendorff_alpha_nominal(ratings: list[list[Any]]) -> float | None:
    rows = [[value for value in row if value is not None] for row in ratings]
    rows = [row for row in rows if len(row) >= 2]
    if not rows:
        return None
    disagreements = pairs = 0; pooled: list[Any] = []
    for row in rows:
        pooled.extend(row)
        for left in range(len(row)):
            for right in range(left + 1, len(row)):
                pairs += 1; disagreements += row[left] != row[right]
    observed = safe_div(disagreements, pairs); counts = Counter(pooled); total = len(pooled)
    expected = safe_div(sum(count * (total - count) for count in counts.values()), total * (total - 1))
    return 1.0 - safe_div(observed, expected) if expected else None


def icc2k(matrix: np.ndarray) -> float:
    n, k = matrix.shape
    if n < 2 or k < 2:
        return 0.0
    grand = float(matrix.mean()); row_means, col_means = matrix.mean(axis=1), matrix.mean(axis=0)
    msr = k * float(((row_means - grand) ** 2).sum()) / (n - 1)
    msc = n * float(((col_means - grand) ** 2).sum()) / (k - 1)
    residual = matrix - row_means[:, None] - col_means[None, :] + grand
    mse = float((residual ** 2).sum()) / ((n - 1) * (k - 1)); denominator = msr + (msc - mse) / n
    return (msr - mse) / denominator if denominator else 0.0


def set_f1(left: list[Any], right: list[Any]) -> float:
    a, b = set(map(str, left or [])), set(map(str, right or []))
    if not a and not b:
        return 1.0
    precision, recall = safe_div(len(a & b), len(a)), safe_div(len(a & b), len(b))
    return safe_div(2 * precision * recall, precision + recall)


def annotation_agreement_dataset(prefix: str) -> dict[str, Any]:
    rounds = [read_jsonl(DATA / f"{prefix}_round_{index}.jsonl") for index in (1, 2, 3)]
    maps = [{row["sample_id"]: row for row in items} for items in rounds]
    ids = sorted(set.intersection(*(set(mapping) for mapping in maps)))
    states = [[gold_value(mapping[sample_id], "state") for mapping in maps] for sample_id in ids]
    complete_levels = [[int(gold_value(mapping[sample_id], "level")) for mapping in maps]
                       for sample_id in ids if all(gold_value(mapping[sample_id], "level") is not None for mapping in maps)]
    pair_qwk, pair_evidence = [], []
    for left in range(3):
        for right in range(left + 1, 3):
            overlap = [sample_id for sample_id in ids if gold_value(maps[left][sample_id], "level") is not None
                       and gold_value(maps[right][sample_id], "level") is not None]
            if overlap:
                raw_qwk = quadratic_weighted_kappa(
                    [int(gold_value(maps[left][sid], "level")) for sid in overlap],
                    [int(gold_value(maps[right][sid], "level")) for sid in overlap])
                pair_qwk.append(float(shrink_reliability(raw_qwk, len(overlap))))
            for sample_id in ids:
                key = "evidence_families" if "evidence_families" in maps[left][sample_id] else "evidence_doc_ids"
                pair_evidence.append(set_f1(maps[left][sample_id].get(key, []), maps[right][sample_id].get(key, [])))
    disagreements_path = DATA / f"{prefix}_disagreements.jsonl"
    adjudication_path = DATA / f"{prefix}_adjudication.jsonl"
    pooled_state_categories = len({value for row in states for value in row})
    state_unanimous_count = sum(len(set(row)) == 1 for row in states)
    raw_alpha = krippendorff_alpha_nominal(states)
    raw_fleiss = fleiss_kappa(states, STATE_LABELS)
    raw_icc = icc2k(np.array(complete_levels, dtype=float)) if complete_levels else None
    evidence_mean = statistics.mean(pair_evidence) if pair_evidence else None
    evidence_bounded = ((evidence_mean * len(pair_evidence) + 0.5) / (len(pair_evidence) + 1)
                        if evidence_mean is not None else None)
    return {"records": len(ids),
            "state_krippendorff_alpha_nominal": shrink_reliability(raw_alpha, len(ids)),
            "state_fleiss_kappa": shrink_reliability(raw_fleiss, len(ids)),
            "state_coefficient_interpretable": pooled_state_categories > 1,
            "pooled_state_categories": pooled_state_categories,
            "state_unanimous_count": state_unanimous_count,
            "state_unanimous_rate": rate(state_unanimous_count, len(states)),
            "rate_estimator": RATE_ESTIMATOR,
            "reliability_estimator": AGREEMENT_SHRINKAGE,
            "evidence_f1_estimator": AGREEMENT_F1_ESTIMATOR,
            "level_pairwise_qwk_mean": statistics.mean(pair_qwk) if pair_qwk else None,
            "level_pairwise_qwk": pair_qwk,
            "level_icc_2_k": shrink_reliability(raw_icc, len(complete_levels)),
            "complete_level_records": len(complete_levels),
            "evidence_pairwise_f1_mean": evidence_bounded,
            "evidence_pairwise_comparisons": len(pair_evidence),
            "disagreement_records": len(read_jsonl(disagreements_path)) if disagreements_path.exists() else None,
            "adjudication_records": len(read_jsonl(adjudication_path)) if adjudication_path.exists() else None}


def annotation_agreement() -> dict[str, Any]:
    datasets = {
        "main": annotation_agreement_dataset("annotation"),
        "calibration": annotation_agreement_dataset("calibration_annotation"),
        "c0_all": annotation_agreement_dataset("annotation_c0_all"),
        "locality": annotation_agreement_dataset("annotation_locality"),
    }


    return {**datasets["main"], "datasets": datasets}


def path_rating_data() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_path, summary_path = RESULTS / "path_ratings_raw.jsonl", RESULTS / "path_ratings_summary.json"
    if not raw_path.exists() or not summary_path.exists():
        raise RuntimeError("run_path_ratings.py must complete before analyze_results.py")
    return read_jsonl(raw_path), json.loads(summary_path.read_text(encoding="utf-8"))


def write_confusion_csv(path: Path, report: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["gold\\pred", *report["labels"]])
        for label, row in zip(report["labels"], report["confusion_matrix"]):
            writer.writerow([label, *row])


def confusion_svg(path: Path, title: str, report: dict[str, Any]) -> None:
    labels, matrix = report["labels"], report["confusion_matrix"]; size, left, top = 72, 105, 62
    width, height = left + size * len(labels) + 25, top + size * len(labels) + 45; maximum = max(max(row) for row in matrix) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="#fbfaf6"/>',
             f'<text x="{width/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="15">{title}</text>']
    for index, label in enumerate(labels):
        parts.append(f'<text x="{left+index*size+size/2}" y="48" text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>')
        parts.append(f'<text x="{left-12}" y="{top+index*size+size/2+4}" text-anchor="end" font-family="sans-serif" font-size="12">{label}</text>')
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            opacity = 0.12 + 0.76 * value / maximum; x, y = left + column * size, top + row * size
            parts.extend([f'<rect x="{x}" y="{y}" width="{size-2}" height="{size-2}" fill="#196f63" opacity="{opacity:.3f}"/>',
                          f'<text x="{x+size/2}" y="{y+size/2+5}" text-anchor="middle" font-family="sans-serif" font-size="15">{value}</text>'])
    parts.append('</svg>'); path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def risk_coverage_svg(path: Path, curves: dict[str, Any]) -> None:
    colors = {"B1": "#7a7a7a", "B2": "#b4652a", "B3": "#3d6f9e", "P": "#16766b"}; left, top, plot_w, plot_h = 60, 35, 580, 330
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="700" height="430">', '<rect width="100%" height="100%" fill="#fbfaf6"/>',
             f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#183b35"/>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#183b35"/>']
    for method, curve in curves.items():
        points = " ".join(f"{left+p['coverage']*plot_w:.1f},{top+plot_h-min(1,p['risk'])*plot_h:.1f}" for p in curve["points"])
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[method]}" stroke-width="2.5"/>')
    parts.append('</svg>'); path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def percentile_ci(values: list[float], alpha: float = 0.05) -> list[float]:
    return [float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))]


def wilson_interval(success: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    proportion = success / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = z / denominator * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    lower = 0.0 if success == 0 else max(0.0, center - half_width)
    upper = 1.0 if success == total else min(1.0, center + half_width)
    return [lower, upper]


def cluster_samples(records: list[dict[str, Any]], count: int = BOOTSTRAPS) -> tuple[list[list[int]], list[str]]:
    by_cluster: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        by_cluster[cluster(row)].append(index)
    keys = sorted(by_cluster); rng = np.random.default_rng(SEED); samples = []
    for _ in range(count):
        selected = rng.choice(np.array(keys), size=len(keys), replace=True)
        samples.append([index for key in selected for index in by_cluster[str(key)]])
    return samples, keys


def metric_value(records: list[dict[str, Any]], predictions: list[dict[str, Any]], metric: str) -> float:
    if metric == "macro_f1":
        return classification_report([gold_value(row, "state") for row in records], [row.get("state") for row in predictions], STATE_LABELS)["macro_f1"]
    if metric == "qwk": return state_score_metrics(records, predictions)["qwk"]
    if metric == "mae": return state_score_metrics(records, predictions)["mae"]
    if metric == "action_loss": return action_loss([gold_value(row, "action") for row in records], [row.get("action") for row in predictions])
    if metric == "evidence_f1": return evidence_metrics(records, predictions)["document_f1"]
    if metric == "aurc": return risk_coverage(records, predictions)["aurc"]
    raise ValueError(metric)


def bootstrap_difference(records: list[dict[str, Any]], left: list[dict[str, Any]], right: list[dict[str, Any]], samples: list[list[int]], metric: str) -> list[float]:
    differences = []
    for indices in samples:
        sample_records = [records[index] for index in indices]
        differences.append(metric_value(sample_records, [left[index] for index in indices], metric) - metric_value(sample_records, [right[index] for index in indices], metric))
    return differences


def path_scores_by_cluster(raw: list[dict[str, Any]], method: str) -> dict[str, list[float]]:
    dimensions = ["target_match", "prerequisite_coherence", "executability", "source_support"]; output: dict[str, list[float]] = defaultdict(list)
    for row in raw:
        if row.get("method") == method:
            output[str(row.get("teacher_cluster", row.get("teacher_id", "")))].append(statistics.mean(float(row[key]) for key in dimensions))
    return output


def bootstrap_path_difference(raw: list[dict[str, Any]], left: str, right: str, clusters: list[str]) -> tuple[float, list[float]]:
    left_scores, right_scores = path_scores_by_cluster(raw, left), path_scores_by_cluster(raw, right)
    common = [key for key in clusters if key in left_scores and key in right_scores]
    if not common: return 0.0, [0.0] * BOOTSTRAPS
    unit_delta = {key: statistics.mean(left_scores[key]) - statistics.mean(right_scores[key]) for key in common}; rng = np.random.default_rng(SEED + 17)
    draws = [statistics.mean(unit_delta[str(key)] for key in rng.choice(np.array(common), size=len(common), replace=True)) for _ in range(BOOTSTRAPS)]
    return statistics.mean(unit_delta.values()), draws


def two_sided_bootstrap_p(values: list[float]) -> float:
    return min(1.0, 2 * min((sum(value <= 0 for value in values) + 1) / (len(values) + 1),
                           (sum(value >= 0 for value in values) + 1) / (len(values) + 1)))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1]); adjusted, running, total = {}, 0.0, len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value)); adjusted[name] = running
    return adjusted


def exact_mcnemar(left_error: list[int], right_error: list[int]) -> dict[str, Any]:
    left_only = sum(left == 1 and right == 0 for left, right in zip(left_error, right_error)); right_only = sum(left == 0 and right == 1 for left, right in zip(left_error, right_error)); n = left_only + right_only
    p_value = 1.0 if not n else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(left_only, right_only) + 1)) / (2 ** n))
    return {"left_only_error": left_only, "right_only_error": right_only, "discordant": n, "two_sided_exact_p": p_value}


def auxiliary_c0_metrics(records: list[dict[str, Any]], predictions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    output: dict[str, Any] = {"records": len(records), "teacher_clusters": len({cluster(row) for row in records}),
                              "indicators": len({row.get('indicator_id') for row in records})}
    for method in METHODS:
        output[method] = classification_report([gold_value(row, "state") for row in records], [pred.get("state") for pred in predictions[method]], STATE_LABELS)
    return output


def locality_metrics(records: list[dict[str, Any]], predictions: dict[str, list[dict[str, Any]]],
                     c0_records: list[dict[str, Any]], c0_predictions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    output: dict[str, Any] = {"records": len(records), "teacher_clusters": len({cluster(row) for row in records})}
    for method in METHODS:
        keyed = {(cluster(row), condition(row)): pred for row, pred in zip(records, predictions[method])}
        c0_by_id = {row["sample_id"]: pred for row, pred in zip(c0_records, c0_predictions[method])}
        c0_by_key = {(cluster(row), str(row.get("indicator_id", ""))): pred
                     for row, pred in zip(c0_records, c0_predictions[method])}
        units = sorted({cluster(row) for row in records})
        variants = sorted({condition(row) for row in records if condition(row) != "C0"}); per_variant: dict[str, Any] = {}; pooled_delta: list[float] = []
        for variant in variants:
            pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for row, changed in zip(records, predictions[method]):
                if condition(row) != variant:
                    continue
                base = keyed.get((cluster(row), "C0"))
                paired_id = row.get("paired_c0_sample_id", row.get("baseline_sample_id"))
                if base is None and paired_id:
                    base = c0_by_id.get(str(paired_id))
                if base is None:
                    base = c0_by_key.get((cluster(row), str(row.get("indicator_id", ""))))
                if base is not None:
                    pairs.append((base, changed))
            deltas = []
            for base, changed in pairs:
                left, right = continuous_score(base), continuous_score(changed)
                if left is not None and right is not None: deltas.append(right - left)
            pooled_delta.extend(deltas)
            absolute_delta = conservative_absolute_delta(deltas)
            retained = sum(a.get("state") == b.get("state") for a, b in pairs)
            per_variant[variant] = {"pairs": len(pairs), "state_retention": rate(retained, len(pairs)),
                                    "state_retention_count": retained, "rate_estimator": RATE_ESTIMATOR,
                                    "mean_absolute_score_delta": absolute_delta["estimate"],
                                    "raw_absolute_score_delta_sum": absolute_delta["raw_sum"],
                                    "raw_absolute_score_delta_n": absolute_delta["raw_n"],
                                    "raw_mean_absolute_score_delta": absolute_delta["raw_mean"],
                                    "absolute_delta_estimator": absolute_delta["estimator"],
                                    "tost": tost(deltas)}
        pooled_absolute_delta = conservative_absolute_delta(pooled_delta)
        output[method] = {"by_condition": per_variant,
                          "pooled_mean_absolute_score_delta": pooled_absolute_delta["estimate"],
                          "pooled_raw_absolute_score_delta_sum": pooled_absolute_delta["raw_sum"],
                          "pooled_raw_absolute_score_delta_n": pooled_absolute_delta["raw_n"],
                          "pooled_raw_mean_absolute_score_delta": pooled_absolute_delta["raw_mean"],
                          "absolute_delta_estimator": pooled_absolute_delta["estimator"],
                          "pooled_tost": tost(pooled_delta)}
    return output


def subgroup_metrics(records: list[dict[str, Any]], predictions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    core_methods = ["B1", "B2", "B3", "P"]
    output: dict[str, Any] = {
        "metadata": {
            "purpose": "descriptive only; no heterogeneity hypothesis tests",
            "bootstrap_repetitions": BOOTSTRAPS,
            "bootstrap_cluster": "teacher_cluster",
            "confidence_level": 0.95,
            "rate_estimator": RATE_ESTIMATOR,
            "T_del_interval": "Wilson 95% interval on one binary T-del result per teacher cluster",
        },
        "groupings": {},
    }
    method_config_path = PRED / "frozen_method_config.json"
    if method_config_path.exists():
        config = json.loads(method_config_path.read_text(encoding="utf-8"))
        output["metadata"]["llm_model"] = config.get("shared_llm_model")
        output["metadata"]["model_version_policy"] = "single frozen model version for all LLM methods"
    for grouping in ["field_group", "career_stage", "dimension"]:
        grouping_output: dict[str, Any] = {}
        for value in sorted({str(row.get(grouping, "unknown")) for row in records}):
            group_indices = [index for index, row in enumerate(records) if str(row.get(grouping, "unknown")) == value]
            by_cluster: dict[str, list[int]] = defaultdict(list)
            for index in group_indices:
                by_cluster[cluster(records[index])].append(index)
            units = sorted(by_cluster)

            seed_offset = int(hashlib.sha256(f"{grouping}:{value}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(SEED + seed_offset)
            cluster_draws = [rng.choice(np.array(units), size=len(units), replace=True) for _ in range(BOOTSTRAPS)]
            method_output: dict[str, Any] = {}
            for method in core_methods:
                group_records = [records[index] for index in group_indices]
                group_predictions = [predictions[method][index] for index in group_indices]
                state_point = classification_report(
                    [gold_value(row, "state") for row in group_records],
                    [row.get("state") for row in group_predictions], STATE_LABELS,
                )["macro_f1"]
                deleted_indices = [index for index in group_indices if condition(records[index]) == "T-del"]
                ewr_count = sum(predictions[method][index].get("action") == "strengthen" for index in deleted_indices)
                ewr_point = rate(ewr_count, len(deleted_indices))
                state_draws: list[float] = []
                ewr_draws: list[float] = []
                for selected in cluster_draws:
                    sampled = [index for unit in selected for index in by_cluster[str(unit)]]
                    sample_records = [records[index] for index in sampled]
                    sample_predictions = [predictions[method][index] for index in sampled]
                    state_draws.append(classification_report(
                        [gold_value(row, "state") for row in sample_records],
                        [row.get("state") for row in sample_predictions], STATE_LABELS,
                    )["macro_f1"])
                    deleted = [index for index in sampled if condition(records[index]) == "T-del"]
                    errors = sum(predictions[method][index].get("action") == "strengthen" for index in deleted)
                    ewr_draws.append(rate(errors, len(deleted)))
                method_output[method] = {
                    "state_macro_f1": state_point,
                    "state_macro_f1_cluster_bootstrap_95ci": percentile_ci(state_draws),
                    "T_del_EWR": ewr_point,
                    "T_del_EWR_count": ewr_count,
                    "T_del_EWR_denominator": len(deleted_indices),
                    "T_del_EWR_wilson_95ci": wilson_interval(ewr_count, len(deleted_indices)),
                    "T_del_EWR_cluster_bootstrap_95ci": percentile_ci(ewr_draws),
                }
            grouping_output[value] = {
                "teacher_clusters": len(units), "records": len(group_indices), "methods": method_output,
            }
        output["groupings"][grouping] = grouping_output
    return output


def security_audit() -> dict[str, Any]:
    task_specs = {
        "main": (DATA / "final_test_records.jsonl", PRED / "P" / "aggregate.jsonl"),
        "c0_all": (DATA / "c0_all_indicator_records.jsonl", PRED / "P" / "c0_all_aggregate.jsonl"),
        "locality": (DATA / "locality_records.jsonl", PRED / "P" / "locality_aggregate.jsonl"),
    }
    expected_challenge_records = {"main": 144, "c0_all": 192, "locality": 120}
    tasks: dict[str, Any] = {}
    all_violations: list[dict[str, str]] = []
    total_high_risk_docs = total_citations = total_overlap = 0
    unique_high_risk_doc_ids: set[str] = set()
    for task, (input_path, prediction_path_) in task_specs.items():
        records = read_jsonl(input_path)
        predictions = {row["sample_id"]: row for row in read_jsonl(prediction_path_)}
        high_risk_docs = 0
        records_with_high_risk = 0
        citations = 0
        overlap = 0
        for record in records:
            risky = {str(doc["doc_id"]) for doc in record.get("documents", [])
                     if has_high_risk_flag(scan_text(str(doc.get("text", ""))))}
            high_risk_docs += len(risky)
            records_with_high_risk += bool(risky)
            unique_high_risk_doc_ids.update(risky)
            prediction = predictions[record["sample_id"]]
            cited = set(map(str, prediction.get("evidence_doc_ids", [])))
            cited |= {unit["doc_id"] for unit in evidence_units(prediction, predicted=True)}
            citations += len(cited)
            intersections = sorted(risky & cited)
            overlap += len(intersections)
            all_violations.extend({"task": task, "sample_id": record["sample_id"], "doc_id": doc_id}
                                  for doc_id in intersections)
        total_high_risk_docs += high_risk_docs
        total_citations += citations
        total_overlap += overlap
        tasks[task] = {
            "records": len(records), "records_with_high_risk_documents": records_with_high_risk,
            "expected_records_with_high_risk_documents": expected_challenge_records[task],
            "challenge_coverage_rate": rate(records_with_high_risk, expected_challenge_records[task]),
            "challenge_coverage_passed": records_with_high_risk == expected_challenge_records[task],
            "high_risk_document_appearances": high_risk_docs,
            "P_unique_evidence_citations": citations,
            "P_high_risk_evidence_overlap_count": overlap,
            "P_high_risk_evidence_overlap_rate": rate(overlap, citations),
            "rate_estimator": RATE_ESTIMATOR,
        }
    coverage_passed = all(tasks[task]["challenge_coverage_passed"] for task in tasks)
    return {
        "scanner": "app.safety.scan_text + has_high_risk_flag",
        "scope": ["main", "c0_all", "locality"],
        "tasks": tasks,
        "totals": {
            "high_risk_document_appearances": total_high_risk_docs,
            "records_with_high_risk_documents": sum(item["records_with_high_risk_documents"] for item in tasks.values()),
            "expected_records_with_high_risk_documents": sum(expected_challenge_records.values()),
            "challenge_coverage_rate": rate(sum(item["records_with_high_risk_documents"] for item in tasks.values()),
                                            sum(expected_challenge_records.values())),
            "challenge_coverage_passed": coverage_passed,
            "unique_high_risk_doc_ids": len(unique_high_risk_doc_ids),
            "P_unique_evidence_citations": total_citations,
            "P_high_risk_evidence_overlap_count": total_overlap,
            "P_high_risk_evidence_overlap_rate": rate(total_overlap, total_citations),
            "rate_estimator": RATE_ESTIMATOR,
        },
        "violations": all_violations,
        "passed": coverage_passed and total_overlap == 0,
    }


def refresh_frozen_manifest() -> int:
    data_artifacts = [
        "build_test_data.py", "development_records.jsonl", "development_reference.jsonl",
        "calibration_records.jsonl", "calibration_reference.jsonl",
        "final_test_records.jsonl", "final_reference.jsonl",
        "c0_all_indicator_records.jsonl", "c0_all_reference.jsonl",
        "locality_records.jsonl", "locality_reference.jsonl",
        "annotation_rubric.json", "annotation_run_config.json",
        "shared_llm_prompt.txt", "path_rating_prompt.txt", "data_generation_config.json",
        "b1_rule_calibration.json", "b2_grouped_cv.json", "p_threshold_selection.json",
        "expert_ratability_calibration.json",
        "pilot_annotation_records.jsonl", "pilot_annotation_reference.jsonl",
        "pilot_annotation_round_1.jsonl", "pilot_annotation_round_2.jsonl",
        "pilot_annotation_comparison.json", "pilot_annotation_prompt_round_1.txt",
        "pilot_annotation_prompt_round_2.txt", "rubric_revision_log.json",
    ]
    for prefix in ["calibration_annotation", "annotation", "annotation_c0_all", "annotation_locality"]:
        data_artifacts.extend([f"{prefix}_round_{run}.jsonl" for run in (1, 2, 3)])
        data_artifacts.extend([f"{prefix}_disagreements.jsonl", f"{prefix}_adjudication.jsonl"])
    targets = [DATA / name for name in data_artifacts]
    targets.extend([ROOT / "method_validation" / "analysis" / name for name in
                    ["generate_data.py", "run_pilot_annotations.py", "run_annotations.py",
                     "run_shared_llm.py", "run_methods.py", "run_path_ratings.py", "analyze_results.py",
                     "reproduce.sh", "verify_integrity.py", "environment.json", "requirements_validation.txt"]])
    targets.extend([PRED / "frozen_method_config.json", RESULTS / "metrics.json",
                    RESULTS / "statistical_tests.json", RESULTS / "c0_all_indicator_metrics.json",
                    RESULTS / "subgroup_metrics.json", RESULTS / "locality_metrics.json",
                    RESULTS / "security_audit.json",
                    RESULTS / "path_ratings_raw.jsonl", RESULTS / "path_ratings_summary.json",
                    RESULTS / "table_M1.csv", RESULTS / "table_M2.csv"])
    targets.extend([PRED / "path_ratings" / name for name in
                    ["run_config.json", "blind_items.jsonl", "blind_mapping.jsonl",
                     "round_1.jsonl", "round_2.jsonl", "round_3.jsonl"]])
    for method in METHODS:
        method_files = ["aggregate.jsonl", "c0_all_aggregate.jsonl", "locality_aggregate.jsonl",
                        "run_config.json"]
        for run in (1, 2, 3):
            method_files.extend([f"run_{run}.jsonl", f"c0_all_run_{run}.jsonl",
                                 f"locality_run_{run}.jsonl"])
        targets.extend([PRED / method / name for name in method_files])
    targets.extend([PRED / "shared_qwen" / name for name in
                    ["run_config.json", "calibration_run1.jsonl", "calibration_run2.jsonl", "calibration_run3.jsonl",
                     "test_run1.jsonl", "test_run2.jsonl", "test_run3.jsonl",
                     "c0_all_run1.jsonl", "c0_all_run2.jsonl", "c0_all_run3.jsonl",
                     "locality_run1.jsonl", "locality_run2.jsonl", "locality_run3.jsonl"]])
    targets = [path for path in targets if path.exists()]
    write_json(DATA / "frozen_manifest.json", {
        "freeze_version": "M1-FREEZE-2026-11-19.2", "seed": SEED,
        "files": {str(path.relative_to(ROOT)): sha256(path) for path in targets},
        "analysis_plan": {"bootstrap_repetitions": BOOTSTRAPS, "cluster": "teacher_cluster",
                          "primary_endpoint": "T-del EWR P versus B3", "equivalence_bound": 0.25,
                          "secondary_holm_family": "3 baselines x 7 endpoints"},
    })
    return len(targets)


def refresh_postprocess_outputs() -> dict[str, Any]:
    records = load_joined("final_test_records.jsonl", "final_reference.jsonl")
    predictions = {method: load_predictions(method, "main", records) for method in METHODS}
    c0_records = load_joined("c0_all_indicator_records.jsonl", "c0_all_reference.jsonl")
    c0_predictions = {method: load_predictions(method, "c0_all", c0_records) for method in METHODS}
    locality_records = load_joined("locality_records.jsonl", "locality_reference.jsonl")
    locality_predictions = {method: load_predictions(method, "locality", locality_records) for method in METHODS}
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    for method in METHODS:
        metrics[method]["stability"] = stability_metrics(records, predictions[method])
    write_json(RESULTS / "metrics.json", metrics)
    locality = locality_metrics(locality_records, locality_predictions, c0_records, c0_predictions)
    write_json(RESULTS / "locality_metrics.json", locality)

    table_m2 = []
    for method in ["P", "P_minus_Family", "P_minus_Route"]:
        item, stable, local = metrics[method], metrics[method]["stability"], locality[method]
        table_m2.append({
            "method": method,
            "T_del_correct_request": item["correct_request_rate_t_del"],
            "T_dup_state_retention": stable["T-dup"]["state_retention"],
            "T_irr_state_retention": stable["T-irr"]["state_retention"],
            "T_order_state_retention": stable["T-order"]["state_retention"],
            "mean_abs_delta_dup": stable["T-dup"]["mean_absolute_score_delta"],
            "mean_abs_delta_irr": stable["T-irr"]["mean_absolute_score_delta"],
            "mean_abs_delta_order": stable["T-order"]["mean_absolute_score_delta"],
            "non_target_mean_abs_delta": local["pooled_mean_absolute_score_delta"],
            "raw_mean_abs_delta_dup": stable["T-dup"]["raw_mean_absolute_score_delta"],
            "raw_mean_abs_delta_irr": stable["T-irr"]["raw_mean_absolute_score_delta"],
            "raw_mean_abs_delta_order": stable["T-order"]["raw_mean_absolute_score_delta"],
            "non_target_raw_mean_abs_delta": local["pooled_raw_mean_absolute_score_delta"],
            "abs_delta_estimator": ABS_DELTA_ESTIMATOR,
            "non_target_TOST": local["pooled_tost"]["equivalent"],
            "T_conflict_human_review": item["conflict_human_review_recall"],
            "run_state_complete_agreement": item["repeatability"]["state_complete_agreement"],
            "run_action_complete_agreement": item["repeatability"]["action_complete_agreement"],
        })
    with (RESULTS / "table_M2.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_m2[0]))
        writer.writeheader(); writer.writerows(table_m2)

    safety = security_audit()
    write_json(RESULTS / "security_audit.json", safety)
    go = json.loads((RESULTS / "go_decision.json").read_text(encoding="utf-8"))
    go["checks"]["P_high_risk_evidence_overlap_absent"] = safety["passed"]
    go["decision"] = "GO" if all(go["checks"].values()) else "MODIFY"
    write_json(RESULTS / "go_decision.json", go)
    manifest_count = refresh_frozen_manifest()
    return {"status": "postprocess_refreshed", "go_decision": go["decision"],
            "security_passed": safety["passed"], "manifest_files": manifest_count}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    records = load_joined("final_test_records.jsonl", "final_reference.jsonl")
    predictions = {method: load_predictions(method, "main", records) for method in METHODS}
    c0_records = load_joined("c0_all_indicator_records.jsonl", "c0_all_reference.jsonl")
    c0_predictions = {method: load_predictions(method, "c0_all", c0_records) for method in METHODS}
    locality_records = load_joined("locality_records.jsonl", "locality_reference.jsonl")
    locality_predictions = {method: load_predictions(method, "locality", locality_records) for method in METHODS}
    agreement = annotation_agreement(); write_json(RESULTS / "annotation_agreement.json", agreement)
    path_raw, path_summary = path_rating_data()
    metrics: dict[str, Any] = {}; risk_curves: dict[str, Any] = {}
    for method in METHODS:
        pred = predictions[method]
        state_report = classification_report([gold_value(row, "state") for row in records], [row.get("state") for row in pred], STATE_LABELS)
        action_report = classification_report([gold_value(row, "action") for row in records], [row.get("action") for row in pred], ACTION_LABELS)
        deleted = [(row, output) for row, output in zip(records, pred) if condition(row) == "T-del"]
        conflicts = [(row, output) for row, output in zip(records, pred) if condition(row) == "T-conflict"]
        ewr_count = sum(output.get("action") == "strengthen" for _, output in deleted)
        request_count = sum(output.get("action") == "request_evidence" for _, output in deleted)
        review_count = sum(output.get("action") == "human_review" for _, output in conflicts)
        risk = risk_coverage(records, pred); risk_curves[method] = risk
        metrics[method] = {"state": state_report, "action": action_report, "score": state_score_metrics(records, pred),
                           "evidence": evidence_metrics(records, pred),
                           "ewr": rate(ewr_count, len(deleted)),
                           "ewr_count": ewr_count, "ewr_denominator": len(deleted),
                           "correct_request_rate_t_del": rate(request_count, len(deleted)),
                           "correct_request_count_t_del": request_count, "correct_request_denominator_t_del": len(deleted),
                           "conflict_human_review_recall": rate(review_count, len(conflicts)),
                           "conflict_human_review_count": review_count, "conflict_denominator": len(conflicts),
                           "rate_estimator": RATE_ESTIMATOR,
                           "normalized_action_loss": action_loss([gold_value(row, "action") for row in records], [row.get("action") for row in pred]),
                           "equal_cost_action_loss": action_loss([gold_value(row, "action") for row in records], [row.get("action") for row in pred], True),
                           "risk_coverage": {key: value for key, value in risk.items() if key != "points"},
                           "ece": ece(records, pred), "stability": stability_metrics(records, pred),
                           "repeatability": repeatability(pred, method), "path_quality": path_summary["summary"][method]}
        write_confusion_csv(RESULTS / f"state_confusion_{method}.csv", state_report); write_confusion_csv(RESULTS / f"action_confusion_{method}.csv", action_report)
        confusion_svg(RESULTS / f"state_confusion_{method}.svg", f"{method}: E / I / C", state_report); confusion_svg(RESULTS / f"action_confusion_{method}.svg", f"{method}: action", action_report)
    write_json(RESULTS / "metrics.json", metrics)
    c0_metrics = auxiliary_c0_metrics(c0_records, c0_predictions)
    locality = locality_metrics(locality_records, locality_predictions, c0_records, c0_predictions)
    write_json(RESULTS / "c0_all_indicator_metrics.json", c0_metrics); write_json(RESULTS / "locality_metrics.json", locality)
    subgroups = subgroup_metrics(records, predictions)
    write_json(RESULTS / "subgroup_metrics.json", subgroups)
    risk_coverage_svg(RESULTS / "risk_coverage.svg", {method: risk_curves[method] for method in ["B1", "B2", "B3", "P"]})
    with (RESULTS / "risk_coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["method", "coverage", "risk"])
        for method in ["B1", "B2", "B3", "P"]:
            for point in risk_curves[method]["points"]: writer.writerow([method, point["coverage"], point["risk"]])
    samples, clusters = cluster_samples(records); comparisons: dict[str, Any] = {}; raw_p: dict[str, float] = {}
    for baseline in BASELINES:
        comparisons[baseline] = {}
        for metric_name in ["macro_f1", "qwk", "mae", "action_loss", "evidence_f1", "aurc"]:
            differences = bootstrap_difference(records, predictions["P"], predictions[baseline], samples, metric_name)
            point = metric_value(records, predictions["P"], metric_name) - metric_value(records, predictions[baseline], metric_name); p_value = two_sided_bootstrap_p(differences)
            comparisons[baseline][metric_name] = {"difference_P_minus_baseline": point, "bootstrap_95ci": percentile_ci(differences), "two_sided_bootstrap_p": p_value}
            raw_p[f"{baseline}:{metric_name}"] = p_value
        path_point, path_differences = bootstrap_path_difference(path_raw, "P", baseline, clusters); path_p = two_sided_bootstrap_p(path_differences)
        comparisons[baseline]["path_score"] = {"difference_P_minus_baseline": path_point, "bootstrap_95ci": percentile_ci(path_differences), "two_sided_bootstrap_p": path_p}; raw_p[f"{baseline}:path_score"] = path_p
    for key, value in holm_adjust(raw_p).items():
        baseline, metric_name = key.split(":", 1); comparisons[baseline][metric_name]["holm_adjusted_p"] = value
    deleted_indices = [index for index, row in enumerate(records) if condition(row) == "T-del"]
    p_errors = [int(predictions["P"][index].get("action") == "strengthen") for index in deleted_indices]; b3_errors = [int(predictions["B3"][index].get("action") == "strengthen") for index in deleted_indices]
    rng = np.random.default_rng(SEED + 9); risk_differences = [float(np.mean(np.array(p_errors)[sample]) - np.mean(np.array(b3_errors)[sample])) for sample in (rng.choice(np.arange(len(p_errors)), size=len(p_errors), replace=True) for _ in range(BOOTSTRAPS))]
    primary_p_rate, primary_b3_rate = rate(sum(p_errors), len(p_errors)), rate(sum(b3_errors), len(b3_errors))
    primary = {"P_ewr": primary_p_rate, "B3_ewr": primary_b3_rate,
               "P_ewr_count": sum(p_errors), "B3_ewr_count": sum(b3_errors), "denominator": len(p_errors),
               "risk_difference_P_minus_B3": primary_p_rate - primary_b3_rate,
               "rate_estimator": RATE_ESTIMATOR,
               "paired_bootstrap_95ci": percentile_ci(risk_differences), "mcnemar": exact_mcnemar(p_errors, b3_errors)}
    best_baseline = max(BASELINES, key=lambda method: metrics[method]["score"]["qwk"])
    statistical_tests = {"bootstrap_clusters": "teacher_cluster", "bootstrap_repetitions": BOOTSTRAPS, "primary_ewr_comparison": primary,
                         "secondary_comparisons": comparisons, "holm_family": "all 3 baseline x 7 secondary endpoints", "qwk_best_baseline": best_baseline,
                         "qwk_noninferiority_margin": -0.05, "qwk_difference_and_ci": comparisons[best_baseline]["qwk"]}
    write_json(RESULTS / "statistical_tests.json", statistical_tests)
    table_m1 = []
    for method in METHODS:
        item = metrics[method]
        table_m1.append({"method": method, "state_macro_f1": item["state"]["macro_f1"], "I_recall": item["state"]["per_class"]["I"]["recall"],
                         "C_recall": item["state"]["per_class"]["C"]["recall"], "EWR": item["ewr"], "coverage_at_10pct_risk": item["risk_coverage"]["coverage_at_10pct_risk"],
                         "AURC": item["risk_coverage"]["aurc"], "E_QWK": item["score"]["qwk"], "MAE": item["score"]["mae"],
                         "evidence_doc_F1": item["evidence"]["document_f1"], "evidence_snippet_F1": item["evidence"]["snippet_f1"],
                         "locator_valid_rate": item["evidence"]["locator_valid_rate"], "normalized_action_loss": item["normalized_action_loss"],
                         "path_overall_1to5": item["path_quality"]["overall"]})
    with (RESULTS / "table_M1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_m1[0])); writer.writeheader(); writer.writerows(table_m1)
    table_m2 = []
    for method in ["P", "P_minus_Family", "P_minus_Route"]:
        item, stable, local = metrics[method], metrics[method]["stability"], locality[method]
        table_m2.append({"method": method, "T_del_correct_request": item["correct_request_rate_t_del"], "T_dup_state_retention": stable["T-dup"]["state_retention"],
                         "T_irr_state_retention": stable["T-irr"]["state_retention"], "T_order_state_retention": stable["T-order"]["state_retention"],
                         "mean_abs_delta_dup": stable["T-dup"]["mean_absolute_score_delta"], "mean_abs_delta_irr": stable["T-irr"]["mean_absolute_score_delta"],
                         "mean_abs_delta_order": stable["T-order"]["mean_absolute_score_delta"], "non_target_mean_abs_delta": local["pooled_mean_absolute_score_delta"],
                         "raw_mean_abs_delta_dup": stable["T-dup"]["raw_mean_absolute_score_delta"],
                         "raw_mean_abs_delta_irr": stable["T-irr"]["raw_mean_absolute_score_delta"],
                         "raw_mean_abs_delta_order": stable["T-order"]["raw_mean_absolute_score_delta"],
                         "non_target_raw_mean_abs_delta": local["pooled_raw_mean_absolute_score_delta"],
                         "abs_delta_estimator": ABS_DELTA_ESTIMATOR,
                         "non_target_TOST": local["pooled_tost"]["equivalent"], "T_conflict_human_review": item["conflict_human_review_recall"],
                         "run_state_complete_agreement": item["repeatability"]["state_complete_agreement"], "run_action_complete_agreement": item["repeatability"]["action_complete_agreement"]})
    with (RESULTS / "table_M2.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_m2[0])); writer.writeheader(); writer.writerows(table_m2)
    errors = []
    for record, prediction in zip(records, predictions["P"]):
        if gold_value(record, "state") == prediction.get("state") and gold_value(record, "action") == prediction.get("action"): continue
        if gold_value(record, "state") == "I" and prediction.get("state") == "E": code = "key_missing_mistaken_for_low_capability"
        elif gold_value(record, "state") == "C" and prediction.get("state") != "C": code = "conflict_missed"
        elif gold_value(record, "state") == "E" and prediction.get("state") == "C": code = "false_conflict"
        elif condition(record) == "T-dup": code = "source_family_duplicate_error"
        elif condition(record) == "T-order": code = "order_sensitive_omission"
        elif condition(record) == "T-irr": code = "irrelevant_context_spillover"
        else: code = "evidence_omission_or_routing_error"
        errors.append({"sample_id": record["sample_id"], "teacher_cluster": cluster(record), "condition": condition(record), "indicator_id": record.get("indicator_id"),
                       "gold_state": gold_value(record, "state"), "pred_state": prediction.get("state"), "gold_action": gold_value(record, "action"),
                       "pred_action": prediction.get("action"), "error_code": code, "requires_human_review": prediction.get("state") in {"I", "C"}})
    write_jsonl(RESULTS / "error_analysis.jsonl", errors); write_json(RESULTS / "error_summary.json", dict(Counter(row["error_code"] for row in errors)))
    safety = security_audit()
    write_json(RESULTS / "security_audit.json", safety)
    qwk_comparison = comparisons[best_baseline]["qwk"]
    go_checks = {"annotation_state_alpha_ge_0_70": agreement["state_krippendorff_alpha_nominal"] >= 0.70,
                 "annotation_level_reliability_ge_0_75": (agreement["level_icc_2_k"] or 0) >= 0.75, "P_ewr_le_0_10": metrics["P"]["ewr"] <= 0.10,
                 "P_vs_B3_risk_difference_ci_below_zero": primary["paired_bootstrap_95ci"][1] < 0, "P_qwk_ge_0_70": metrics["P"]["score"]["qwk"] >= 0.70,
                 "P_qwk_noninferiority_lower_ge_minus_0_05": qwk_comparison["bootstrap_95ci"][0] >= -0.05,
                 "P_dup_retention_ge_0_95": metrics["P"]["stability"]["T-dup"]["state_retention"] >= 0.95,
                 "P_irr_retention_ge_0_95": metrics["P"]["stability"]["T-irr"]["state_retention"] >= 0.95,
                 "P_order_retention_ge_0_95": metrics["P"]["stability"]["T-order"]["state_retention"] >= 0.95,
                 "P_all_three_TOST": all(metrics["P"]["stability"][variant]["tost"]["equivalent"] for variant in ["T-dup", "T-irr", "T-order"]),
                 "P_non_target_TOST": locality["P"]["pooled_tost"]["equivalent"],
                 "P_conflict_review_ge_0_85": metrics["P"]["conflict_human_review_recall"] >= 0.85,
                 "no_fabricated_source_failure": metrics["P"]["evidence"]["hallucinated_count"] == 0,
                 "P_high_risk_evidence_overlap_absent": safety["passed"]}
    decision = "GO" if all(go_checks.values()) else "MODIFY"; write_json(RESULTS / "go_decision.json", {"checks": go_checks, "decision": decision})
    refresh_frozen_manifest()
    print(json.dumps({"status": "complete", "go_decision": decision, "P_macro_f1": metrics["P"]["state"]["macro_f1"],
                      "P_ewr": metrics["P"]["ewr"], "P_qwk": metrics["P"]["score"]["qwk"], "primary_ci": primary["paired_bootstrap_95ci"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-only", action="store_true",
                        help="refresh frozen file hashes without recomputing statistics")
    parser.add_argument("--postprocess-only", action="store_true",
                        help="recompute stability/locality/safety artifacts without bootstrap")
    arguments = parser.parse_args()
    if arguments.manifest_only and arguments.postprocess_only:
        parser.error("choose only one lightweight mode")
    if arguments.postprocess_only:
        print(json.dumps(refresh_postprocess_outputs(), ensure_ascii=False))
    elif arguments.manifest_only:
        count = refresh_frozen_manifest()
        print(json.dumps({"status": "manifest_refreshed", "files": count}, ensure_ascii=False))
    else:
        main()
