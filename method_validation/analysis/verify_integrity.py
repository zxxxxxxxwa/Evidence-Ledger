from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "method_validation" / "data"
PRED = ROOT / "method_validation" / "predictions"
ANALYSIS = ROOT / "method_validation" / "analysis"

TASKS = {
    "main": ("final_test_records.jsonl", "final_reference.jsonl", 288),
    "c0_all": ("c0_all_indicator_records.jsonl", "c0_all_reference.jsonl", 384),
    "locality": ("locality_records.jsonl", "locality_reference.jsonl", 240),
}
METHODS = ("B1", "B2", "B3", "P", "P_minus_Family", "P_minus_Route")
FORBIDDEN_TOP_LEVEL = {
    "gold_state", "gold_level", "gold_action", "gold_evidence_families",
    "gold_reason_code", "state", "level", "action", "condition", "version",
    "variant", "teacher_id", "teacher_cluster", "case_no", "base_level",
}
FORBIDDEN_DOCUMENT_KEYS = {
    "family_key", "role", "polarity", "observables", "prompt_injection",
    "gold", "condition", "variant",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise AssertionError(f"missing file: {path.relative_to(ROOT)}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def signature(documents: list[dict[str, Any]]) -> str:
    payload = json.dumps(documents, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_task(task: str, input_name: str, reference_name: str, expected: int) -> None:
    inputs = read_jsonl(DATA / input_name)
    references = read_jsonl(DATA / reference_name)
    assert len(inputs) == expected, (task, len(inputs), expected)
    assert len(references) == expected, (task, len(references), expected)
    input_ids = [str(row["sample_id"]) for row in inputs]
    reference_ids = [str(row["sample_id"]) for row in references]
    assert len(input_ids) == len(set(input_ids)), f"duplicate input ID in {task}"
    assert len(reference_ids) == len(set(reference_ids)), f"duplicate reference ID in {task}"
    assert set(input_ids) == set(reference_ids), f"input/reference mismatch in {task}"
    reference_map = {str(row["sample_id"]): row for row in references}
    for row in inputs:
        leaked = FORBIDDEN_TOP_LEVEL & set(row)
        assert not leaked, f"{task}/{row['sample_id']} top-level leak: {sorted(leaked)}"
        assert re.fullmatch(r"SID-[0-9A-F]{18}", str(row["sample_id"])), row["sample_id"]
        for document in row.get("documents", []):
            leaked_document = FORBIDDEN_DOCUMENT_KEYS & set(document)
            assert not leaked_document, (
                f"{task}/{row['sample_id']} document leak: {sorted(leaked_document)}"
            )


    for start in range(0, len(inputs), 12):
        clusters = [
            str(reference_map[str(row["sample_id"])].get("teacher_cluster", ""))
            for row in inputs[start : start + 12]
        ]
        assert all(clusters), f"missing teacher_cluster in {task} reference"
        assert len(clusters) == len(set(clusters)), f"same cluster repeated in {task} batch {start // 12}"


def check_design_relations() -> None:
    main_reference = read_jsonl(DATA / "final_reference.jsonl")
    counts = Counter(str(row.get("condition")) for row in main_reference)
    expected = {condition: 48 for condition in ("C0", "T-del", "T-dup", "T-irr", "T-order", "T-conflict")}
    assert counts == expected, counts

    c0_inputs = {str(row["sample_id"]): row for row in read_jsonl(DATA / "c0_all_indicator_records.jsonl")}
    c0_reference = read_jsonl(DATA / "c0_all_reference.jsonl")
    portfolio_signatures: dict[str, set[str]] = defaultdict(set)
    indicators: dict[str, set[str]] = defaultdict(set)
    for reference in c0_reference:
        cluster = str(reference["teacher_cluster"])
        input_row = c0_inputs[str(reference["sample_id"])]
        portfolio_signatures[cluster].add(signature(input_row["documents"]))
        indicators[cluster].add(str(input_row["indicator_id"]))
    assert len(portfolio_signatures) == 48
    assert all(len(values) == 1 for values in portfolio_signatures.values())
    assert all(len(values) == 8 for values in indicators.values())

    locality_inputs = {str(row["sample_id"]): row for row in read_jsonl(DATA / "locality_records.jsonl")}
    locality_reference = read_jsonl(DATA / "locality_reference.jsonl")
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for reference in locality_reference:
        assert reference.get("baseline_sample_id") in c0_inputs
        by_cluster[str(reference["teacher_cluster"])].append(reference)
    assert len(by_cluster) == 48
    assert all(len(rows) == 5 for rows in by_cluster.values())
    for cluster, references in by_cluster.items():
        indicators_for_cluster = {
            locality_inputs[str(reference["sample_id"])]["indicator_id"] for reference in references
        }
        assert len(indicators_for_cluster) == 1, cluster

    split_reference_files = {
        "development": "development_reference.jsonl",
        "calibration": "construction_reference_calibration.jsonl",
        "test": "construction_reference_main.jsonl",
    }
    split_clusters = {
        split: {str(row["teacher_cluster"]) for row in read_jsonl(DATA / filename)}
        for split, filename in split_reference_files.items()
    }
    assert len(split_clusters["development"]) == 36
    assert len(split_clusters["calibration"]) == 12
    assert len(split_clusters["test"]) == 48
    assert split_clusters["development"].isdisjoint(split_clusters["calibration"])
    assert split_clusters["development"].isdisjoint(split_clusters["test"])
    assert split_clusters["calibration"].isdisjoint(split_clusters["test"])


def check_prediction_artifacts() -> None:
    suffixes = {"main": "aggregate.jsonl", "c0_all": "c0_all_aggregate.jsonl", "locality": "locality_aggregate.jsonl"}
    expected = {task: size for task, (_, _, size) in TASKS.items()}
    input_ids = {
        task: {str(row["sample_id"]) for row in read_jsonl(DATA / input_name)}
        for task, (input_name, _, _) in TASKS.items()
    }
    for method in METHODS:
        for task, suffix in suffixes.items():
            rows = read_jsonl(PRED / method / suffix)
            assert len(rows) == expected[task], (method, task, len(rows))
            assert {str(row["sample_id"]) for row in rows} == input_ids[task]


def check_source_boundaries() -> None:
    source = (ANALYSIS / "run_methods.py").read_text(encoding="utf-8")
    forbidden_patterns = {
        "case-number prediction branch": r"row\s*\[\s*['\"]case_no['\"]\s*\]",
        "condition prediction branch": r"row\s*\[\s*['\"](?:version|condition|variant)['\"]\s*\]",
        "construction safety flag": r"(?:doc|document)\.get\(\s*['\"]prompt_injection['\"]",
        "test reference opened by predictor": r"final_reference\.jsonl",
    }
    for label, pattern in forbidden_patterns.items():
        assert not re.search(pattern, source), label


def check_frozen_manifest() -> None:
    manifest = json.loads((DATA / "frozen_manifest.json").read_text(encoding="utf-8"))
    for relative_path, expected_digest in manifest["files"].items():
        path = ROOT / relative_path
        assert path.is_file(), f"missing frozen artifact: {relative_path}"
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_digest == expected_digest, f"frozen artifact changed: {relative_path}"


def main() -> None:
    check_frozen_manifest()
    for task, spec in TASKS.items():
        check_task(task, *spec)
    check_design_relations()
    check_source_boundaries()
    check_prediction_artifacts()
    print(json.dumps({"status": "passed", "tasks": TASKS, "methods": METHODS}, ensure_ascii=False))


if __name__ == "__main__":
    main()
