from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from call_llm import chat


DATA = ROOT / "method_validation" / "data"
PRED = ROOT / "method_validation" / "predictions"
RESULTS = ROOT / "method_validation" / "results"
OUT = PRED / "path_ratings"
METHODS = ["B1", "B2", "B3", "P", "P_minus_Family", "P_minus_Route"]
CORE_PATH_METHODS = ["B1", "B2", "B3", "P"]
MODEL = "qwen3.7-flash"
TEMPERATURE = 0.25
MAX_TOKENS = 16000
TIMEOUT = 180
BATCH_SIZE = 12
SEED = 20261119

ROLE_PROMPTS = {
    1: "你是高校教师发展课程设计评审者，重点检查发展目标是否匹配以及活动是否可以执行。",
    2: "你是教育测量与学习进阶评审者，重点检查先修关系、难度递进和建议是否越过证据边界。",
    3: "你是资源治理与事实核查评审者，重点检查来源支持、资源版本与路径陈述的一致性。",
}

RUBRIC = """你将评价已经匿名化的教师发展路径。你不知道生成方法，也不得猜测方法。
每项用整数 1–5 分：
1. target_match：路径是否直接服务给定指标与目标；
2. prerequisite_coherence：步骤是否具有合理先修和递进；
3. executability：任务、资源和产出是否具体可执行；
4. source_support：引用资源能否支持对应建议，来源是否可识别。
只根据给出的路径文本和资源信息评分；不要因文风、长度或匿名编号加减分。
返回严格 JSON：{\"ratings\":[{\"blind_output_id\":\"...\",\"target_match\":1,\"prerequisite_coherence\":1,\"executability\":1,\"source_support\":1,\"brief_reason\":\"...\"}]}
每个输入 blind_output_id 必须且只能出现一次。"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def content_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def gold_value(row: dict[str, Any], key: str) -> Any:
    return row.get(f"gold_{key}", row.get(key))


def load_reference() -> dict[str, dict[str, Any]]:
    path = DATA / "final_reference.jsonl"
    return {row["sample_id"]: row for row in read_jsonl(path)}


def prediction_path(method: str) -> Path:
    candidates = [PRED / method / "aggregate.jsonl", PRED / method / "main_aggregate.jsonl"]
    return next((path for path in candidates if path.exists()), candidates[0])


def load_resource_catalog() -> dict[str, dict[str, Any]]:
    path = ROOT / "app" / "data" / "resources.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("resources", payload) if isinstance(payload, dict) else payload
    return {str(row["id"]): row for row in rows}


def render_resource(resource: dict[str, Any]) -> str:
    rid = resource.get("id") or resource.get("resource_id") or ""
    title = resource.get("title_zh") or resource.get("title") or ""
    publisher = resource.get("publisher") or resource.get("source") or ""
    summary = resource.get("summary_zh") or resource.get("summary_en") or resource.get("summary") or ""
    prerequisites = resource.get("prerequisites") or []
    url = resource.get("url") or ""
    duration = resource.get("duration_minutes")
    return (f"资源 {rid}｜{title}\n发布者：{publisher}；链接：{url}\n"
            f"摘要：{summary}\n先修资源：{prerequisites or '无'}；建议时长：{duration or '未注明'}分钟")


def path_text(pathway: Any, catalog: dict[str, dict[str, Any]]) -> str:
    if isinstance(pathway, str):
        return pathway.strip()
    if not isinstance(pathway, dict):
        return "未提供可评价的路径。"
    lines: list[str] = []
    for key in ("path_text", "text", "narrative", "recommendation"):
        if isinstance(pathway.get(key), str) and pathway[key].strip():
            lines.append("路径说明：" + pathway[key].strip())
            break
    steps = pathway.get("steps")
    if isinstance(steps, list):
        for index, step in enumerate(steps, 1):
            if isinstance(step, dict):
                title = step.get("title") or step.get("objective") or step.get("task") or step.get("description") or ""
                detail = step.get("summary") or step.get("reason") or ""
                practice = step.get("practice_task") or ""
                evidence = step.get("completion_evidence") or step.get("output") or ""
                duration = step.get("duration_minutes") or "未注明"
                lines.append(f"步骤{index}：{title}。理由：{detail}。实践任务：{practice}。完成证据：{evidence}。时长：{duration}分钟。")
            elif str(step).strip():
                lines.append(f"步骤{index}：{step}")
    resources = pathway.get("resources")
    rendered_ids: set[str] = set()
    if isinstance(resources, list):
        for resource in resources:
            if isinstance(resource, dict):
                rid = str(resource.get("id") or resource.get("resource_id") or "")
                enriched = {**catalog.get(rid, {}), **resource}
                lines.append(render_resource(enriched))
                rendered_ids.add(rid)
    ids = pathway.get("resource_ids")
    if isinstance(ids, list) and ids:
        for rid in map(str, ids):
            if rid not in rendered_ids:
                lines.append(render_resource(catalog.get(rid, {"id": rid})))
    if not lines:
        lines.append(json.dumps(pathway, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines)


def build_blind_items(max_per_method: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference = load_reference()
    rng = random.Random(SEED)
    catalog = load_resource_catalog()
    all_predictions = {method: {row["sample_id"]: row for row in read_jsonl(prediction_path(method))}
                       for method in METHODS}
    common_ids = set.intersection(*(set(all_predictions[method]) for method in CORE_PATH_METHODS))
    common_ids = {sample_id for sample_id in common_ids
                  if sample_id in reference and gold_value(reference[sample_id], "state") == "E"
                  and all(all_predictions[method][sample_id].get("action") == gold_value(reference[sample_id], "action")
                          for method in CORE_PATH_METHODS)}


    candidate_ids = sorted(common_ids)
    rng.shuffle(candidate_ids)
    selected_ids: list[str] = []
    seen_clusters: set[str] = set()
    for sample_id in candidate_ids:
        unit = str(reference[sample_id].get("teacher_cluster", reference[sample_id].get("teacher_id", "")))
        if unit in seen_clusters:
            continue
        selected_ids.append(sample_id)
        seen_clusters.add(unit)
        if len(selected_ids) >= max_per_method:
            break
    items: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for method in METHODS:
        for sample_id in selected_ids:
            prediction, ref = all_predictions[method][sample_id], reference[sample_id]



            if method not in CORE_PATH_METHODS and prediction.get("action") != gold_value(ref, "action"):
                continue
            blind_id = "R" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(11))
            items.append({
                "blind_output_id": blind_id,
                "target_indicator": ref.get("indicator_title") or ref.get("indicator_id") or prediction.get("indicator_id"),
                "development_goal": ref.get("goal_text") or ref.get("goal_indicator_id") or ref.get("indicator_title") or "围绕目标指标形成可执行的发展进阶",
                "pathway": path_text(prediction.get("path", prediction.get("pathway", {})), catalog),
            })
            mapping.append({
                "blind_output_id": blind_id,
                "method": method,
                "sample_id": prediction["sample_id"],
                "teacher_cluster": ref.get("teacher_cluster", ref.get("teacher_id", prediction.get("teacher_id"))),
            })
    rng.shuffle(items)
    return items, mapping


def parse_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(cleaned[start : end + 1])
    if not isinstance(obj.get("ratings"), list):
        raise ValueError("response lacks ratings list")
    return obj


def validate(obj: dict[str, Any], expected: list[str]) -> list[dict[str, Any]]:
    rows = obj["ratings"]
    if sorted(str(row.get("blind_output_id")) for row in rows) != sorted(expected):
        raise ValueError("blind output IDs do not match batch")
    for row in rows:
        for dimension in ("target_match", "prerequisite_coherence", "executability", "source_support"):
            value = row.get(dimension)
            if not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(f"invalid {dimension}: {value!r}")
    return rows


def rate_batch(round_no: int, batch_no: int, batch: list[dict[str, Any]]) -> dict[str, Any]:
    round_dir = OUT / f"round_{round_no}"
    round_dir.mkdir(parents=True, exist_ok=True)
    raw_path = round_dir / f"batch_{batch_no:03d}.txt"
    parsed_path = round_dir / f"batch_{batch_no:03d}.json"
    meta_path = round_dir / f"batch_{batch_no:03d}.meta.json"
    expected = [item["blind_output_id"] for item in batch]
    system_prompt = ROLE_PROMPTS[round_no] + " 独立完成本轮，不参考其他评审轮次。"
    prompt = RUBRIC + "\n\n待评价路径：\n" + json.dumps(
        batch, ensure_ascii=False, separators=(",", ":")
    )
    request_payload = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "timeout_seconds": TIMEOUT,
        "system_prompt": system_prompt,
        "user_prompt": prompt,
    }
    request_sha256 = content_sha256(request_payload)
    if parsed_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("request_sha256") != request_sha256:
                raise ValueError("cached rating request fingerprint differs")
            validate(json.loads(parsed_path.read_text(encoding="utf-8")), expected)
            return {
                "round": round_no, "batch": batch_no, "cached": True,
                "request_sha256": request_sha256,
            }
        except Exception:
            pass
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=TIMEOUT,
            )
            raw_path.write_text(response.rstrip() + "\n", encoding="utf-8")
            parsed = parse_response(response)
            validate(parsed, expected)
            write_json(parsed_path, parsed)
            write_json(meta_path, {
                "request_sha256": request_sha256,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "timeout_seconds": TIMEOUT,
                "blind_output_ids": expected,
            })
            return {
                "round": round_no, "batch": batch_no, "cached": False,
                "attempt": attempt, "request_sha256": request_sha256,
            }
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"round {round_no} batch {batch_no}: {last_error}")


def aggregate(items: list[dict[str, Any]], mapping: list[dict[str, Any]], n_batches: int) -> None:
    map_by_id = {row["blind_output_id"]: row for row in mapping}
    all_rows: list[dict[str, Any]] = []
    for round_no in (1, 2, 3):
        round_rows: list[dict[str, Any]] = []
        for batch_no in range(n_batches):
            obj = json.loads((OUT / f"round_{round_no}" / f"batch_{batch_no:03d}.json").read_text(encoding="utf-8"))
            for rating in obj["ratings"]:
                round_rows.append({**rating, **map_by_id[rating["blind_output_id"]], "rater_round": round_no})
        round_rows.sort(key=lambda row: row["blind_output_id"])
        write_jsonl(OUT / f"round_{round_no}.jsonl", round_rows)
        all_rows.extend(round_rows)
    write_jsonl(RESULTS / "path_ratings_raw.jsonl", all_rows)
    dimensions = ["target_match", "prerequisite_coherence", "executability", "source_support"]
    summary: dict[str, Any] = {}
    for method in METHODS:
        rows = [row for row in all_rows if row["method"] == method]
        summary[method] = {
            "unique_paths": len({row["blind_output_id"] for row in rows}),
            "ratings": len(rows),
            **{dimension: statistics.mean(float(row[dimension]) for row in rows) if rows else None for dimension in dimensions},
            "overall": statistics.mean(float(row[dimension]) for row in rows for dimension in dimensions) if rows else None,
        }
    write_json(RESULTS / "path_ratings_summary.json", {
        "design": "three independent method-blind LLM rating rounds",
        "model": MODEL,
        "temperature": TEMPERATURE,
        "dimensions": dimensions,
        "summary": summary,
    })


def make_separated_batches(items: list[dict[str, Any]], mapping: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    sample_by_blind = {row["blind_output_id"]: row["sample_id"] for row in mapping}
    remaining = list(items)
    batches: list[list[dict[str, Any]]] = []
    while remaining:
        batch: list[dict[str, Any]] = []
        used_samples: set[str] = set()
        deferred: list[dict[str, Any]] = []
        for item in remaining:
            sample_id = sample_by_blind[item["blind_output_id"]]
            if len(batch) < BATCH_SIZE and sample_id not in used_samples:
                batch.append(item)
                used_samples.add(sample_id)
            else:
                deferred.append(item)
        if not batch:
            batch, deferred = deferred[:BATCH_SIZE], deferred[BATCH_SIZE:]
        batches.append(batch)
        remaining = deferred
    return batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-per-method", type=int, default=24)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    items, mapping = build_blind_items(args.max_per_method)
    if not items:
        raise RuntimeError("no eligible paths were found")
    write_jsonl(OUT / "blind_items.jsonl", items)
    write_jsonl(OUT / "blind_mapping.jsonl", mapping)
    (DATA / "path_rating_prompt.txt").write_text(RUBRIC + "\n", encoding="utf-8")
    run_config = {
        "model": MODEL, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
        "timeout_seconds": TIMEOUT, "batch_size": BATCH_SIZE, "round_roles": ROLE_PROMPTS,
        "max_paths_per_method": args.max_per_method, "seed": SEED,
        "candidate_order": "sorted_sample_id_then_seeded_shuffle",
        "cache_validation": "sha256(canonical model configuration + system prompt + full user prompt)",
        "blind_items_sha256": content_sha256(items),
        "blind_mapping_sha256": content_sha256(mapping),
    }
    write_json(OUT / "run_config.json", run_config)
    batches = make_separated_batches(items, mapping)
    tasks = [(round_no, batch_no, batch) for round_no in (1, 2, 3) for batch_no, batch in enumerate(batches)]
    task_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 24))) as pool:
        futures = [pool.submit(rate_batch, *task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            task_results.append(result)
            print(json.dumps({"completed": completed, "total": len(tasks), **result}, ensure_ascii=False), flush=True)
    run_config["batches"] = len(batches)
    run_config["rating_requests"] = len(task_results)
    run_config["cache_hits"] = sum(bool(item["cached"]) for item in task_results)
    run_config["cache_misses"] = sum(not bool(item["cached"]) for item in task_results)
    run_config["request_sha256"] = sorted(item["request_sha256"] for item in task_results)
    write_json(OUT / "run_config.json", run_config)
    aggregate(items, mapping, len(batches))
    print(json.dumps({"status": "complete", "unique_paths": len(items), "ratings": len(items) * 3}, ensure_ascii=False))


if __name__ == "__main__":
    main()
