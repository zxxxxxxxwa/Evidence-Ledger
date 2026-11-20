import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "method_validation" / "analysis"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))
SPEC = importlib.util.spec_from_file_location(
    "run_pilot_annotations_tested", ANALYSIS / "run_pilot_annotations.py")
PILOT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PILOT)


def record(text: str = "逐字证据材料。") -> dict:
    return {
        "sample_id": "SID-TEST",
        "indicator_id": "T_ALIGN",
        "indicator_title": "测试指标",
        "critical_observables": ["observable-a"],
        "other_observable": "observable-b",
        "goal_indicator_id": "T_ALIGN",
        "documents": [{
            "doc_id": "DOC-TEST", "title": "测试文档", "text": text,
            "material_type": "project", "source_ref": "SRC-TEST",
            "source_date": "2025-01-01", "contribution_level": "lead",
        }],
    }


class PilotCacheFingerprintTests(unittest.TestCase):
    def test_request_fingerprint_is_deterministic_and_input_sensitive(self):
        first = PILOT.request_metadata(1, [record()])
        second = PILOT.request_metadata(1, [record()])
        changed = PILOT.request_metadata(1, [record("另一段逐字证据材料。")])
        self.assertEqual(first, second)
        self.assertNotEqual(first["input_records_sha256"], changed["input_records_sha256"])
        self.assertNotEqual(first["request_sha256"], changed["request_sha256"])

    def test_request_fingerprint_changes_with_prompt_model_and_temperature(self):
        baseline = PILOT.request_metadata(1, [record()])
        with patch.object(PILOT, "PRELIMINARY_RUBRIC", PILOT.PRELIMINARY_RUBRIC + "\n新增规则"):
            rubric = PILOT.request_metadata(1, [record()])
        with patch.object(PILOT, "MODEL", "different-model"):
            model = PILOT.request_metadata(1, [record()])
        with patch.dict(PILOT.TEMPERATURES, {1: 0.77}):
            temperature = PILOT.request_metadata(1, [record()])
        self.assertNotEqual(baseline["request_sha256"], rubric["request_sha256"])
        self.assertNotEqual(baseline["request_sha256"], model["request_sha256"])
        self.assertNotEqual(baseline["request_sha256"], temperature["request_sha256"])

    def test_response_hashes_are_verified(self):
        expected = PILOT.request_metadata(1, [record()])
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            parsed, raw, meta = base / "batch.json", base / "batch.txt", base / "batch.meta.json"
            parsed.write_text('{"records": []}\n', encoding="utf-8")
            raw.write_text('{"records": []}\n', encoding="utf-8")
            PILOT.save_cache_metadata(meta, expected, parsed, raw)
            metadata = PILOT.json.loads(meta.read_text(encoding="utf-8"))
            self.assertTrue(PILOT.cache_metadata_matches(metadata, expected, parsed, raw))
            raw.write_text('{"records": ["changed"]}\n', encoding="utf-8")
            self.assertFalse(PILOT.cache_metadata_matches(metadata, expected, parsed, raw))


if __name__ == "__main__":
    unittest.main()
