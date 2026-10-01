import json
import tempfile
import unittest
from pathlib import Path

import audit_requirements as audit


HERE = Path(__file__).resolve().parent


class RequirementsAuditTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((HERE / "requirements_status.json").read_text(encoding="utf-8"))

    def test_ids_and_statuses_are_valid(self):
        rows = self.spec["requirements"]
        ids = [row["id"] for row in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, [f"R{i:02d}" for i in range(1, len(ids) + 1)])
        self.assertTrue(all(row["code_status"] in {"complete", "partial", "missing"} for row in rows))

    def test_all_declared_evidence_paths_exist(self):
        missing = []
        for row in self.spec["requirements"]:
            for evidence in row["evidence"]:
                if not audit.resolve_evidence(evidence).exists():
                    missing.append((row["id"], evidence))
        self.assertEqual(missing, [])

    def test_status_reader_accepts_controller_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.jsonl"
            path.write_text('{"step":"probe","status":"success","detail":"ok"}\n', encoding="utf-8")
            self.assertEqual(audit.load_run_status(Path(directory))["probe"]["status"], "success")


if __name__ == "__main__":
    unittest.main()
