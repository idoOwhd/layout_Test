import unittest

import audit_ref_talks_evidence as audit


class RefTalksEvidenceTest(unittest.TestCase):
    def test_canonical_rule_count_and_rq_coverage(self):
        rows = audit.parse_v14_rules(audit.CANONICAL)
        self.assertEqual(len(rows), 65)
        seen = {rq for row in rows for rq in row["rqs"]}
        self.assertEqual(seen, {f"RQ{i}" for i in range(1, 7)})

    def test_forensic_ledger_is_pinned(self):
        rows = audit.parse_v4_ledger(audit.LEDGER)
        self.assertEqual(len(rows), 50)
        for row in rows:
            self.assertRegex(row["sha"], r"^[0-9a-f]{40}$")
            self.assertIn(row["sha"], row["immutable_url"])

    def test_all_ten_rqs_have_source_coverage(self):
        report = audit.build_report(
            audit.parse_v14_rules(audit.CANONICAL),
            audit.parse_v4_ledger(audit.LEDGER),
            [],
        )
        self.assertTrue(report["source_chain_pass"])
        for i in range(1, 11):
            self.assertGreaterEqual(len(report["rq_coverage"][f"RQ{i}"]["frameworks"]), 2)


if __name__ == "__main__":
    unittest.main()
