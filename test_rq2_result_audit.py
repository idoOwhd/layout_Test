import unittest
from analyze_rq2_graph_growth import audit_input_fingerprints


class ResultAuditTests(unittest.TestCase):
    def row(self,framework,value="same"):
        return {"record_type":"group_data_fingerprint","framework":framework,
                "group":["real_model","decode",4,1,512,[512,257,127,67]],
                "fingerprint":{"sample_digest":value}}
    def test_identical_inputs_can_be_compared(self):
        v=audit_input_fingerprints([[self.row("triton")],[self.row("vllm")]])
        self.assertEqual(v["compared_groups"],1);self.assertEqual(v["mismatching_groups"],0)
    def test_mismatch_is_not_hidden(self):
        v=audit_input_fingerprints([[self.row("triton")],[self.row("sglang","changed")]])
        self.assertEqual(v["mismatching_groups"],1)
    def test_single_framework_is_not_cross_framework_evidence(self):
        v=audit_input_fingerprints([[self.row("triton")]])
        self.assertEqual(v["compared_groups"],0)
    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):audit_input_fingerprints([[self.row("triton"),self.row("triton")]])


if __name__=="__main__":unittest.main()
