#!/usr/bin/env python3
"""CPU-only tests for source extraction; these are NOT GPU RQ2 evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import archive_layout_shared_discussion as archive
import build_rq2_source_test_audit as audit
import build_rq2_real_graph_growth_cases as growth


class ArchiveTests(unittest.TestCase):
    def test_only_unique_visible_messages(self):
        array=[]
        def add(value):
            array.append(value);return len(array)-1
        def message(role,channel,text):
            author=add({"role":add(role)})
            content=add({"parts":add([add(text)])})
            add({"author":author,"content":content,"channel":add(channel)})
        message("user",None,"question")
        message("assistant","analysis","not visible")
        message("assistant","final","visible")
        message("assistant","final","visible")
        html="streamController.enqueue("+json.dumps(json.dumps(array))+");"
        self.assertEqual([(m["role"],m["text"]) for m in archive.messages(html)],
                         [("user","question"),("assistant","visible")])

    def test_catalog_canonical_key_includes_repo_not_link_kind(self):
        visible=[{"role":"assistant","text":
            "https://github.com/owner/repo/pull/42?x=1\n"
            "https://github.com/OWNER/REPO/issues/42#anchor\n"
            "https://github.com/owner/other/pull/42"}]
        catalog=archive.catalog(visible,"source","sha")
        self.assertEqual(catalog["unique_pr_issue_count"],2)
        self.assertEqual(len(catalog["records"][1]["aliases"]),2)
        self.assertTrue(catalog["records"][1]["occurrences"][0]["context"].startswith("https://"))

    def test_bare_chains_do_not_change_owner_after_descriptive_parenthesis(self):
        text="## SGLang\n#41848（AITER direct slices）、#38092、#40682。\n"
        explicit={"records":[]}
        rows=archive.unlinked_references([{"role":"assistant","text":text}],explicit)
        self.assertEqual({(r["repo"],r["number"]) for r in rows},
            {("sgl-project/sglang",n) for n in (41848,38092,40682)})

    def test_prefix_carries_across_slash_and_arrow(self):
        text="## Triton\n- CUTLASS #3030/#3453；\n| SGLang #40326→#38592→#40330 |\n"
        rows=archive.unlinked_references([{"role":"assistant","text":text}],{"records":[]})
        self.assertEqual({(r["repo"],r["number"]) for r in rows},
            {("NVIDIA/cutlass",3030),("NVIDIA/cutlass",3453)}|
            {("sgl-project/sglang",n) for n in (40326,38592,40330)})

    def test_linked_label_cannot_be_assigned_previous_section_owner(self):
        text="## CUTLASS\n| TileLang url#1386https://github.com/tile-ai/tilelang/pull/1386 / url#1336https://github.com/tile-ai/tilelang/issues/1336 |\n"
        visible=[{"role":"assistant","text":text}]
        explicit=archive.catalog(visible,"source","sha")
        self.assertEqual(archive.unlinked_references(visible,explicit),[])


class SourceAuditTests(unittest.TestCase):
    def test_dtype_hints_do_not_confuse_bfloat16_with_float16(self):
        self.assertNotIn("FP16",audit.dtype_hints("torch.bfloat16 BF16"))
        self.assertEqual(audit.dtype_hints("torch.half cutlass.half_t"),["FP16"])
        self.assertEqual(audit.dtype_hints("float32 tf32x3"),["FP32/TF32"])

    def test_unlisted_source_is_not_native_success(self):
        status,_=audit.restriction("owner/repo",12)
        self.assertEqual(status,"pending_fp16_contract_review")

    def test_pr_diff_preserves_paths_and_test_definition(self):
        with tempfile.TemporaryDirectory(prefix="rq2_source_test_") as temporary:
            path=Path(temporary)/"sample.diff"
            # Test fixture only; no production file is written here.
            path.write_text("diff --git a/tests/example.py b/tests/example.py\n+def test_fp16():\n+    pass\n")
            parsed=audit.patches_from_diff(path)
            self.assertEqual(parsed[0]["filename"],"tests/example.py")

    def test_complete_snapshot_has_no_missing_or_duplicate_primary_sources(self):
        directory=audit.HERE/"ref_talks/rq2_source_audit_20261001_139"
        catalog=json.loads((audit.HERE/"rq2_shared_github_catalog_20261001.json").read_text())
        if not directory.exists():self.skipTest("archived source snapshot absent")
        rows=audit.collect(directory,catalog)
        self.assertEqual(len(rows),207)
        self.assertEqual(len({(r["repo"].lower(),r["number"]) for r in rows}),207)
        self.assertEqual(sum(r["reference_kind"]=="explicit_link" for r in rows),139)
        self.assertEqual(sum(r["kind"]=="pull" and bool(r["changed_files"]) for r in rows),193)
        self.assertTrue(all(r["rq2_whole_graph_native_replay_status"]=="not_run" for r in rows))
        row=next(r for r in rows if r["repo"]=="flashinfer-ai/flashinfer" and r["number"]==5405)
        self.assertIn("tests/experimental/test_fused_qk_rope_append.py",row["changed_files"])

    def test_emit_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory(prefix="rq2_emit_test_") as temporary:
            with self.assertRaises(FileExistsError):audit.emit(Path(temporary),[],"source")


class RealGraphGrowthTests(unittest.TestCase):
    def setUp(self):
        # Geometry from pinned Qwen3-0.6B; this is not a new timing sample.
        self.config={"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,
                     "head_dim":128,"intermediate_size":3072,"rms_norm_eps":1e-6,
                     "rope_theta":1000000,"attention_bias":False}

    def test_explicit_head_dim_not_hidden_div_heads(self):
        nodes=growth.nodes_for_stage(self.config,1,1,4096,4)
        output=next(n for n in nodes if n["id"]=="L0.output_projection")
        self.assertEqual(output["attrs"]["weight_shape"],[2048,1024])

    def test_every_growth_step_preserves_old_nodes_math_and_shape(self):
        previous={}
        for stage in range(len(growth.STAGES)):
            nodes=growth.nodes_for_stage(self.config,4,8,4096,stage)
            self.assertTrue(growth.validate_nodes(nodes))
            now={n["id"]:n["semantic_signature"] for n in nodes}
            self.assertTrue(all(now.get(k)==v for k,v in previous.items()))
            self.assertGreater(len(now),len(previous));previous=now

    def test_state_versions_and_request_count_not_flattened_away(self):
        nodes=growth.nodes_for_stage(self.config,16,1,4096,3)
        attention=next(n for n in nodes if n["id"]=="L0.attention")
        self.assertIn("L0.kv_state_prev",attention["inputs"])
        self.assertEqual(attention["outputs"]["L0.kv_state_next"]["logical_shape"],
                         [2,16,4096,8,128])
        self.assertEqual(attention["attrs"]["request_count"],16)

    def test_dense_recipe_is_explicitly_not_all_framework_experimental_evidence(self):
        path=audit.HERE/"ref_talks/rq2_qwen3_real_forward_growth_design_20261001.json"
        if not path.exists():self.skipTest("design manifest absent")
        value=json.loads(path.read_text())
        self.assertEqual(value["case_count"],720)
        self.assertEqual(value["nested_pair_count"],648)
        self.assertEqual(value["new_gpu_experiments_completed"],0)
        self.assertTrue(all(c["layout_measurements"] is None for c in value["cases"]))


if __name__ == "__main__":unittest.main()
