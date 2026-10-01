import importlib.util
import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
VISION = HERE.parent / "vision_shapes"
SPEC = importlib.util.spec_from_file_location("build_vision_manifest", VISION / "build_vision_manifest.py")
builder = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(builder)
PIN_SPEC = importlib.util.spec_from_file_location("pin_vision_sources", VISION / "pin_vision_sources.py")
pinner = importlib.util.module_from_spec(PIN_SPEC); PIN_SPEC.loader.exec_module(pinner)


class VisionCatalogTests(unittest.TestCase):
    def setUp(self):
        self.corpus = json.loads((VISION / "vision_model_corpus.json").read_text(encoding="utf-8"))

    def test_modern_balanced_corpus(self):
        models = self.corpus["models"]
        self.assertEqual(len(models), 12)
        self.assertEqual(sum(x["domain"] == "image" for x in models), 6)
        self.assertEqual(sum(x["domain"] == "video" for x in models), 6)
        self.assertTrue(all(2025 <= x["year"] <= 2026 for x in models))

    def test_discussion_incidence_invariants(self):
        stats = builder.incidence(self.corpus["models"])
        self.assertEqual(stats["dit_block"]["all"]["rate"], 1.0)
        self.assertEqual(stats["dual_stream"]["image"]["count"], 4)
        self.assertEqual(stats["dual_single_hybrid"]["image"]["count"], 3)
        self.assertGreaterEqual(stats["rope"]["all"]["rate"], 2 / 3)
        self.assertGreaterEqual(stats["qk_norm"]["all"]["rate"], .5)
        self.assertGreaterEqual(stats["structured_attention"]["video"]["rate"], .5)

    def test_every_common_motif_has_multiple_shapes(self):
        manifest = json.loads((VISION / "vision_common_subgraph_manifest.json").read_text(encoding="utf-8"))
        counts = {}
        for row in manifest["cases"]: counts[row["motif"]] = counts.get(row["motif"], 0) + 1
        for motif in manifest["common_motifs"]:
            self.assertGreaterEqual(counts.get(motif, 0), 3, motif)
        self.assertTrue(all(row["shape_source"].startswith("declared benchmark policy") for row in manifest["cases"]))

    def test_shared_urls_are_not_concatenated(self):
        sources = json.loads((HERE / "shared_sources.json").read_text(encoding="utf-8"))
        urls = [x["url"] for x in sources["sources"]]
        self.assertEqual(len(urls), len(set(urls)))
        share_pattern = re.compile(r"https://chatgpt\.com/share/[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
        thread_pattern = re.compile(r"codex://threads/[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
        self.assertTrue(all(share_pattern.fullmatch(url) for url in urls))
        self.assertTrue(all(thread_pattern.fullmatch(x["url"]) for x in sources["local_references"]))

    @patch.object(pinner, "get_json")
    def test_source_pinner_builds_immutable_hf_and_github_urls(self, get_json):
        get_json.side_effect = [
            {"sha": "a" * 40},
            {"sha": "b" * 40},
            {"default_branch": "main"},
            {"sha": "c" * 40},
        ]
        hf_root = pinner.resolve({"model_id": "org/model", "source_url": "https://huggingface.co/org/model"})
        hf_file = pinner.resolve({"model_id": "org/model", "source_url": "https://huggingface.co/org/model/blob/main/config.json"})
        github = pinner.resolve({"model_id": "org/repo", "source_url": "https://github.com/org/repo"})
        self.assertEqual(hf_root["immutable_url"], f"https://huggingface.co/org/model/tree/{'a' * 40}")
        self.assertEqual(hf_file["immutable_url"], f"https://huggingface.co/org/model/blob/{'b' * 40}/config.json")
        self.assertEqual(github["immutable_url"], f"https://github.com/org/repo/tree/{'c' * 40}")


if __name__ == "__main__": unittest.main()
