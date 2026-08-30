import json
import tempfile
import unittest
from pathlib import Path

from tools.research.manage_ideas import (
    MANIFEST_SCHEMA_VERSION,
    RegistryError,
    apply_artifact_prune,
    apply_retirement,
    artifact_prune_plan,
    retirement_plan,
    validate_registry,
)


class ResearchIdeaLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "docs/research/demo").mkdir(parents=True)
        (self.root / "tools/research/demo").mkdir(parents=True)
        (self.root / "test/research/demo").mkdir(parents=True)
        (self.root / "logs/demo").mkdir(parents=True)
        (self.root / "docs/research/demo/idea.md").write_text("claim\n")
        (self.root / "tools/research/demo/run.py").write_text("pass\n")
        (self.root / "test/research/demo/test_demo.py").write_text("pass\n")
        (self.root / "logs/demo/report.json").write_text("{}\n")
        (self.root / "logs/demo/raw.pt").write_bytes(b"raw")
        (self.root / "docs/research/EXPERIENCE_LOG.md").write_text(
            "# log\n\n## EXP-DEMO：decision\n")
        self.registry = {
            "schema_version": "research-idea-registry-v1",
            "reserved_ids": ["o2"],
            "ideas": [{
                "id": "demo",
                "title": "demo",
                "status": "rejected",
                "claim": "demo",
                "verdict": "NOT_SUPPORTED",
                "experience_entry": "EXP-DEMO",
                "owned_paths": [
                    "docs/research/demo",
                    "tools/research/demo",
                    "test/research/demo",
                ],
                "preserved_evidence": ["logs/demo/report.json"],
                "discard_artifacts_on_retire": ["logs/demo/raw.pt"],
                "retired_on": None,
            }],
        }
        (self.root / "docs/research/registry.json").write_text(
            json.dumps(self.registry))

    def tearDown(self):
        self.temporary.cleanup()

    def test_dry_run_is_non_destructive_and_exact(self):
        plan = retirement_plan(self.root, "demo")
        self.assertFalse(plan["prune_artifacts"])
        self.assertEqual(plan["artifact_paths"], [])
        self.assertEqual(plan["source_paths"], [
            "docs/research/demo", "tools/research/demo", "test/research/demo",
        ])
        self.assertTrue((self.root / "tools/research/demo/run.py").is_file())

    def test_apply_retires_source_and_keeps_small_evidence(self):
        plan = apply_retirement(
            self.root, "demo", prune_artifacts=False, retired_on="2026-08-26")
        self.assertFalse(plan["prune_artifacts"])
        self.assertFalse((self.root / "docs/research/demo").exists())
        self.assertFalse((self.root / "tools/research/demo").exists())
        self.assertFalse((self.root / "test/research/demo").exists())
        self.assertTrue((self.root / "logs/demo/raw.pt").is_file())
        self.assertTrue((self.root / "logs/demo/report.json").is_file())
        value = json.loads(
            (self.root / "docs/research/registry.json").read_text())
        self.assertEqual(value["ideas"][0]["status"], "retired")
        self.assertEqual(value["ideas"][0]["retired_on"], "2026-08-26")
        validate_registry(self.root)

        prune = artifact_prune_plan(self.root, "demo")
        self.assertEqual(prune["existing_artifact_paths"], ["logs/demo/raw.pt"])
        apply_artifact_prune(self.root, "demo")
        self.assertFalse((self.root / "logs/demo/raw.pt").exists())
        self.assertTrue((self.root / "logs/demo/report.json").is_file())

    def test_missing_experience_card_is_rejected(self):
        (self.root / "docs/research/EXPERIENCE_LOG.md").write_text("# empty\n")
        with self.assertRaisesRegex(RegistryError, "experience card"):
            validate_registry(self.root)

    def test_owned_path_escape_and_reserved_id_are_rejected(self):
        self.registry["ideas"][0]["owned_paths"] = ["docs/research/other"]
        with self.assertRaisesRegex(RegistryError, "must equal"):
            validate_registry(self.root, self.registry)
        with self.assertRaisesRegex(RegistryError, "Reserved"):
            retirement_plan(self.root, "o2")

    def test_branch_manifest_is_discovered_and_archived_on_retirement(self):
        idea_id = "fresh"
        for parent in (
            "docs/research", "tools/research", "test/research",
        ):
            (self.root / parent / idea_id).mkdir(parents=True)
        with (self.root / "docs/research/EXPERIENCE_LOG.md").open("a") as stream:
            stream.write("\n## EXP-FRESH：decision\n")
        idea = {
            "id": idea_id,
            "title": "fresh",
            "status": "rejected",
            "claim": "fresh",
            "verdict": "NOT_SUPPORTED",
            "experience_entry": "EXP-FRESH",
            "base_tag": "obb-o2-baseline-v1",
            "base_commit": "0" * 40,
            "branch": f"idea/{idea_id}",
            "owned_paths": [
                f"docs/research/{idea_id}",
                f"tools/research/{idea_id}",
                f"test/research/{idea_id}",
            ],
            "preserved_evidence": [],
            "discard_artifacts_on_retire": [],
            "retired_on": None,
        }
        (self.root / f"docs/research/{idea_id}/manifest.json").write_text(
            json.dumps({
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "idea": idea,
            }))

        self.assertIn(idea_id, validate_registry(self.root))
        apply_retirement(self.root, idea_id, retired_on="2026-08-30")
        self.assertFalse((self.root / f"docs/research/{idea_id}").exists())
        archived = json.loads(
            (self.root / "docs/research/registry.json").read_text())
        archived_idea = next(
            item for item in archived["ideas"] if item["id"] == idea_id)
        self.assertEqual(archived_idea["status"], "retired")
        self.assertEqual(archived_idea["retired_on"], "2026-08-30")
        validate_registry(self.root)


if __name__ == "__main__":
    unittest.main()
