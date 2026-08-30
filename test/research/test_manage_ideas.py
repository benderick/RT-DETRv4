import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.research.manage_ideas import (
    MANIFEST_SCHEMA_VERSION,
    RegistryError,
    apply_artifact_prune,
    apply_retirement,
    artifact_prune_plan,
    experience_log_path,
    initialize_ledger,
    record_experience,
    registry_path,
    research_ledger_root,
    retirement_plan,
    validate_registry,
)


class ResearchIdeaLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "docs/ideas/demo").mkdir(parents=True)
        (self.root / "tools/research/demo").mkdir(parents=True)
        (self.root / "test/research/demo").mkdir(parents=True)
        (self.root / "logs/demo").mkdir(parents=True)
        self.ledger = self.root / "research/ledger"
        self.ledger.mkdir(parents=True)
        (self.root / "docs/ideas/demo/idea.md").write_text("claim\n")
        (self.root / "tools/research/demo/run.py").write_text("pass\n")
        (self.root / "test/research/demo/test_demo.py").write_text("pass\n")
        (self.root / "logs/demo/report.json").write_text("{}\n")
        (self.root / "logs/demo/raw.pt").write_bytes(b"raw")
        (self.ledger / "EXPERIENCE_LOG.md").write_text(
            "# log\n\n## EXP-DEMO：decision\n")
        self.registry = {
            "schema_version": "research-idea-registry-v1",
            "reserved_ids": ["o2", "research_ledger"],
            "ideas": [{
                "id": "demo",
                "title": "demo",
                "status": "rejected",
                "claim": "demo",
                "verdict": "NOT_SUPPORTED",
                "verdict_scope": "scientific_claim",
                "experience_entry": "EXP-DEMO",
                "owned_paths": [
                    "docs/ideas/demo",
                    "tools/research/demo",
                    "test/research/demo",
                ],
                "preserved_evidence": ["logs/demo/report.json"],
                "discard_artifacts_on_retire": ["logs/demo/raw.pt"],
                "retired_on": None,
            }],
        }
        (self.ledger / "registry.json").write_text(
            json.dumps(self.registry))

    def tearDown(self):
        self.temporary.cleanup()

    def test_dry_run_is_non_destructive_and_exact(self):
        plan = retirement_plan(self.root, "demo")
        self.assertFalse(plan["prune_artifacts"])
        self.assertEqual(plan["artifact_paths"], [])
        self.assertEqual(plan["source_paths"], [
            "docs/ideas/demo", "tools/research/demo", "test/research/demo",
        ])
        self.assertTrue((self.root / "tools/research/demo/run.py").is_file())

    def test_apply_retires_source_and_keeps_small_evidence(self):
        plan = apply_retirement(
            self.root, "demo", prune_artifacts=False, retired_on="2026-08-26")
        self.assertFalse(plan["prune_artifacts"])
        self.assertFalse((self.root / "docs/ideas/demo").exists())
        self.assertFalse((self.root / "tools/research/demo").exists())
        self.assertFalse((self.root / "test/research/demo").exists())
        self.assertTrue((self.root / "logs/demo/raw.pt").is_file())
        self.assertTrue((self.root / "logs/demo/report.json").is_file())
        value = json.loads(
            (self.ledger / "registry.json").read_text())
        self.assertEqual(value["ideas"][0]["status"], "retired")
        self.assertEqual(value["ideas"][0]["retired_from"], "rejected")
        self.assertEqual(value["ideas"][0]["retired_on"], "2026-08-26")
        backup = json.loads((self.ledger / "registry.json.bak").read_text())
        self.assertEqual(backup["ideas"][0]["status"], "rejected")
        validate_registry(self.root)

        prune = artifact_prune_plan(self.root, "demo")
        self.assertEqual(prune["existing_artifact_paths"], ["logs/demo/raw.pt"])
        apply_artifact_prune(self.root, "demo")
        self.assertFalse((self.root / "logs/demo/raw.pt").exists())
        self.assertTrue((self.root / "logs/demo/report.json").is_file())

    def test_missing_experience_card_is_rejected(self):
        (self.ledger / "EXPERIENCE_LOG.md").write_text("# empty\n")
        with self.assertRaisesRegex(RegistryError, "experience card"):
            validate_registry(self.root)

    def test_decision_requires_explicit_verdict_scope(self):
        self.registry["ideas"][0].pop("verdict_scope")
        with self.assertRaisesRegex(RegistryError, "verdict_scope"):
            validate_registry(self.root, self.registry)

    def test_owned_path_escape_and_reserved_id_are_rejected(self):
        self.registry["ideas"][0]["owned_paths"] = ["docs/ideas/other"]
        with self.assertRaisesRegex(RegistryError, "must equal"):
            validate_registry(self.root, self.registry)
        with self.assertRaisesRegex(RegistryError, "Reserved"):
            retirement_plan(self.root, "o2")

    def test_branch_manifest_is_discovered_and_archived_on_retirement(self):
        idea_id = "fresh"
        for parent in (
            "docs/ideas", "tools/research", "test/research",
        ):
            (self.root / parent / idea_id).mkdir(parents=True)
        idea = {
            "id": idea_id,
            "title": "fresh",
            "status": "rejected",
            "claim": "fresh",
            "verdict": "NOT_SUPPORTED",
            "verdict_scope": "scientific_claim",
            "experience_entry": "EXP-FRESH",
            "base_tag": "obb-o2-baseline-v1",
            "base_commit": "0" * 40,
            "branch": f"idea/{idea_id}",
            "owned_paths": [
                f"docs/ideas/{idea_id}",
                f"tools/research/{idea_id}",
                f"test/research/{idea_id}",
            ],
            "preserved_evidence": [],
            "discard_artifacts_on_retire": [],
            "retired_on": None,
        }
        (self.root / f"docs/ideas/{idea_id}/manifest.json").write_text(
            json.dumps({
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "idea": idea,
            }))
        (self.root / f"docs/ideas/{idea_id}/experience.md").write_text(
            "## EXP-FRESH：decision\n\n- result\n")

        recorded = record_experience(self.root, idea_id)
        self.assertTrue(recorded["recorded"])
        self.assertTrue((self.ledger / "EXPERIENCE_LOG.md.bak").is_file())
        self.assertFalse(record_experience(self.root, idea_id)["recorded"])

        self.assertIn(idea_id, validate_registry(self.root))
        apply_retirement(self.root, idea_id, retired_on="2026-08-30")
        self.assertFalse((self.root / f"docs/ideas/{idea_id}").exists())
        archived = json.loads(
            (self.ledger / "registry.json").read_text())
        archived_idea = next(
            item for item in archived["ideas"] if item["id"] == idea_id)
        self.assertEqual(archived_idea["status"], "retired")
        self.assertEqual(archived_idea["retired_from"], "rejected")
        self.assertEqual(archived_idea["retired_on"], "2026-08-30")
        validate_registry(self.root)

    def test_inconclusive_can_retire_without_becoming_claim_rejection(self):
        idea = self.registry["ideas"][0]
        idea["status"] = "inconclusive"
        idea["verdict"] = "PROBE_NOT_FAITHFUL"
        idea["verdict_scope"] = "probe"
        (self.ledger / "registry.json").write_text(json.dumps(self.registry))

        plan = retirement_plan(self.root, "demo")
        self.assertEqual(plan["status_before"], "inconclusive")
        apply_retirement(self.root, "demo", retired_on="2026-08-30")
        archived = json.loads(
            (self.ledger / "registry.json").read_text())["ideas"][0]
        self.assertEqual(archived["status"], "retired")
        self.assertEqual(archived["retired_from"], "inconclusive")
        self.assertEqual(archived["verdict_scope"], "probe")

    def test_parked_requires_resume_condition_and_cannot_retire(self):
        idea = self.registry["ideas"][0]
        idea["status"] = "parked"
        idea["pause_reason"] = "external dependency unavailable"
        idea["resume_condition"] = "dependency is released"
        validate_registry(self.root, self.registry)
        (self.ledger / "registry.json").write_text(json.dumps(self.registry))
        with self.assertRaisesRegex(RegistryError, "before retirement"):
            retirement_plan(self.root, "demo")

        idea.pop("resume_condition")
        with self.assertRaisesRegex(RegistryError, "resume_condition"):
            validate_registry(self.root, self.registry)

    def test_candidate_may_only_own_its_documentation(self):
        idea = self.registry["ideas"][0]
        idea["status"] = "candidate"
        idea["owned_paths"] = ["docs/ideas/demo"]
        validate_registry(self.root, self.registry)
        idea["owned_paths"].append("tools/research/demo")
        with self.assertRaisesRegex(RegistryError, "only own"):
            validate_registry(self.root, self.registry)

    def test_removed_stage0_status_is_rejected(self):
        self.registry["ideas"][0]["status"] = "stage0"
        with self.assertRaisesRegex(RegistryError, "Unknown status"):
            validate_registry(self.root, self.registry)

    def test_learning_dependent_lifecycle_statuses_are_supported(self):
        for status in ("feasibility", "pilot", "prototype", "promoted"):
            with self.subTest(status=status):
                self.registry["ideas"][0]["status"] = status
                validate_registry(self.root, self.registry)

    def test_linked_baseline_is_visible_but_never_disposable(self):
        with tempfile.TemporaryDirectory() as external_name:
            external = Path(external_name)
            (external / "report.json").write_text("{}\n")
            baseline = self.root / "logs/baseline"
            baseline.symlink_to(external, target_is_directory=True)
            self.registry["ideas"][0]["preserved_evidence"] = [
                "logs/baseline/report.json",
            ]
            self.registry["ideas"][0]["discard_artifacts_on_retire"] = []
            validate_registry(self.root, self.registry)

            self.registry["ideas"][0]["discard_artifacts_on_retire"] = [
                "logs/baseline/report.json",
            ]
            validate_registry(self.root, self.registry)
            (self.ledger / "registry.json").write_text(
                json.dumps(self.registry))
            with self.assertRaisesRegex(RegistryError, "escapes logs"):
                retirement_plan(self.root, "demo", prune_artifacts=True)

    def test_local_ledger_paths_are_centralized_and_protected(self):
        self.assertEqual(research_ledger_root(self.root), self.ledger)
        self.assertEqual(registry_path(self.root), self.ledger / "registry.json")
        self.assertEqual(
            experience_log_path(self.root), self.ledger / "EXPERIENCE_LOG.md")

        self.registry["ideas"][0]["discard_artifacts_on_retire"] = [
            "research/ledger/registry.json",
        ]
        with self.assertRaisesRegex(RegistryError, "ledger is protected"):
            validate_registry(self.root, self.registry)

    def test_initialize_ledger_is_idempotent_and_refuses_partial_state(self):
        with tempfile.TemporaryDirectory() as empty_name:
            empty = Path(empty_name)
            first = initialize_ledger(empty)
            self.assertTrue(first["created"])
            self.assertFalse(initialize_ledger(empty)["created"])
            registry = json.loads(
                (empty / "research/ledger/registry.json").read_text())
            self.assertIn("research_ledger", registry["reserved_ids"])

        with tempfile.TemporaryDirectory() as partial_name:
            partial = Path(partial_name)
            (partial / "research/ledger").mkdir(parents=True)
            (partial / "research/ledger/registry.json").write_text("{}")
            with self.assertRaisesRegex(RegistryError, "incomplete"):
                initialize_ledger(partial)

    def test_linked_worktree_resolves_the_main_worktree_ledger(self):
        with tempfile.TemporaryDirectory() as parent_name:
            parent = Path(parent_name)
            main = parent / "project"
            idea = parent / "project-new_idea"
            main.mkdir()
            idea.mkdir()
            (idea / ".git").write_text("gitdir: shared/worktrees/new_idea\n")
            porcelain = (
                f"worktree {main}\n"
                "HEAD 1111111111111111111111111111111111111111\n"
                "branch refs/heads/main\n\n"
                f"worktree {idea}\n"
                "HEAD 2222222222222222222222222222222222222222\n"
                "branch refs/heads/idea/new_idea\n"
            )
            with patch(
                    "tools.research.manage_ideas._optional_git_value",
                    return_value=porcelain):
                self.assertEqual(
                    research_ledger_root(idea),
                    main / "research/ledger")


if __name__ == "__main__":
    unittest.main()
