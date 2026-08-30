#!/usr/bin/env python3
"""Validate and safely retire isolated research ideas.

Destructive operations are deliberately narrow: source ownership must be an
exact ``<allowed-root>/<idea-id>`` path, retirement defaults to a dry run, and
large artifacts require a second explicit flag.  Retired records live in one
ignored local ledger shared by all worktrees, so retirement never dirties the
stable ``main`` worktree.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import json
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "research-idea-registry-v1"
MANIFEST_SCHEMA_VERSION = "research-idea-manifest-v1"
VALID_STATUSES = {
    "candidate", "stage0", "prototype", "promoted", "rejected", "retired",
}
RETIRABLE_STATUS = "rejected"
SOURCE_ROOTS = (
    Path("docs/research"),
    Path("tools/research"),
    Path("test/research"),
    Path("engine/rtv4/obb/incubator"),
    Path("configs/incubator"),
)
IDEA_ID = re.compile(r"^[a-z][a-z0-9_]*$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
LEDGER_ENVIRONMENT = "RTV4_RESEARCH_LEDGER"
LEDGER_RELATIVE = Path("logs/research_ledger")
LEDGER_REGISTRY_NAME = "registry.json"
LEDGER_EXPERIENCE_NAME = "EXPERIENCE_LOG.md"
LEDGER_RESERVED_ID = "research_ledger"


class RegistryError(RuntimeError):
    """Raised when the registry or a requested lifecycle action is unsafe."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _optional_git_value(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=root, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _main_worktree_root(root: Path) -> Path:
    """Return the worktree that currently owns ``main``.

    A non-Git temporary root is treated as its own main worktree.  This keeps
    the lifecycle functions deterministic and easy to test without weakening
    real multi-worktree isolation.
    """

    root = root.resolve()
    if not (root / ".git").exists():
        return root
    output = _optional_git_value(root, "worktree", "list", "--porcelain")
    if output is None:
        raise RegistryError("Cannot discover Git worktrees for research ledger")
    worktree: Path | None = None
    for line in [*output.splitlines(), ""]:
        if line.startswith("worktree "):
            worktree = Path(line.removeprefix("worktree ")).resolve()
        elif line == "branch refs/heads/main" and worktree is not None:
            return worktree
        elif not line:
            worktree = None
    raise RegistryError(
        "No worktree currently owns branch 'main'; either restore the main "
        f"worktree or set {LEDGER_ENVIRONMENT} to its logs/research_ledger path")


def research_ledger_root(
    root: Path,
    override: Path | str | None = None,
) -> Path:
    """Resolve the single protected local ledger shared by all worktrees."""

    root = root.resolve()
    configured = override
    if configured is None:
        configured = os.environ.get(LEDGER_ENVIRONMENT)
    if configured is None:
        candidate = _main_worktree_root(root) / LEDGER_RELATIVE
    else:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = root / candidate
    candidate = candidate.resolve(strict=False)
    if candidate.name != LEDGER_RELATIVE.name or \
            candidate.parent.name != LEDGER_RELATIVE.parent.name:
        raise RegistryError(
            "Research ledger must be an exact logs/research_ledger directory; "
            f"got {candidate}")
    return candidate


def registry_path(root: Path) -> Path:
    return research_ledger_root(root) / LEDGER_REGISTRY_NAME


def experience_log_path(root: Path) -> Path:
    return research_ledger_root(root) / LEDGER_EXPERIENCE_NAME


@contextmanager
def _ledger_lock(root: Path):
    """Serialize read-modify-write operations from concurrent worktrees."""

    ledger = research_ledger_root(root)
    ledger.mkdir(parents=True, exist_ok=True)
    lock_path = ledger / ".manage_ideas.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def manifest_path(root: Path, idea_id: str) -> Path:
    return root / "docs/research" / idea_id / "manifest.json"


def load_registry(root: Path) -> dict:
    path = registry_path(root)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryError(
            f"Cannot read local research registry {path}: {error}. "
            "Run 'python tools/research/manage_ideas.py init-ledger' in the "
            "main worktree before managing ideas.") from error
    if not isinstance(value, dict):
        raise RegistryError("Research registry must be a JSON object")
    return value


def _validate_id(value: object) -> str:
    if not isinstance(value, str) or IDEA_ID.fullmatch(value) is None:
        raise RegistryError(f"Invalid research idea id: {value!r}")
    return value


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _source_path(root: Path, idea_id: str, raw: object) -> Path:
    if not isinstance(raw, str):
        raise RegistryError(f"Owned path for {idea_id} must be a string")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise RegistryError(f"Unsafe owned path for {idea_id}: {raw!r}")
    allowed = {base / idea_id for base in SOURCE_ROOTS}
    if relative not in allowed:
        raise RegistryError(
            f"Owned path must equal an isolated <root>/{idea_id}: {raw!r}")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve(strict=False)
    if not _inside(resolved, resolved_root):
        raise RegistryError(f"Owned path escapes repository: {raw!r}")
    return resolved


def _artifact_path(
    root: Path,
    idea_id: str,
    raw: object,
    *,
    allow_read_only_external: bool = False,
) -> Path:
    if not isinstance(raw, str):
        raise RegistryError(f"Artifact path for {idea_id} must be a string")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RegistryError(f"Unsafe artifact path for {idea_id}: {raw!r}")
    if relative.parts[0] != "logs":
        raise RegistryError(f"Research artifact must be under logs/: {raw!r}")
    if relative == LEDGER_RELATIVE or LEDGER_RELATIVE in relative.parents:
        raise RegistryError(
            f"Research ledger is protected from artifact lifecycle actions: {raw!r}")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve(strict=False)
    if not allow_read_only_external and not _inside(
            resolved, resolved_root / "logs"):
        raise RegistryError(f"Artifact path escapes logs/: {raw!r}")
    # Preserved evidence is never deleted by this tool.  It may therefore be
    # read through a deliberately linked stable-baseline directory in another
    # worktree.  Disposable artifacts retain the strict resolved-path check.
    return (root / relative).absolute() if allow_read_only_external else resolved


def _assert_no_symlink(path: Path, stop: Path) -> None:
    current = path
    while current != stop:
        if current.is_symlink():
            raise RegistryError(f"Refusing symlinked lifecycle path: {current}")
        current = current.parent
    if path.exists() and path.is_dir():
        for child in path.rglob("*"):
            if child.is_symlink():
                raise RegistryError(f"Refusing directory containing symlink: {child}")


def _idea_map(registry: dict) -> dict[str, dict]:
    ideas = registry.get("ideas")
    if not isinstance(ideas, list):
        raise RegistryError("Registry field 'ideas' must be a list")
    result: dict[str, dict] = {}
    for idea in ideas:
        if not isinstance(idea, dict):
            raise RegistryError("Each registry idea must be an object")
        idea_id = _validate_id(idea.get("id"))
        if idea_id in result:
            raise RegistryError(f"Duplicate research idea id: {idea_id}")
        result[idea_id] = idea
    return result


def _manifest_ideas(root: Path) -> dict[str, dict]:
    research_root = root / "docs/research"
    result: dict[str, dict] = {}
    for path in sorted(research_root.glob("*/manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RegistryError(f"Cannot read idea manifest {path}: {error}") from error
        if not isinstance(payload, dict) or \
                payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise RegistryError(
                f"Idea manifest {path} must use {MANIFEST_SCHEMA_VERSION!r}")
        idea = payload.get("idea")
        if not isinstance(idea, dict):
            raise RegistryError(f"Idea manifest {path} must contain an idea object")
        idea_id = _validate_id(idea.get("id"))
        if path.parent.name != idea_id:
            raise RegistryError(
                f"Idea manifest directory {path.parent.name!r} does not match "
                f"idea id {idea_id!r}")
        if idea.get("status") == "retired":
            raise RegistryError(
                f"Retired idea {idea_id} belongs in registry.json, not a live manifest")
        if idea_id in result:
            raise RegistryError(f"Duplicate active idea manifest: {idea_id}")
        result[idea_id] = idea
    return result


def _combined_ideas(root: Path, registry: dict) -> dict[str, dict]:
    archived = _idea_map(registry)
    active = _manifest_ideas(root)
    overlap = set(archived).intersection(active)
    if overlap:
        raise RegistryError(
            "Idea ids cannot exist in both registry.json and a branch-local "
            f"manifest: {sorted(overlap)}")
    return {**archived, **active}


def _git_value(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments], cwd=root, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RegistryError(
            f"Cannot verify Git metadata {' '.join(arguments)!r}: {error}") from error


def _git_is_ancestor(root: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=root, check=False, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
    except OSError as error:
        raise RegistryError(f"Cannot verify Git ancestry: {error}") from error
    if result.returncode not in (0, 1):
        raise RegistryError(
            f"Cannot verify whether {ancestor} is an ancestor of {descendant}: "
            f"{result.stderr.strip()}")
    return result.returncode == 0


def validate_registry(root: Path, registry: dict | None = None) -> dict[str, dict]:
    root = root.resolve()
    discover_manifests = registry is None
    registry = load_registry(root) if discover_manifests else registry
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise RegistryError(
            f"Expected schema_version {SCHEMA_VERSION!r}, got "
            f"{registry.get('schema_version')!r}")
    reserved_values = registry.get("reserved_ids", [])
    if not isinstance(reserved_values, list):
        raise RegistryError("Registry field 'reserved_ids' must be a list")
    reserved = {_validate_id(value) for value in reserved_values}
    required_reserved = {"o2", LEDGER_RESERVED_ID}
    if not required_reserved.issubset(reserved):
        raise RegistryError(
            "Registry must reserve lifecycle ids "
            f"{sorted(required_reserved)}")
    ideas = (
        _combined_ideas(root, registry)
        if discover_manifests else _idea_map(registry)
    )
    overlap = reserved.intersection(ideas)
    if overlap:
        raise RegistryError(f"Reserved ids cannot be research ideas: {sorted(overlap)}")

    try:
        experience_text = experience_log_path(root).read_text(encoding="utf-8")
    except OSError as error:
        raise RegistryError(f"Cannot read experience log: {error}") from error

    manifest_ids = set(_manifest_ideas(root)) if discover_manifests else set()

    for idea_id, idea in ideas.items():
        status = idea.get("status")
        if status not in VALID_STATUSES:
            raise RegistryError(f"Unknown status for {idea_id}: {status!r}")
        if idea_id in manifest_ids:
            if not isinstance(idea.get("base_tag"), str) or not idea["base_tag"]:
                raise RegistryError(f"Active idea {idea_id} lacks base_tag")
            if not isinstance(idea.get("base_commit"), str) or \
                    GIT_COMMIT.fullmatch(idea["base_commit"]) is None:
                raise RegistryError(
                    f"Active idea {idea_id} lacks a full 40-character base_commit")
            if idea.get("branch") != f"idea/{idea_id}":
                raise RegistryError(
                    f"Active idea {idea_id} must declare branch idea/{idea_id}")
            if (root / ".git").exists():
                tag_commit = _git_value(
                    root, "rev-parse", "--verify",
                    f"refs/tags/{idea['base_tag']}^{{}}")
                if tag_commit != idea["base_commit"]:
                    raise RegistryError(
                        f"Active idea {idea_id} base_tag resolves to "
                        f"{tag_commit}, not declared {idea['base_commit']}")
                if not _git_is_ancestor(root, tag_commit):
                    raise RegistryError(
                        f"Active idea {idea_id} does not descend from declared "
                        f"scientific baseline {idea['base_tag']} ({tag_commit})")
                branch = _git_value(root, "branch", "--show-current")
                if branch != idea["branch"]:
                    raise RegistryError(
                        f"Active idea {idea_id} manifest belongs to "
                        f"{idea['branch']}, current branch is {branch or 'detached HEAD'}")
        paths = idea.get("owned_paths", [])
        if not isinstance(paths, list) or len(paths) != len(set(map(str, paths))):
            raise RegistryError(f"owned_paths for {idea_id} must be a unique list")
        resolved_paths = [_source_path(root, idea_id, raw) for raw in paths]
        if status == "retired":
            leftovers = [str(path) for path in resolved_paths if path.exists()]
            if leftovers:
                raise RegistryError(
                    f"Retired idea {idea_id} still owns live paths: {leftovers}")
        else:
            missing = [str(path) for path in resolved_paths if not path.exists()]
            if missing:
                raise RegistryError(
                    f"Active idea {idea_id} declares missing paths: {missing}")
        for key in ("preserved_evidence", "discard_artifacts_on_retire"):
            values = idea.get(key, [])
            if not isinstance(values, list) or len(values) != len(set(map(str, values))):
                raise RegistryError(f"{key} for {idea_id} must be a unique list")
            for raw in values:
                _artifact_path(
                    root, idea_id, raw,
                    # Validation may inspect another worktree's explicitly
                    # linked evidence.  Every deletion plan resolves the same
                    # path again with the strict default before mutating it.
                    allow_read_only_external=True)
        if status in {"rejected", "retired"}:
            entry = idea.get("experience_entry")
            if not isinstance(entry, str) or f"## {entry}：" not in experience_text:
                raise RegistryError(
                    f"Rejected/retired idea {idea_id} lacks a durable experience card")
            if not idea.get("verdict"):
                raise RegistryError(f"Rejected/retired idea {idea_id} lacks a verdict")
    return ideas


def _walk_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    return sorted(child for child in path.rglob("*") if child.is_file())


def retirement_plan(
    root: Path,
    idea_id: str,
    *,
    prune_artifacts: bool = False,
) -> dict:
    root = root.resolve()
    idea_id = _validate_id(idea_id)
    registry = load_registry(root)
    ideas = validate_registry(root)
    reserved = set(registry.get("reserved_ids", []))
    if idea_id in reserved:
        raise RegistryError(f"Reserved research id cannot be retired: {idea_id}")
    if idea_id not in ideas:
        raise RegistryError(f"Unknown research idea: {idea_id}")
    idea = ideas[idea_id]
    if idea.get("status") != RETIRABLE_STATUS:
        raise RegistryError(
            f"Idea {idea_id} must be {RETIRABLE_STATUS!r} before retirement; "
            f"got {idea.get('status')!r}")

    source_paths = [_source_path(root, idea_id, raw)
                    for raw in idea.get("owned_paths", [])]
    artifact_paths = [_artifact_path(root, idea_id, raw)
                      for raw in idea.get("discard_artifacts_on_retire", [])] \
        if prune_artifacts else []
    for path in [*source_paths, *artifact_paths]:
        _assert_no_symlink(path, root)
    return {
        "idea_id": idea_id,
        "status_before": RETIRABLE_STATUS,
        "status_after": "retired",
        "ledger_root": str(research_ledger_root(root)),
        "ledger_registry": str(registry_path(root)),
        "source_paths": [str(path.relative_to(root)) for path in source_paths],
        "source_files": [str(path.relative_to(root))
                         for parent in source_paths for path in _walk_files(parent)],
        "artifact_paths": [str(path.relative_to(root)) for path in artifact_paths],
        "prune_artifacts": bool(prune_artifacts),
    }


def _atomic_write_text(
    path: Path,
    text: str,
    *,
    keep_backup: bool = True,
) -> Path | None:
    """Atomically replace a ledger file and retain its previous generation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_name(f"{path.name}.bak") if path.exists() and keep_backup \
        else None
    if backup is not None:
        shutil.copy2(path, backup)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return backup


def _atomic_write_json(
    path: Path,
    value: dict,
    *,
    keep_backup: bool = True,
) -> Path | None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    return _atomic_write_text(path, text, keep_backup=keep_backup)


def initialize_ledger(root: Path) -> dict:
    """Create an empty ignored ledger without overwriting partial state."""

    ledger = research_ledger_root(root)
    registry = ledger / LEDGER_REGISTRY_NAME
    experience = ledger / LEDGER_EXPERIENCE_NAME
    with _ledger_lock(root):
        existing = [path for path in (registry, experience) if path.exists()]
        if existing and len(existing) != 2:
            raise RegistryError(
                "Research ledger is incomplete; refusing to guess which local "
                f"state is authoritative: {[str(path) for path in existing]}")
        created = not existing
        if created:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "reserved_ids": ["o2", LEDGER_RESERVED_ID],
                "ideas": [],
            }
            _atomic_write_json(registry, payload, keep_backup=False)
            _atomic_write_text(
                experience,
                "# 本地研究经验账本\n\n"
                "> 本目录被 Git 忽略，由 manage_ideas.py 原子维护；请单独备份。\n",
                keep_backup=False,
            )
        validate_registry(root)
    return {
        "ledger_root": str(ledger),
        "registry": str(registry),
        "experience_log": str(experience),
        "created": created,
    }


def record_experience(root: Path, idea_id: str) -> dict:
    """Append one branch-local experience card to the shared local ledger."""

    root = root.resolve()
    idea_id = _validate_id(idea_id)
    ideas = _manifest_ideas(root)
    if idea_id not in ideas:
        raise RegistryError(f"No live manifest found for idea {idea_id}")
    idea = ideas[idea_id]
    entry = idea.get("experience_entry")
    if not isinstance(entry, str) or not entry:
        raise RegistryError(
            f"Idea {idea_id} must declare experience_entry before recording")
    source = manifest_path(root, idea_id).parent / "experience.md"
    try:
        card = source.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RegistryError(f"Cannot read experience card: {error}") from error
    heading = f"## {entry}："
    if heading not in card:
        raise RegistryError(
            f"Experience card {source} must contain heading {heading!r}")
    with _ledger_lock(root):
        try:
            current = experience_log_path(root).read_text(encoding="utf-8")
        except OSError as error:
            raise RegistryError(f"Cannot read experience log: {error}") from error
        if heading in current:
            return {
                "idea_id": idea_id,
                "experience_entry": entry,
                "recorded": False,
                "experience_log": str(experience_log_path(root)),
            }
        updated = current.rstrip() + "\n\n" + card + "\n"
        backup = _atomic_write_text(experience_log_path(root), updated)
    return {
        "idea_id": idea_id,
        "experience_entry": entry,
        "recorded": True,
        "experience_log": str(experience_log_path(root)),
        "backup": str(backup) if backup is not None else None,
    }


def apply_retirement(
    root: Path,
    idea_id: str,
    *,
    prune_artifacts: bool = False,
    retired_on: str | None = None,
) -> dict:
    root = root.resolve()
    with _ledger_lock(root):
        # Re-read and re-plan under the shared lock so two worktrees cannot
        # overwrite one another's newly archived record.
        plan = retirement_plan(root, idea_id, prune_artifacts=prune_artifacts)
        registry = load_registry(root)
        ideas = validate_registry(root)
        idea = copy.deepcopy(ideas[idea_id])
        for raw in plan["source_paths"]:
            path = root / raw
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        for raw in plan["artifact_paths"]:
            path = root / raw
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        idea["status"] = "retired"
        idea["retired_on"] = retired_on or dt.date.today().isoformat()
        archived = _idea_map(registry)
        if idea_id in archived:
            archived[idea_id].update(idea)
        else:
            registry["ideas"].append(idea)
        validate_registry(root, registry)
        backup = _atomic_write_json(registry_path(root), registry)
        plan["registry_backup"] = str(backup) if backup is not None else None
    return plan


def artifact_prune_plan(root: Path, idea_id: str) -> dict:
    """Plan removal of explicitly disposable artifacts for a retired idea."""

    root = root.resolve()
    idea_id = _validate_id(idea_id)
    ideas = validate_registry(root)
    if idea_id not in ideas:
        raise RegistryError(f"Unknown research idea: {idea_id}")
    idea = ideas[idea_id]
    if idea.get("status") != "retired":
        raise RegistryError(
            f"Artifact-only pruning requires a retired idea; "
            f"{idea_id} is {idea.get('status')!r}")
    paths = [_artifact_path(root, idea_id, raw)
             for raw in idea.get("discard_artifacts_on_retire", [])]
    for path in paths:
        _assert_no_symlink(path, root)
    return {
        "idea_id": idea_id,
        "status": "retired",
        "artifact_paths": [str(path.relative_to(root)) for path in paths],
        "existing_artifact_paths": [
            str(path.relative_to(root)) for path in paths if path.exists()],
    }


def apply_artifact_prune(root: Path, idea_id: str) -> dict:
    root = root.resolve()
    plan = artifact_prune_plan(root, idea_id)
    for raw in plan["existing_artifact_paths"]:
        path = root / raw
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    validate_registry(root)
    return plan


def _print_rows(ideas: Iterable[dict]) -> None:
    for idea in ideas:
        print(f"{idea['id']:<32} {idea['status']:<10} {idea.get('verdict', '-')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger-root", type=Path,
        help=("override the protected local logs/research_ledger directory; "
              f"equivalent to {LEDGER_ENVIRONMENT}"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "init-ledger", help="initialize or validate the ignored local ledger")
    subparsers.add_parser(
        "ledger-path", help="print the local ledger paths used by this worktree")
    subparsers.add_parser("list", help="list registered research ideas")
    subparsers.add_parser("validate", help="validate registry and live ownership")
    record = subparsers.add_parser(
        "record-experience",
        help="atomically append docs/research/<idea>/experience.md to the ledger",
    )
    record.add_argument("idea_id")
    retire = subparsers.add_parser("retire", help="plan or apply safe idea retirement")
    retire.add_argument("idea_id")
    retire.add_argument("--apply", action="store_true",
                        help="apply the plan; omission is always a dry run")
    retire.add_argument("--prune-artifacts", action="store_true",
                        help="also delete exact large artifacts declared by the idea")
    prune = subparsers.add_parser(
        "prune-artifacts", help="plan or apply artifact-only cleanup after retirement")
    prune.add_argument("idea_id")
    prune.add_argument("--apply", action="store_true",
                       help="apply the plan; omission is always a dry run")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = repository_root()
    if args.ledger_root is not None:
        os.environ[LEDGER_ENVIRONMENT] = str(args.ledger_root)
    try:
        if args.command == "init-ledger":
            print(json.dumps(initialize_ledger(root), ensure_ascii=False, indent=2))
        elif args.command == "ledger-path":
            print(json.dumps({
                "ledger_root": str(research_ledger_root(root)),
                "registry": str(registry_path(root)),
                "experience_log": str(experience_log_path(root)),
            }, ensure_ascii=False, indent=2))
        elif args.command == "list":
            ideas = validate_registry(root)
            _print_rows(ideas.values())
        elif args.command == "validate":
            ideas = validate_registry(root)
            print(json.dumps({"status": "ok", "idea_count": len(ideas)}))
        elif args.command == "record-experience":
            print(json.dumps(
                record_experience(root, args.idea_id),
                ensure_ascii=False, indent=2))
        elif args.command == "retire":
            if args.apply:
                result = apply_retirement(
                    root, args.idea_id, prune_artifacts=args.prune_artifacts)
                result["applied"] = True
            else:
                result = retirement_plan(
                    root, args.idea_id, prune_artifacts=args.prune_artifacts)
                result["applied"] = False
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "prune-artifacts":
            if args.apply:
                result = apply_artifact_prune(root, args.idea_id)
                result["applied"] = True
            else:
                result = artifact_prune_plan(root, args.idea_id)
                result["applied"] = False
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except RegistryError as error:
        raise SystemExit(f"research lifecycle error: {error}") from error


if __name__ == "__main__":
    main()
