#!/usr/bin/env python3
"""Validate and safely retire isolated research ideas.

Destructive operations are deliberately narrow: source ownership must be an
exact ``<allowed-root>/<idea-id>`` path, retirement defaults to a dry run, and
large artifacts require a second explicit flag.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import re
import shutil
import tempfile
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


class RegistryError(RuntimeError):
    """Raised when the registry or a requested lifecycle action is unsafe."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def registry_path(root: Path) -> Path:
    return root / "docs/research/registry.json"


def experience_log_path(root: Path) -> Path:
    return root / "docs/research/EXPERIENCE_LOG.md"


def manifest_path(root: Path, idea_id: str) -> Path:
    return root / "docs/research" / idea_id / "manifest.json"


def load_registry(root: Path) -> dict:
    path = registry_path(root)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryError(f"Cannot read registry {path}: {error}") from error
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
                    allow_read_only_external=(key == "preserved_evidence"))
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
        "source_paths": [str(path.relative_to(root)) for path in source_paths],
        "source_files": [str(path.relative_to(root))
                         for parent in source_paths for path in _walk_files(parent)],
        "artifact_paths": [str(path.relative_to(root)) for path in artifact_paths],
        "prune_artifacts": bool(prune_artifacts),
    }


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def apply_retirement(
    root: Path,
    idea_id: str,
    *,
    prune_artifacts: bool = False,
    retired_on: str | None = None,
) -> dict:
    root = root.resolve()
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
    _atomic_write_json(registry_path(root), registry)
    validate_registry(root, registry)
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
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list registered research ideas")
    subparsers.add_parser("validate", help="validate registry and live ownership")
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
    try:
        if args.command == "list":
            ideas = validate_registry(root)
            _print_rows(ideas.values())
        elif args.command == "validate":
            ideas = validate_registry(root)
            print(json.dumps({"status": "ok", "idea_count": len(ideas)}))
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
