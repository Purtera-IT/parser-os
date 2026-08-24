"""Registry retention: bounded growth that never breaks rollback.

The registry was append-only — one version per relation per nightly run, never
removed. It reached 14,165 files / ~5GB, most of the ~9GB ml-artifacts container,
which the workers fetch onto 4-8Gi of ephemeral disk. That filled /tmp and every
compile died with OSError: [Errno 28]. Unbounded growth in a store nothing reads
is a slow outage.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from app.learning.head_registry import HeadRegistry


class _Head:
    """Minimal NeuralHead stand-in: the registry only calls save()/classes_."""

    classes_ = ("a", "b")
    trained = True

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, x=np.zeros(4, dtype=np.float32))


def _reg(tmp_path) -> HeadRegistry:
    return HeadRegistry(str(tmp_path / "reg"))


def _register(reg: HeadRegistry, relation: str, n: int) -> list[str]:
    import time as _t

    out = []
    for _ in range(n):
        out.append(reg.register(relation, _Head()).version)
        _t.sleep(0.01)  # version ids embed HH:MM:SS — keep them ordered
    return out


def test_keeps_only_the_most_recent_versions(tmp_path):
    reg = _reg(tmp_path)
    versions = _register(reg, "atom_type", 12)
    removed = reg.prune(keep_per_relation=5)
    kept = {m.version for m in reg.history("atom_type")}
    assert kept == set(versions[-5:])
    assert len(removed) == 7


def test_never_deletes_the_serving_champion(tmp_path):
    """Deleting the champion would take the head out of service to save disk."""
    reg = _reg(tmp_path)
    versions = _register(reg, "atom_type", 10)
    reg.promote("atom_type", versions[0])  # oldest is champion
    reg.prune(keep_per_relation=2)
    kept = {m.version for m in reg.history("atom_type")}
    assert versions[0] in kept
    assert reg.champion_version("atom_type") == versions[0]


def test_never_breaks_rollback(tmp_path):
    """previous_champion must survive: a retention policy that silently makes
    rollback impossible is worse than the disk problem it solves."""
    reg = _reg(tmp_path)
    versions = _register(reg, "atom_type", 10)
    reg.promote("atom_type", versions[0])
    reg.promote("atom_type", versions[9])  # versions[0] becomes previous
    reg.prune(keep_per_relation=1)
    assert reg.rollback("atom_type") == versions[0]
    assert reg.champion_version("atom_type") == versions[0]


def test_deletes_the_files_not_just_the_index(tmp_path):
    reg = _reg(tmp_path)
    versions = _register(reg, "atom_type", 6)
    reg.prune(keep_per_relation=2)
    gone = versions[0]
    rel_dir = os.path.join(str(tmp_path / "reg"), "atom_type")
    remaining = os.listdir(rel_dir) if os.path.isdir(rel_dir) else []
    assert not any(gone in f for f in remaining), "disk must actually shrink"


def test_prunes_every_relation_independently(tmp_path):
    reg = _reg(tmp_path)
    _register(reg, "atom_type", 8)
    _register(reg, "service_routing", 8)
    reg.prune(keep_per_relation=3)
    assert len(reg.history("atom_type")) == 3
    assert len(reg.history("service_routing")) == 3


def test_is_idempotent(tmp_path):
    reg = _reg(tmp_path)
    _register(reg, "atom_type", 9)
    first = reg.prune(keep_per_relation=4)
    second = reg.prune(keep_per_relation=4)
    assert first and second == [], "a second prune has nothing left to do"


def test_nothing_to_prune_is_a_no_op(tmp_path):
    reg = _reg(tmp_path)
    _register(reg, "atom_type", 3)
    assert reg.prune(keep_per_relation=5) == []
    assert len(reg.history("atom_type")) == 3


@pytest.mark.parametrize("keep", [0, -1])
def test_zero_keeps_only_the_protected_set(tmp_path, keep):
    reg = _reg(tmp_path)
    versions = _register(reg, "atom_type", 5)
    reg.promote("atom_type", versions[2])
    reg.prune(keep_per_relation=keep)
    kept = {m.version for m in reg.history("atom_type")}
    assert kept == {versions[2]}, "champion survives even at keep=0"
