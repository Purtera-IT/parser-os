"""Optional static ONNX classifier wrapper (PR8).

Used by the legend locator as the fourth detection layer
(``classifier`` callback). The classifier is a frozen, CPU-only
ONNX model shipped with the repo (path + SHA-256 declared in
``config``). At load time we hash the model bytes and refuse to use
a mismatched file — that pin keeps inference deterministic across
machines and guarantees the same model is used by every reviewer.

The wrapper exposes two surfaces:

- ``LegendBlockClassifier.score(crop_ndarray)`` — return a delta in
  ``[-1, 1]`` for "this crop looks like a legend block." The locator
  will clamp the delta into a small bounded boost before adding it
  to the rule-based score.
- ``LegendBlockClassifier.from_config(...)`` — build the classifier
  from a config dict (path, expected_sha256, input_shape). Returns
  ``None`` (with a structured reason on ``last_error``) when the
  model file is missing or its hash does not match.

No runtime LLM. No network calls. No training. The model is
treated as a static asset; if it ever needs an upgrade, the
expected SHA changes and CI fails until the file is rotated.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LegendBlockClassifier:
    """A loaded ONNX classifier for legend-block ranking.

    Construct via ``from_config`` — direct construction is allowed
    but skips the hash check, so should only be used in tests.
    """

    session: Any
    input_name: str
    input_shape: tuple[int, ...]
    sha256: str

    def score(self, crop: Any) -> float:
        """Return a bounded "is legend block" delta in ``[-1, 1]``.

        ``crop`` must be a ``np.ndarray`` matching ``input_shape``
        (the wrapper does not resize for you; the locator is
        expected to pre-resize before scoring so the determinism
        contract is visible at the call site). Returns 0.0 on any
        runtime error so the locator falls back to its rule-based
        score.
        """
        try:
            import numpy as np  # type: ignore[import-not-found]
        except Exception:  # pragma: no cover
            return 0.0
        try:
            tensor = np.asarray(crop, dtype=np.float32)
            if tensor.shape != tuple(self.input_shape):
                return 0.0
            outputs = self.session.run(None, {self.input_name: tensor[None, ...]})
        except Exception:  # pragma: no cover
            return 0.0
        if not outputs:
            return 0.0
        try:
            value = float(outputs[0].reshape(-1)[0])
        except Exception:  # pragma: no cover
            return 0.0
        # Network output is the raw logit; clip to [-1, 1] so the
        # locator never gets a runaway boost from a noisy model.
        return max(-1.0, min(1.0, value))


@dataclass(frozen=True)
class ClassifierConfig:
    model_path: Path
    expected_sha256: str
    input_shape: tuple[int, ...]
    input_name: str = "input"


def hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def from_config(cfg: ClassifierConfig) -> tuple[LegendBlockClassifier | None, str]:
    """Build the classifier or return ``(None, reason)``.

    ``reason`` is a human-readable explanation; the parser layer
    can turn it into a structured warning. The function never
    raises — failure modes are explicit return values.
    """
    if not cfg.model_path.is_file():
        return None, f"classifier model not found at {cfg.model_path}"
    actual = hash_file(cfg.model_path)
    if actual != cfg.expected_sha256:
        return (
            None,
            (
                f"classifier model hash mismatch at {cfg.model_path}: "
                f"expected {cfg.expected_sha256}, got {actual}"
            ),
        )
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except Exception:
        return None, "onnxruntime not installed; classifier disabled"
    try:
        session = ort.InferenceSession(str(cfg.model_path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # pragma: no cover
        return None, f"onnxruntime failed to load model: {exc}"
    return (
        LegendBlockClassifier(
            session=session,
            input_name=cfg.input_name,
            input_shape=tuple(cfg.input_shape),
            sha256=actual or "",
        ),
        "ok",
    )
