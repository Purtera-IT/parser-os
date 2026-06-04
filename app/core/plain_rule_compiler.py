"""Plain-English rule compiler (upgrade #7).

A PM should not have to learn the shape of a :class:`Correction` to teach the
parser. They type one sentence — *"PurTera is our own company, it is never a
site"*, *"a price-book line is not deal scope"*, *"don't drop the Lookup tab"* —
and this module turns that sentence into a **verify-gated** learned rule.

The compiler does NOT trust the sentence. It runs the synthesized rule through
the exact same nine-invariant gate (:func:`correction_eval.gated_confirm`) that
guards every other correction, so an English rule can only commit if it:

* fixes the thing it claims to fix (B),
* generalizes to paraphrases the LLM proposed (C),
* disturbs none of the control cases the LLM proposed (D, *the hard one*),
* and clears the reversibility / latency / offline-safety / scope / provenance
  invariants (E–I).

Anything less and the rule is refused — the PM's sentence becomes a no-op with a
readable report, never a silent guess. This is the same precedence the whole
system runs on: STORE → LLM → UNDECIDED. The LLM here only *proposes*; the gate
*decides*.

The LLM synthesizer is **injected** (``synthesize=...``), so the whole pipeline
is hermetically testable with a fake that returns a fixed proposal — no network,
no model. The default synthesizer talks to the dev Ollama big model only when no
fake is supplied.

Universal by construction: the proposal is grounded in *relation + meaning +
scope*, never in a customer-specific keyword list. The store generalizes by
role/shape via embedding kNN, so a rule taught on one deal stays warm for new
deals with new names.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.complaint_intake import (
    Complaint,
    ComplaintResolution,
    intake,
)
from app.core.correction_eval import EvalReport, Probe, gated_confirm
from app.core.decide import DecisionScope
from app.core.feedback_store import (
    SCOPE_DEAL,
    SCOPE_GLOBAL,
    SCOPE_PACK,
    FeedbackStore,
)

# A synthesizer maps one English sentence to a raw proposal dict (see
# RuleProposal.from_raw for the accepted shape). Injected for testability.
Synthesizer = Callable[[str], dict]

_VALID_SCOPES = {SCOPE_GLOBAL, SCOPE_PACK, SCOPE_DEAL}


@dataclass
class ControlCase:
    """A case the rule must leave UNTOUCHED (a collateral / control probe)."""

    text: str
    candidates: list[str] = field(default_factory=list)


@dataclass
class RuleProposal:
    """A structured, validated proposal synthesized from a PM's sentence.

    Attributes:
        relation: the decide() relation the rule governs (e.g. ``physical_site``,
            ``atom_type``, ``entity_keep:org``).
        verdict: the verdict matching text should resolve to. MUST appear in
            ``candidates``.
        candidates: the closed verdict set for this relation.
        exemplar: the canonical positive example of the rule (the embedding
            anchor and the invariant-B fix probe).
        paraphrases: semantically-equivalent restatements; invariant-C
            generalization probes (same expected verdict as ``exemplar``).
        controls: cases that must NOT change; invariant-D collateral probes.
        scope/scope_key: where the rule applies.
    """

    relation: str
    verdict: str
    candidates: list[str]
    exemplar: str
    paraphrases: list[str] = field(default_factory=list)
    controls: list[ControlCase] = field(default_factory=list)
    scope: str = SCOPE_GLOBAL
    scope_key: str = ""

    @classmethod
    def from_raw(cls, raw: Any) -> "RuleProposal":
        """Coerce a loosely-typed synthesizer dict into a validated proposal.

        Raises ``ValueError`` on a proposal that could never gate-pass (missing
        relation/verdict/exemplar, or a verdict outside its candidate set), so a
        malformed LLM response fails loudly here rather than silently producing a
        rule that can never fire.
        """
        if not isinstance(raw, dict):
            raise ValueError("synthesizer must return a JSON object")

        relation = str(raw.get("relation") or "").strip()
        verdict = str(raw.get("verdict") or "").strip()
        exemplar = str(raw.get("exemplar") or "").strip()

        candidates = [
            str(c).strip()
            for c in (raw.get("candidates") or [])
            if str(c).strip()
        ]
        # A verdict the candidate set doesn't list is unenforceable — but a
        # well-formed verdict with an empty candidate list is recoverable: the
        # verdict itself is at least one valid candidate.
        if verdict and not candidates:
            candidates = [verdict]

        if not relation:
            raise ValueError("proposal missing 'relation'")
        if not verdict:
            raise ValueError("proposal missing 'verdict'")
        if not exemplar:
            raise ValueError("proposal missing 'exemplar' (nothing to embed)")
        if verdict not in candidates:
            raise ValueError(
                f"verdict {verdict!r} not in candidates {candidates!r}"
            )

        paraphrases = [
            str(p).strip()
            for p in (raw.get("paraphrases") or [])
            if str(p).strip() and str(p).strip() != exemplar
        ]

        controls: list[ControlCase] = []
        for c in raw.get("controls") or []:
            if isinstance(c, str):
                ct = c.strip()
                if ct:
                    controls.append(ControlCase(text=ct, candidates=list(candidates)))
            elif isinstance(c, dict):
                ct = str(c.get("text") or "").strip()
                if not ct:
                    continue
                ccand = [
                    str(x).strip()
                    for x in (c.get("candidates") or [])
                    if str(x).strip()
                ] or list(candidates)
                controls.append(ControlCase(text=ct, candidates=ccand))

        scope = str(raw.get("scope") or SCOPE_GLOBAL).strip().lower()
        if scope not in _VALID_SCOPES:
            scope = SCOPE_GLOBAL
        scope_key = str(raw.get("scope_key") or "").strip()

        return cls(
            relation=relation,
            verdict=verdict,
            candidates=candidates,
            exemplar=exemplar,
            paraphrases=paraphrases,
            controls=controls,
            scope=scope,
            scope_key=scope_key,
        )

    def _decision_scope(self) -> DecisionScope:
        if self.scope == SCOPE_DEAL:
            return DecisionScope(deal_id=self.scope_key)
        if self.scope == SCOPE_PACK:
            return DecisionScope(pack_id=self.scope_key)
        return DecisionScope()

    def fix_probes(self) -> list[Probe]:
        sc = self._decision_scope()
        return [
            Probe(
                text=self.exemplar,
                relation=self.relation,
                expect=self.verdict,
                candidates=list(self.candidates),
                scope=sc,
            )
        ]

    def generalization_probes(self) -> list[Probe]:
        sc = self._decision_scope()
        return [
            Probe(
                text=p,
                relation=self.relation,
                expect=self.verdict,
                candidates=list(self.candidates),
                scope=sc,
            )
            for p in self.paraphrases
        ]

    def collateral_probes(self) -> list[Probe]:
        sc = self._decision_scope()
        return [
            Probe(
                text=c.text,
                relation=self.relation,
                expect=None,  # controls: what matters is baseline == trial
                candidates=list(c.candidates or self.candidates),
                scope=sc,
            )
            for c in self.controls
        ]


@dataclass
class CompiledRule:
    """The outcome of compiling one English sentence into a gated rule."""

    sentence: str
    proposal: RuleProposal
    resolution: ComplaintResolution
    committed: bool
    report: EvalReport

    @property
    def correction_id(self) -> str:
        return self.proposal_correction_id

    @property
    def proposal_correction_id(self) -> str:
        return getattr(self.resolution.proposed_correction, "id", "")

    def explain(self) -> str:
        verdict = "COMMITTED" if self.committed else "REFUSED"
        return (
            f"[{verdict}] rule on '{self.proposal.relation}' "
            f"({self.proposal.exemplar!r} → '{self.proposal.verdict}') "
            f"{self.report.summary()}"
        )


def compile_rule(
    sentence: str,
    store: FeedbackStore,
    *,
    synthesize: Synthesizer | None = None,
    created_by: str = "",
    default_scope: str = SCOPE_GLOBAL,
    default_scope_key: str = "",
) -> CompiledRule:
    """Compile one PM sentence into a verify-gated learned rule.

    Steps: synthesize a structured proposal (injected LLM, or the default), turn
    it into a :class:`Complaint`/:class:`ComplaintResolution`, then run it
    through :func:`gated_confirm`. The rule commits to ``store`` only on a clean
    nine-invariant pass; otherwise nothing is written.

    Args:
        sentence: the PM's plain-English rule.
        store: the live :class:`FeedbackStore` (mutated only on a clean pass).
        synthesize: injected sentence→proposal function. Defaults to the Ollama
            big-model synthesizer when omitted.
        created_by: audit attribution for the resulting correction.
        default_scope/default_scope_key: fallback scope when the proposal omits
            one (the synthesizer may override).

    Returns:
        A :class:`CompiledRule` with ``committed`` and the full
        :class:`EvalReport`. Never raises on a refused rule — only on an
        unusable (malformed) proposal, which is a programming/transport error,
        not a learning outcome.
    """
    if not sentence or not sentence.strip():
        raise ValueError("empty rule sentence")

    synth = synthesize or _default_synthesizer
    raw = synth(sentence)
    proposal = RuleProposal.from_raw(raw)

    # The proposal may inherit the caller's default scope if it didn't pick one.
    if proposal.scope == SCOPE_GLOBAL and default_scope != SCOPE_GLOBAL:
        proposal.scope = default_scope
        proposal.scope_key = proposal.scope_key or default_scope_key

    complaint = Complaint(
        relation=proposal.relation,
        desired_verdict=proposal.verdict,
        text=proposal.exemplar,
        scope=proposal.scope,
        scope_key=proposal.scope_key,
        note=sentence.strip(),
        created_by=created_by,
    )
    resolution = intake(complaint, result=None, store=store)

    committed, report = gated_confirm(
        store,
        resolution,
        fix_probes=proposal.fix_probes(),
        generalization_probes=proposal.generalization_probes(),
        collateral_probes=proposal.collateral_probes(),
    )
    return CompiledRule(
        sentence=sentence.strip(),
        proposal=proposal,
        resolution=resolution,
        committed=committed,
        report=report,
    )


# ──────────────────────── default LLM synthesizer ────────────────────────

_DEFAULT_HOST = "http://100.114.102.122:11434"
_DEFAULT_MODEL = "qwen3:14b"
_DEFAULT_TIMEOUT = 120

_SYNTH_PROMPT = """You convert a project manager's plain-English correction \
about a bid-document parser into a STRUCTURED rule for a learned correction store.

The store decides questions of the form: for a given RELATION, does this text \
resolve to a VERDICT? Verdicts are a small closed set (the candidates).

Common relations and their candidates:
- "physical_site": ["keep","drop"]   (is this text a real physical site/location?)
- "atom_type": the type taxonomy plus "_keep"   (what kind of atom is this?)
- "entity_keep:<type>": ["keep","drop"]   (keep this extracted entity, or drop as noise?)

Given the PM sentence, output ONLY a JSON object (no prose) with keys:
  relation     : string, the decide() relation this rule governs
  verdict      : string, the verdict matching text should resolve to (in candidates)
  candidates   : array of strings, the closed verdict set
  exemplar     : string, the clearest single example the rule is about
  paraphrases  : array of strings, 2-4 different phrasings that should resolve
                 the SAME way (to prove the rule generalizes by meaning)
  controls     : array of strings, 2-4 cases that are DIFFERENT and must NOT be
                 affected by this rule (to prove no collateral damage)
  scope        : one of "global","pack","deal" (default "global")
  scope_key    : string, the pack/deal id when scope is not global (else "")

PM sentence:
{sentence}

JSON:"""


def _default_synthesizer(sentence: str) -> dict:
    """Synthesize a proposal via the dev Ollama big model. Returns ``{}`` on any
    transport/parse failure so the caller's ``from_raw`` raises a clear error
    rather than this function guessing."""
    text = _call_ollama(_SYNTH_PROMPT.format(sentence=sentence))
    return _extract_json_object(text)


def _extract_json_object(text: str) -> dict:
    if not text:
        return {}
    # Fast path: the whole thing is JSON.
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    # Otherwise grab the first balanced {...} block.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _call_ollama(prompt: str, *, max_tokens: int = 1024) -> str:
    import http.client
    import urllib.request

    host = os.environ.get("OLLAMA_HOST", _DEFAULT_HOST).rstrip("/")
    model = os.environ.get("SOWSMITH_RULE_MODEL") or os.environ.get(
        "OLLAMA_BIG_MODEL", _DEFAULT_MODEL
    )
    timeout = int(os.environ.get("SOWSMITH_LLM_TIMEOUT", str(_DEFAULT_TIMEOUT)))

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    body = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except http.client.IncompleteRead as exc:
        try:
            body = (exc.partial or b"").decode("utf-8", errors="ignore")
        except Exception:
            body = ""
    except Exception:
        return ""
    if not body:
        return ""
    try:
        result = json.loads(body)
        return str(result.get("response") or "")
    except json.JSONDecodeError:
        m = re.search(r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"', body)
        if m:
            try:
                return json.loads('"' + m.group(1) + '"')
            except json.JSONDecodeError:
                return m.group(1)
        return ""


__all__ = [
    "Synthesizer",
    "ControlCase",
    "RuleProposal",
    "CompiledRule",
    "compile_rule",
]
