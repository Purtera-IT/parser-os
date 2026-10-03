"""Planted-rule twins: teach and test "follow the rule the text states".

Real labels cannot tell the model whether it followed a rule's *words* or
just memorized which lines get rejected. Twins can. Each episode builds one
set of synthetic lines and two rule texts that differ in one detail (a
threshold, a role). The labels follow each twin's text. A model that reads
the rule gets both twins right; a model that ignores the text can get at
most one.

Used two ways:
* **training** (``c3_loss(model, batch, bank=ExplanationBank(), rules=rules)``
  on both twins): grounds the reason compiler in what rule text says, with
  exact labels and no customer data;
* **evaluation** (``rule_following``): the headline test for v5, the share of
  lines a held-out rule text gets right with no examples of it.

Everything here is generated. Nothing comes from a deal.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .data import DealExample, featurize
from .explain import Explanation, ExplanationBank
from .schema import Schema

ROOMS = ("lobby", "boardroom", "break room", "training room", "waiting area", "cafeteria")
DOCS = ("drawings", "price sheet", "site survey", "floor plan", "parts list")
ROLES = ("seller", "partner", "customer", "project_manager")
COMPANY = "planted_co"


@dataclass
class Twin:
    deal: DealExample
    rules: ExplanationBank
    truth: list[str]            # co_action per line


def _deal(lines: list[tuple[str, str]], labels: list[str], reasons: list[str], tag: str) -> DealExample:
    atoms = []
    for i, ((text, role), act, why) in enumerate(zip(lines, labels, reasons)):
        reads = {"co_action": act}
        if act != "keep":
            reads["co_reason"] = why
        atoms.append({"key": f"{tag}-{i}", "text": text, "entered_at": 1_700_000_000 + 60 * i,
                      "doc_id": f"{tag}-mail-{i // 4}", "doc_kind": "email", "section": "body",
                      "order": i, "speaker_role": role,
                      "speaker_side": "customer" if role == "customer" else "provider",
                      "label": {"label_type": "requirement", "reads_set": reads}})
    return DealExample.from_dict({"deal_id": tag, "company": COMPANY, "atoms": atoms})


def threshold_twins(seed: int = 0, n: int = 16, t_a: int = 55, t_b: int = 75) -> tuple[Twin, Twin]:
    """'Displays of T inches or smaller are ignored', with T = t_a vs t_b."""
    rng = random.Random(seed)
    sizes = [rng.choice([43, 50, 55, 65, 70, 75, 85, 98]) for _ in range(n)]
    lines = [(f"Mount {rng.randint(1, 6)} {s}\" displays in the {rng.choice(ROOMS)}.", "customer")
             for s in sizes]

    def twin(t: int, tag: str) -> Twin:
        text = (f"Displays of {t} inches or smaller are ignored, because this company only "
                f"installs large-format displays above {t} inches.")
        truth = ["ignore" if s <= t else "keep" for s in sizes]
        rule = Explanation(text, "company", company=COMPANY, ref=f"{tag}-rule")
        return Twin(_deal(lines, truth, ["out_of_service_line"] * n, tag), ExplanationBank([rule]), truth)

    return twin(t_a, f"thr{seed}a"), twin(t_b, f"thr{seed}b")


def role_twins(seed: int = 0, n: int = 16, role_a: str = "seller",
               role_b: str = "partner") -> tuple[Twin, Twin]:
    """'When a <role> promises to send a document, reject it', role_a vs role_b."""
    rng = random.Random(seed)
    roles = [rng.choice(ROLES) for _ in range(n)]
    lines = [(f"I will send the {rng.choice(DOCS)} over by Friday.", r) for r in roles]

    def twin(role: str, tag: str) -> Twin:
        text = (f"When the {role.replace('_', ' ')} promises to send a document, the line is "
                f"rejected: that promise belongs to the {role.replace('_', ' ')}, not to delivery.")
        truth = ["reject" if r == role else "keep" for r in roles]
        rule = Explanation(text, "company", company=COMPANY, ref=f"{tag}-rule")
        return Twin(_deal(lines, truth, ["seller_promise"] * n, tag), ExplanationBank([rule]), truth)

    return twin(role_a, f"role{seed}a"), twin(role_b, f"role{seed}b")


@torch.no_grad()
def rule_following(model, twin: Twin, schema: Schema) -> float:
    """Share of lines whose predicted co_action matches what the rule text says."""
    batch = featurize(twin.deal, schema)
    out = model(batch.inputs(), company=COMPANY, bank=twin.rules)
    answers = [a.value for a in schema.by_key()["read:co_action"].answers]
    pred = [answers[i] for i in out.logits["read:co_action"].argmax(-1).tolist()]
    return sum(p == t for p, t in zip(pred, twin.truth)) / len(twin.truth)
