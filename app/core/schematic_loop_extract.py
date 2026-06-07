"""Loop-until-critic-quiet extraction — push automated coverage to its ceiling
so the human/verification queue is as small as possible.

A single extraction pass misses items on dense sheets (~56% measured). This loops:

  round 1: EXTRACT everything -> rows
  critic : "here is what was extracted; what is MISSING?" -> missed[]
  if missed: RE-EXTRACT targeting the missed items -> merge (dedup)
  repeat until the critic reports nothing new for `patience` rounds, or max_rounds.

That convergence ("dry") is how dense sheets climb from ~56% toward ~95%+ before a
human ever sees them. The VLM client is injected (default complete_vision) so the
loop is unit-testable without the network.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


def _parse_rows(text: str) -> list[dict]:
    if not text:
        return []
    s = re.sub(r"^```(json)?|```$", "", text.strip()).strip()
    a, b = s.find("{"), s.rfind("}")
    if a >= 0 and b > a:
        s = s[a:b + 1]
    try:
        obj = json.loads(s)
    except Exception:
        return []
    rows = obj.get("rows") or obj.get("items") or obj.get("missed") or []
    return [r for r in rows if isinstance(r, dict)]


def _key(row: dict) -> str:
    return json.dumps({k: row.get(k) for k in sorted(row)}, sort_keys=True).lower()[:160]


@dataclass
class LoopResult:
    rows: list[dict]
    rounds: int
    converged: bool
    missed_per_round: list[int] = field(default_factory=list)


def extract_until_quiet(image_b64: str, *, extract_prompt: str, critic_prompt: str,
                        vlm=None, max_rounds: int = 4, patience: int = 1,
                        max_tokens: int = 3000, mime: str = "image/png") -> LoopResult:
    """Loop extract->critic->re-extract until the critic stops finding new items.

    ``vlm(prompt, image_b64, max_tokens=...) -> str`` is injected; defaults to
    llm_client.complete_vision. Returns the merged, deduped rows."""
    if vlm is None:
        from app.core import llm_client
        def vlm(prompt, img, max_tokens=max_tokens):
            return llm_client.complete_vision(prompt, img, mime=mime, max_tokens=max_tokens)

    seen: set[str] = set()
    rows: list[dict] = []
    missed_per_round: list[int] = []
    dry = 0
    rounds = 0
    for r in range(max_rounds):
        rounds += 1
        prompt = extract_prompt if r == 0 else (
            extract_prompt + "\n\nFocus ONLY on items NOT in this list; do not repeat them:\n"
            + json.dumps([row.get("text") or row for row in rows])[:4000]
        )
        new = _parse_rows(vlm(prompt, image_b64, max_tokens=max_tokens))
        added = 0
        for row in new:
            k = _key(row)
            if k and k not in seen:
                seen.add(k)
                rows.append(row)
                added += 1
        # critic: what is still missing given everything so far?
        crit = vlm(
            critic_prompt + "\n\nAlready extracted (do not relist):\n"
            + json.dumps([row.get("text") or row for row in rows])[:4000],
            image_b64, max_tokens=1200,
        )
        missed = _parse_rows(crit)
        # count only genuinely-new missed items
        fresh_missed = [m for m in missed if _key(m) not in seen]
        missed_per_round.append(len(fresh_missed))
        if added == 0 and len(fresh_missed) == 0:
            dry += 1
            if dry >= patience:
                return LoopResult(rows, rounds, True, missed_per_round)
        else:
            dry = 0
        if not fresh_missed and added == 0:
            return LoopResult(rows, rounds, True, missed_per_round)
    return LoopResult(rows, rounds, False, missed_per_round)
