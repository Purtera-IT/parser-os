"""Score an LLM scope-router against the DeepSeek gold labels.

The service-router head is a specialist over four packs measured at 0.529
held-out, and it drives `envelope.service_routing`, which drives
`detect_project_mode`, which decides which questions the PM asks the customer.
On Clayton it said `wireless` for a 437-store dispatch job and the PM was asked
for an AP count and a channel plan.

Before replacing it, measure the replacement. This scores any OpenAI-compatible
or Ollama endpoint against `_router_cache/*.json` -- the same DeepSeek labels the
head was trained on -- using the same `scope_summary` representation, which
excludes BOM noise (`pricing_assumption`, `commercial_total`, `rate_card`,
`line_item`). That exclusion is why the teacher is not fooled by 366 cabling
rows on a wireless deal while the keyword scorer is.

Run against the mac box::

    python _eval_router_llm.py --base-url http://100.114.102.122:11434 \
        --model qwen2.5:3b

Against the proxy, a hosted OpenAI-compatible API, or anything else::

    python _eval_router_llm.py --base-url https://api.deepseek.com \
        --model deepseek-chat --api-key-env DEEPSEEK_API_KEY --openai

``--limit`` scores a sample first; a full pass is 90 deals. Results print a
confusion summary so a model that is merely biased toward the majority class
(low_voltage_cabling, 39 of 90) cannot look good on accuracy alone.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

CACHE = Path("_router_cache")
PACKS_YAML_CANDIDATES = (
    Path("../Orbitbrief-Core/src/orbitbrief_core/world_model/data/domain_packs.yaml"),
    Path("domain_packs.yaml"),
)


def load_packs() -> list[tuple[str, str]]:
    """(pack_id, display_name) for the registry the labeller used."""
    import yaml
    for p in PACKS_YAML_CANDIDATES:
        if p.is_file():
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
            entries = raw if isinstance(raw, list) else (raw.get("packs") or [])
            out = []
            for e in entries:
                if isinstance(e, dict):
                    pid = e.get("id") or e.get("pack_id")
                    if pid:
                        out.append((pid, str(e.get("display_name") or pid)))
            if out:
                return out
    raise SystemExit(
        "domain_packs.yaml not found — pass --packs-yaml or clone Orbitbrief-Core alongside"
    )


def load_gold() -> list[tuple[str, str, str]]:
    """(slug, gold_primary, scope_summary)."""
    rows = []
    for p in sorted(glob.glob(str(CACHE / "*.json"))):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        primary = d.get("primary")
        summary = d.get("scope_summary") or ""
        if primary and summary:
            rows.append((Path(p).stem, primary, summary))
    return rows


def build_prompt(packs: list[tuple[str, str]], scope: str, max_chars: int) -> str:
    listing = "\n".join(f"- {pid}: {name}" for pid, name in packs)
    scope = scope[:max_chars]
    return (
        "You classify a services deal into its PRIMARY managed-service pack.\n"
        "Judge by the ACTUAL SCOPE OF WORK, not the customer or file names — a "
        "customer called \"Data Center Warehouse\" buying TV installs is "
        "audio_visual, not datacenter.\n\n"
        f"Packs:\n{listing}\n\n"
        f"Deal scope:\n{scope}\n\n"
        "Reply with ONLY the pack id, nothing else."
    )


def call_ollama(base: str, model: str, prompt: str, timeout: int) -> str:
    payload = {
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": 0.0, "num_predict": 16},
    }
    req = urllib.request.Request(
        f"{base.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return str(json.loads(r.read()).get("response") or "")


def call_openai(base: str, model: str, prompt: str, timeout: int, key: str) -> str:
    payload = {
        "model": model, "temperature": 0.0, "max_tokens": 16,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return str(body["choices"][0]["message"]["content"] or "")


def normalise(reply: str, valid: set[str]) -> str:
    """A small model will pad with prose; take the first valid pack id it says."""
    text = (reply or "").strip().lower()
    for token in re.findall(r"[a-z_]+", text):
        if token in valid:
            return token
    for pid in valid:                      # substring fallback
        if pid in text:
            return pid
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--openai", action="store_true", help="OpenAI-compatible /v1/chat/completions")
    ap.add_argument("--api-key-env", default="")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-scope-chars", type=int, default=6000)
    args = ap.parse_args()

    packs = load_packs()
    valid = {pid for pid, _ in packs}
    gold = load_gold()
    if args.limit:
        gold = gold[: args.limit]
    if not gold:
        print("no gold labels in _router_cache/", file=sys.stderr)
        return 2
    key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""

    right = 0
    abstain = 0
    confusion: dict[tuple[str, str], int] = collections.Counter()
    per_class: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    t0 = time.time()
    for i, (slug, want, scope) in enumerate(gold, 1):
        prompt = build_prompt(packs, scope, args.max_scope_chars)
        try:
            reply = (call_openai if args.openai else call_ollama)(
                args.base_url, args.model, prompt, args.timeout, *( [key] if args.openai else [] )
            )
        except Exception as exc:
            print(f"  {slug[:30]:32s} CALL FAILED {type(exc).__name__}: {str(exc)[:60]}")
            continue
        got = normalise(reply, valid)
        ok = got == want
        right += ok
        abstain += (got == "")
        confusion[(want, got or "<none>")] += 1
        per_class[want][1] += 1
        per_class[want][0] += ok
        print(f"  {i:3d}/{len(gold)} {slug[:28]:30s} want={want:22s} got={got or '<none>':22s} {'OK' if ok else ''}")

    n = len(gold)
    print(f"\nmodel={args.model}  endpoint={args.base_url}")
    print(f"accuracy      {right}/{n} ({100*right/max(n,1):.0f}%)   unparseable={abstain}")
    print(f"wall clock    {time.time()-t0:.0f}s  ({(time.time()-t0)/max(n,1):.1f}s per deal)")
    print("\nper class (gold count in brackets):")
    for cls, (ok, tot) in sorted(per_class.items(), key=lambda kv: -kv[1][1]):
        print(f"  {cls:26s} {ok:3d}/{tot:<3d} [{tot}]")
    print("\ntop confusions:")
    for (want, got), c in collections.Counter(confusion).most_common(10):
        if want != got:
            print(f"  {want:24s} -> {got:24s} {c}")
    # Majority-class baseline: a model that always guesses the biggest class.
    counts = collections.Counter(w for _, w, _ in gold)
    maj, majn = counts.most_common(1)[0]
    print(f"\nmajority-class baseline ({maj}): {majn}/{n} ({100*majn/n:.0f}%)")
    print(f"service-router head, recorded:   0.529 held-out (failed its eval gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
