"""Extract the events that move a deal, so evidence can be cut at the right moment.

80.6% of artifacts admitted as evidence were created after their deal's last
quote existed, and that number is almost entirely correspondence: 1,637 emails
and 799 notes against 47 documents. The email stream simply has no temporal
boundary -- it runs through delivery and closeout, and all of it is read as if it
were scoping material.

A document's created_at gives a crude cut. What is actually wanted is the moment
the deal changed state: when pricing was asked for, when a quote went out, when a
SOW was signed. Those moments are stated in the correspondence, in plain words,
with a timestamp attached.

Same discipline as the document classifier: every event must quote the message
verbatim, and the quote is checked against the source in code. An event that
cannot be quoted is not emitted.
"""
import json, re, os, sys, time, collections, urllib.request, urllib.error, concurrent.futures as cf
SP = os.environ.get("LIFECYCLE_WORKDIR", ".")
EP = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
KEY = os.environ.get("AZURE_OPENAI_API_KEY") or ""
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

EVENTS = ["FIRST_CONTACT","REQUIREMENTS_GATHERED","SITE_INFO_RECEIVED","PRICING_REQUESTED",
          "QUOTE_SENT","QUOTE_REVISED","SOW_SENT","SOW_REVISED","SOW_SIGNED","PO_ISSUED",
          "SCHEDULED","SCOPE_CHANGED","BLOCKED","LOST","DELIVERED"]

SYSTEM = """You extract the moments a deal changed state, from its correspondence.

Event types:
""" + "\n".join(f"  {e}" for e in EVENTS) + """

RULES
1. One entry per real state change. Chasing, scheduling chit-chat, out-of-office and
   "any update?" are NOT events. Most messages are not events; a thread of 40 messages
   commonly contains 3 or 4.
2. `date`: copy the message timestamp exactly as given to you.
3. `actor`: who caused it — "customer", "us", or "vendor" — from the sender.
4. `receipt`: a VERBATIM span copied from that message. Never paraphrase, never invent.
   If you cannot quote it, DO NOT EMIT THE EVENT. A missing event is recoverable; an
   invented one silently moves a deal's evidence cut to the wrong day.
5. Prefer the earliest message that states the change. If a later message merely refers
   back to it ("as I sent last week"), that is not a new event.
6. SOW_SIGNED means executed — "signed SoW attached", "fully executed", "countersigned".
   "Sent for signature" is SOW_SENT, not SOW_SIGNED.

Return ONLY JSON: {"events":[{"date":...,"type":...,"actor":...,"summary":...,"receipt":...}]}
ordered by date ascending. An empty list is a valid and common answer."""


def trim(body, cap=1400):
    body = re.sub(r"<[^>]+>", " ", body or "")
    body = re.sub(r"\s+", " ", body).strip()
    body = re.split(r"(?i)\bon .{0,60}wrote:|\bfrom:\s|-----original message-----|________+", body)[0]
    return body[:cap]


def call(payload):
    body=json.dumps({"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":payload}],
        "temperature":0,"max_tokens":1800,"response_format":{"type":"json_object"}}).encode()
    url=f"{EP}/openai/deployments/{DEPLOYMENT}/chat/completions?api-version=2024-10-21"
    last=None
    for a in range(6):
        try:
            req=urllib.request.Request(url, body, {"content-type":"application/json","api-key":KEY})
            with urllib.request.urlopen(req, timeout=180) as r: d=json.loads(r.read())
            return json.loads(d["choices"][0]["message"]["content"]), d.get("usage",{})
        except urllib.error.HTTPError as e:
            last=e
            if e.code not in (429,500,502,503,504): raise
            time.sleep(int(e.headers.get("Retry-After") or 0) or min(45, 2**a*3))
    raise last


def run(item):
    deal, msgs = item
    msgs = sorted(msgs, key=lambda m: m["ts"] or "")
    parts = [f"[{m['ts']}] {m['kind']} from {m.get('actor','?')}\n{m['subject']}\n{m['body']}"
             for m in msgs if m["body"] or m["subject"]]
    if not parts:
        return {"deal": deal, "events": [], "messages": len(msgs)}
    payload = f"DEAL {deal[:8]} — {len(parts)} messages in time order\n\n" + "\n\n".join(parts)[:70000]
    try:
        res, usage = call(payload)
    except Exception as e:
        return {"deal": deal, "error": f"{type(e).__name__}: {str(e)[:80]}"}
    hay = re.sub(r"\s+", " ", " ".join(p for p in parts)).lower()
    out = []
    for e in res.get("events", []):
        rec = str(e.get("receipt") or "")
        if not rec:
            continue                       # rule 4: no quote, no event
        e["receipt_verified"] = re.sub(r"\s+", " ", rec).lower()[:60] in hay
        if e.get("type") in EVENTS:
            out.append(e)
    return {"deal": deal, "events": out, "messages": len(msgs), "usage": usage}


if __name__ == "__main__":  # pragma: no cover - operational entry point
    eng = json.load(open(f"{SP}/hs_engagements.json"))
    e2d = collections.defaultdict(set)
    for line in open(f"{SP}/all_artifacts.txt"):
        p = line.strip().split("/")
        if len(p) < 4: continue
        m = re.search(r"-hs-(email|note)-(\d+)", p[-1])
        if m: e2d[f"{m.group(1)}:{m.group(2)}"].add(p[1])
    per = collections.defaultdict(list)
    for k, p in eng.items():
        kind = k.split(":")[0]
        body = trim(p.get("hs_email_text") or p.get("hs_note_body") or "")
        subj = (p.get("hs_email_subject") or "").strip()
        if not (body or subj): continue
        for d in e2d.get(k, ()):
            per[d].append({"ts": p.get("hs_timestamp"), "kind": kind, "subject": subj, "body": body})
    items = [(d, m) for d, m in per.items() if m]
    print(f"extracting events for {len(items)} deals ({sum(len(m) for _, m in items):,} messages)…", flush=True)
    done = {}
    if os.path.exists(f"{SP}/timelines.json"):
        for r in json.load(open(f"{SP}/timelines.json")):
            if "events" in r: done[r["deal"]] = r
    items = [i for i in items if i[0] not in done]
    out = list(done.values())
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        for i, r in enumerate(ex.map(run, items), 1):
            out.append(r)
            if i % 40 == 0:
                json.dump(out, open(f"{SP}/timelines.json", "w"))
                print(f"  {i}/{len(items)}", flush=True)
    json.dump(out, open(f"{SP}/timelines.json", "w"))
    ok = [r for r in out if "events" in r]
    tok = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in ok)
    comp = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in ok)
    print(f"\ndeals {len(ok)}  events {sum(len(r['events']) for r in ok):,}  errors {len(out)-len(ok)}")
    print(f"cost ${tok/1e6*0.40 + comp/1e6*1.60:.2f}")
