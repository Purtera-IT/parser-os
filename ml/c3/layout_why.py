"""The same row under many layouts and field namings: same answers, same WHY (``layout_why``).

``layout_lock`` (losses.py) reads each table row in one of a handful of layouts precomputed by
runE/build.py ``rewrites()`` and ties the answers only. The swap gate (~/c3-local/runE/swap_gate.py) then showed
answers swinging 30-50% on layouts neither saw (HTML cells, fixed-width columns, a sentence), and ~20% on renamed
field labels. A handful of layouts teaches "these handful mean the same"; this teaches "the layout never decides":

``relayout(text, rng)``  a fresh layout of a row's cells (build.py's ``cells_of``), drawn per step from many
                         families (space-aligned columns, arrows, bullets, ``k(v)``, a key/value grid, YAML-ish,
                         XML-ish, ``k=v&k=v``, ``k of v``, comma-joined, numbered cells), with the keys kept,
                         upper/lower/title-cased, renamed (``synonym``: a broad map plus generated spellings) or
                         dropped, and the cells sometimes reordered. Values (numbers, names) stay verbatim.
``layout_why_loss``      one more forward with a sample of labeled table rows re-laid out: the saved answers (CE),
                         the clean reading (KL, detached) and the line's own WHY (``why_contrast`` on the moved
                         rows), as ``why_carry.keyword_dropout_loss`` does for masked keywords.

HELD OUT, never generated (``gate_shape`` checks every output; tests/test_c3_layout_why.py checks the generator
against swap_gate.py's own ``layout`` / ``layout_new``): kv_lines (one "k: v" per line), markdown (``| a | b |`` and a
``---`` rule), "k = v; ..." semicolons, HTML ``<td>``/``<tr>`` cells, fixed-width columns with no separator, and the
sentence "k is v, ... and v.". No generated field name uses a word of swap_gate's FIELD_SYN or SURFACE values.
"""
from __future__ import annotations

import random
import re
from dataclasses import replace
from typing import Any

# ---------------------------------------------------------------- cells (as runE/build.py)

_CELL_KEY = re.compile(r"^([A-Za-z][\w ./()#&-]{0,40}):\s+(.*)$")


def cells_of(text: str) -> list[tuple[str | None, str]]:
    """(key or None, value) per cell; tabs or pipes (runE/build.py ``cells_of``)."""
    t = str(text or "").strip()
    parts = [p.strip() for p in (t.split("\t") if "\t" in t and "|" not in t else t.split("|"))]
    out: list[tuple[str | None, str]] = []
    for p in parts:
        if not p:
            continue
        m = _CELL_KEY.match(p)
        out.append((m.group(1).strip(), m.group(2).strip()) if m else (None, p))
    return out


def is_row(text: str) -> bool:
    """A table row as swap_gate.is_row reads one: three cells or more, split by pipes or tabs."""
    t = str(text or "")
    return len(cells_of(t)) >= 3 and (t.count("|") >= 2 or t.count("\t") >= 2)


# ---------------------------------------------------------------- the gate's held-out vocabulary

#: swap_gate.FIELD_SYN values and SURFACE values (2026-10-27). The test re-reads swap_gate.py and fails if these
#: drift. Training never generates a field name with one of their words.
GATE_FIELD_SYN_VALUES = ("Municipality", "Mail code", "Territory", "Postal addr", "Premises", "Key", "Title",
                         "Venue", "Zone", "Telephone #", "POC", "E-addr", "Units", "Notes", "Time (h)", "Storeys",
                         "Footprint", "Outlet", "Outlet key", "Called")
GATE_SURFACE_VALUES = ("premises", "venue", "venues", "export", "ledger", "leaf", "panel", "roster", "matrix",
                       "record", "records", "attribute", "attributes", "caption", "appended", "annex", "postal",
                       "outlet", "outlets", "facility", "ledgers")
_TOK = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list[str]:
    return _TOK.findall(str(s).lower())


GATE_WORDS = frozenset(t for v in GATE_FIELD_SYN_VALUES + GATE_SURFACE_VALUES for t in _tokens(v))
#: Words that would make a line read like the gate's sentence family.
_SENTENCE_WORDS = frozenset({"is", "and", "are"})


def _blocked(token: str) -> bool:
    return (token in GATE_WORDS or token in _SENTENCE_WORDS
            or any(len(g) >= 4 and token.startswith(g) for g in GATE_WORDS))


def name_ok(new: str, old: str = "") -> bool:
    """A generated field name: no word the gate uses (also as a prefix: "mailing", "addr1"), unless the
    original name already had it (an upper-cased "Site Key" is still the training's own word)."""
    own = set(_tokens(old))
    return bool(str(new).strip()) and not any(_blocked(t) for t in _tokens(new) if t not in own)


# ---------------------------------------------------------------- field names, generated per step

#: Broad paraphrases per common field (lower-cased key -> names). Filtered by ``name_ok`` at import.
_SYN_POOL = {
    "city": ("Town", "Metro", "Locality", "Burg", "Township", "City name", "Urban area", "Place name"),
    "zip": ("ZIP5", "Zipcode", "Postcode", "Zip no", "ZIP-5", "Post no."),
    "state": ("Prov", "ST", "Province", "State/Prov", "Commonwealth", "Admin area"),
    "address": ("Street", "Street line", "Line 1", "Thoroughfare", "Delivery point", "Physical loc", "Street no."),
    "site": ("Property", "Locale", "Spot", "Plant", "Campus", "Workplace", "Place"),
    "site id": ("ID", "Ref no", "Num", "Ident", "Loc ID", "Prop ID", "Ref"),
    "site name": ("Label", "Name", "Display name", "Moniker", "Handle", "Nm"),
    "name": ("Label", "Display name", "Moniker", "Handle", "Nm", "Designation"),
    "location": ("Place", "Spot", "Locale", "Whereabouts", "Position", "Loc"),
    "region": ("District", "Division", "Market", "Area", "Sector", "Rgn"),
    "phone": ("Ph", "Tel no", "Mobile", "Cell", "Contact no", "Phone no", "Tel"),
    "contact": ("Person", "Rep", "Lead", "Liaison", "Owner", "Attn"),
    "email": ("Eml", "Inbox", "Electronic", "Mbox"),
    "qty": ("Count", "Num", "Amount", "Pcs", "Volume", "Nbr", "How many"),
    "quantity": ("Count", "Num", "Amount", "Pcs", "Volume", "Nbr"),
    "description": ("Details", "Summary", "Info", "Scope", "Remarks", "Comment", "Desc"),
    "hours": ("Hrs", "Duration", "Effort", "Labor", "Man-hours", "Est hrs"),
    "hrs": ("Hours", "Duration", "Effort", "Labor", "Man-hours"),
    "total hrs": ("Sum hours", "Hours total", "Effort", "Duration", "Est hrs"),
    "floors": ("Levels", "Stories", "Flrs", "Floor count"),
    "sq ft": ("Sqft", "Area", "Size", "SF", "Square feet", "Gross area"),
    "type": ("Kind", "Class", "Category", "Variant", "Sort"),
    "date": ("Day", "When", "Dt", "Scheduled", "On"),
    "status": ("Stage", "Condition", "Progress", "Sts"),
    "country": ("Nation", "Ctry"),
    "county": ("Parish", "Cnty"),
    "store": ("Shop", "Branch", "Storefront"),
    "store #": ("Shop no", "Branch no", "Store no"),
    "building": ("Bldg", "Structure", "Block"),
    "notes": ("Remarks", "Comments", "Memo"),
}
_SYN_POOL = {k: tuple(s for s in v if name_ok(s, k)) for k, v in _SYN_POOL.items()}
_PREFIX = ("Cust ", "Svc ", "Job ", "Loc ", "Prj ", "Ord ")
_SUFFIX = (" no.", " #", " info", " value", " field", " val", " txt")
_VOWEL = re.compile(r"(?<=\w)[aeiou]", re.I)


def _neutral(rng: random.Random, n: int) -> str:
    return rng.choice((f"c{n}", f"col{n}", f"F{n}", f"fld_{'abcdfgjkmn'[n % 10]}", f"x{n}", f"item{n}"))


def synonym(key: str, rng: random.Random, n: int = 1) -> str:
    """Another name for field ``key``, drawn this step: a paraphrase, a respelling (snake/camel/kebab case,
    an abbreviation, vowels dropped, a prefix or suffix), or a neutral column name. Never a gate word."""
    k = str(key or "").strip()
    low = k.lower()
    for _ in range(6):
        pick = rng.random()
        words = re.split(r"[\s_]+", k) if k else [""]
        if pick < 0.35 and _SYN_POOL.get(low):
            cand = rng.choice(_SYN_POOL[low])
        elif pick < 0.45:
            cand = "_".join(w.lower() for w in words)
        elif pick < 0.52:
            cand = words[0].lower() + "".join(w[:1].upper() + w[1:].lower() for w in words[1:])
        elif pick < 0.57:
            cand = "-".join(w.lower() for w in words)
        elif pick < 0.65:
            cand = (k[:3] + ".") if len(k) > 4 else k.upper()
        elif pick < 0.72:
            cand = _VOWEL.sub("", k) or k
        elif pick < 0.79:
            cand = rng.choice(_PREFIX) + k
        elif pick < 0.86:
            cand = k + rng.choice(_SUFFIX)
        else:
            cand = _neutral(rng, n)
        cand = cand.strip()
        if cand and cand.lower() != low and name_ok(cand, k):
            return cand
    return f"c{n}"


def _case(key: str, mode: str) -> str:
    return key.upper() if mode == "upper" else key.lower() if mode == "lower" else key.title()


# ---------------------------------------------------------------- held-out shapes (never emitted)

_HTML = re.compile(r"<\s*/?\s*(td|tr|th|table|tbody|thead)\b", re.I)
_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_KV_LINE = re.compile(r"^([A-Za-z][\w ./()#&-]{0,40}): (.*)$")


def gate_shape(text: str, source: str) -> str | None:
    """The swap gate's held-out family ``text`` has the shape of (a layout of the row ``source``), or None."""
    t, src = str(text or ""), str(source or "")
    values = {v for _, v in cells_of(src)}

    def intro(pat: str) -> bool:
        return len(re.findall(pat, t)) > len(re.findall(pat, src))

    if _HTML.search(t) and not _HTML.search(src):
        return "html"
    lines = [ln for ln in t.split("\n") if ln.strip()]
    if any(_RULE.match(ln) for ln in lines) or (t.strip().startswith("|") and t.strip().endswith("|")):
        return "markdown"
    if intro(r";") or intro(r" = "):
        return "semicolon"
    if intro(r"\bis\b") or intro(r"\band\b") or (t.rstrip().endswith(".") and not src.rstrip().endswith(".")):
        return "sentence"
    if len(lines) >= 2 and all(ln == ln.lstrip() and (ln.strip() in values or (
            (m := _KV_LINE.match(ln)) is not None and m.group(2).strip() in values)) for ln in lines):
        return "kv_lines"
    # Fixed width: one line, cells set apart by spaces alone (padded into columns, or bare values with any gap).
    if len(lines) == 1:
        rest = t
        for v in sorted(values, key=len, reverse=True):
            rest = rest.replace(v, "\x00")
        if "\x00" in rest and (not rest.replace("\x00", "").strip() or (
                intro(r" {3,}") and re.fullmatch(r"[A-Za-z0-9_ ./()#&\x00-]*", rest))):
            return "fixed"
    return None


# ---------------------------------------------------------------- layouts

SEPS = (" / ", " ~ ", " // ", " ¦ ", " • ", " >> ", " :: ", " ,")
FAMILIES = ("spaced", "arrow", "bullets", "paren", "grid", "yaml", "xml", "query", "of", "comma", "numbered")
KEY_MODES = ("keep", "upper", "lower", "title", "rename", "rename", "drop", "mixed")


def _tag(key: str | None, n: int) -> str:
    t = re.sub(r"[^A-Za-z0-9_]+", "_", str(key or "")).strip("_").lower()
    if not t or not t[0].isalpha():
        t = f"c{n}" if not t else f"f_{t}"
    return f"f_{t}" if t in {"td", "tr", "th", "table", "tbody", "thead"} else t


def _render(fam: str, cells: list[tuple[str | None, str]], rng: random.Random) -> str:
    kv = [f"{k}: {v}" if k else v for k, v in cells]
    if fam == "spaced":
        # Only when every cell keeps "k: v": bare values spaced apart are the gate's fixed width.
        return (" " * rng.choice((2, 3, 4))).join(kv) if all(k for k, _ in cells) else ""
    if fam == "arrow":
        arrow = rng.choice(("->", "=>", "→", "⇒"))
        return rng.choice(SEPS).join(f"{k} {arrow} {v}" if k else v for k, v in cells)
    if fam == "bullets":
        b = rng.choice(("-", "*", "•", "+", "·"))
        bare = rng.random() < 0.3
        return "\n".join(f"{b} {v}" if bare or not k else f"{b} {k}: {v}" for k, v in cells)
    if fam == "paren":
        sp = rng.choice(("", " "))
        return rng.choice(SEPS).join(f"{k}{sp}({v})" if k else v for k, v in cells)
    if fam == "grid":
        sep = rng.choice(("  ", " ¦ ", " / ", " // "))
        return sep.join(k or f"c{n}" for n, (k, _) in enumerate(cells, 1)) + "\n" + sep.join(v for _, v in cells)
    if fam == "yaml":
        if rng.random() < 0.5:   # a mapping under a heading
            head = rng.choice(("entry:", "item:", "row_data:"))
            return head + "\n" + "\n".join(f"  {k}: {v}" if k else f"  - {v}" for k, v in cells)
        # a list item: "- k: v" then the rest indented under it
        return "- " + "\n  ".join(f"{k}: {v}" if k else f"- {v}" for k, v in cells)
    if fam == "xml":
        inner = "".join(f"<{_tag(k, n)}>{v}</{_tag(k, n)}>" for n, (k, v) in enumerate(cells, 1))
        wrap = rng.choice(("", "item", "entry", "line"))
        return f"<{wrap}>{inner}</{wrap}>" if wrap else inner
    if fam == "query":
        q = "&".join(f"{_tag(k, n)}={v}" for n, (k, v) in enumerate(cells, 1))
        return ("?" + q) if rng.random() < 0.3 else q
    if fam == "of":
        return rng.choice(SEPS).join(f"{k} of {v}" if k else v for k, v in cells)
    if fam == "comma":
        return rng.choice((", ", ",", " , ")).join(kv)
    if fam == "numbered":
        style = rng.choice(("{n}) ", "[{n}] ", "#{n} ", "({n}) ", "{n}/ "))
        sep = rng.choice((" ", "  ", "\n"))
        return sep.join(style.format(n=n) + c for n, c in enumerate(kv, 1))
    raise ValueError(fam)


def _keys(cells: list[tuple[str | None, str]], mode: str, rng: random.Random) -> list[tuple[str | None, str]]:
    out = []
    for n, (k, v) in enumerate(cells, 1):
        m = rng.choice(("keep", "upper", "lower", "rename", "drop")) if mode == "mixed" else mode
        if not k or m == "keep":
            out.append((k, v))
        elif m == "drop":
            out.append((None, v))
        elif m == "rename":
            out.append((synonym(k, rng, n), v))
        else:
            out.append((_case(k, m), v))
    return out


def relayout(text: str, rng: random.Random | None = None, *, family: str | None = None,
             key_mode: str | None = None, shuffle: float = 0.3) -> str | None:
    """A fresh layout of table row ``text`` (None when it is not a row or nothing new came out). Never one of the
    swap gate's held-out families (``gate_shape``)."""
    rng = rng or random
    if not is_row(text):
        return None
    base = cells_of(text)
    if any("\n" in v for _, v in base):
        return None
    for _ in range(4):
        cells = list(base)
        if len(cells) >= 3 and rng.random() < shuffle:
            rng.shuffle(cells)
        cells = _keys(cells, key_mode or rng.choice(KEY_MODES), rng)
        out = _render(family or rng.choice(FAMILIES), cells, rng).strip()
        if out and " ".join(out.split()) != " ".join(str(text).split()) and gate_shape(out, text) is None:
            return out
    return None


# ---------------------------------------------------------------- the loss

def layout_why_loss(model, out, batch, w, *, rng: random.Random | None = None,
                    desc: dict[str, Any] | None = None):
    """One more forward with a sample of labeled table rows in a fresh layout (``relayout``): the saved answers,
    the clean reading (KL to the first pass, detached) and the line's own WHY (``why_contrast`` on the moved
    rows). Sampling: each labeled row with probability ``layout_why_p``, at most ``layout_why_cap`` per part."""
    import torch
    from torch.nn import functional as F

    from .data import number_tokens
    from .losses import head_losses, row_weights  # noqa: PLC0415 (cycle)
    from .why_carry import why_contrast_loss  # noqa: PLC0415 (cycle)

    rng = rng or random
    zero = out.r.new_zeros(())
    p, cap = float(getattr(w, "layout_why_p", 0.3)), int(getattr(w, "layout_why_cap", 128))
    texts, numbers = list(batch.texts), list(batch.numbers)
    lines: list[int] = []
    for i, on in enumerate(batch.labeled):
        if not on or not is_row(batch.texts[i]) or rng.random() >= p:
            continue
        t = relayout(batch.texts[i], rng)
        if t is None:
            continue
        texts[i], numbers[i] = t, number_tokens(t)
        lines.append(i)
        if len(lines) >= cap:
            break
    if not lines:
        return zero
    keep = set(lines)
    base = batch.weights if len(batch.weights) == len(batch) else [1.0] * len(batch)
    b = replace(batch, texts=texts, numbers=numbers, layout_rewrites=[],
                weights=[x if i in keep else 0.0 for i, x in enumerate(base)])
    moved = model(b.inputs(), company=batch.company, desc=desc)
    ce, _ = head_losses(model, moved, b, w, "universal")
    co, _ = head_losses(model, moved, b, w, "company")
    rows = torch.tensor(lines, device=out.r.device)
    rw = row_weights(b, out.r.device)[rows]
    kl, n = zero, 0
    for key, logits in moved.logits.items():
        ref = out.logits.get(key)
        opp = model.schema.by_key().get(key)
        if ref is None or opp is None or opp.multiple or logits.shape != ref.shape:
            continue
        per = F.kl_div(F.log_softmax(logits[rows].float(), -1),
                       F.log_softmax(ref[rows].detach().float(), -1),
                       reduction="none", log_target=True).sum(-1)
        kl = kl + (per * rw).sum() / rw.sum().clamp(min=1e-6)
        n += 1
    tied = why_contrast_loss(model, moved, batch, hard=float(getattr(w, "why_hard_negative", 4.0)), rows=lines)
    return ce + w.conduct * co + (kl / n if n else zero) + tied


__all__ = ["FAMILIES", "GATE_FIELD_SYN_VALUES", "GATE_SURFACE_VALUES", "GATE_WORDS", "cells_of", "gate_shape",
           "is_row", "layout_why_loss", "name_ok", "relayout", "synonym"]
