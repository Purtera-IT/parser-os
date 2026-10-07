"""Cross-email threading (compile stage).

Each ``.eml`` arrives as its own artifact and is parsed in isolation, so a
short reply — *"Yes, approved, go ahead with 36."* — lands as an atom with no
idea what it is answering. This stage reconstructs the **conversation** across
those separate files and stamps every atom with its thread position and the
gist of the message it is replying to, so a one-line reply carries the context
it was written against (the way a transcript carries the turns before it).

Design guarantees
-----------------
* **Universal.** Grouping uses RFC 5322 ``Message-ID`` / ``In-Reply-To`` /
  ``References`` first, then a subject-normalisation fallback (``Re:``/``Fwd:``
  stripped). No per-deal vocabulary. One compile == one deal, so subject
  grouping cannot bleed across deals.
* **Lossless / additive.** This stage *only* writes ``value["email_thread"]``
  onto existing atoms. It never removes, splits, reorders, or rewrites an atom,
  never changes an atom ``id``. Nothing can be dropped here — proven by the
  no-drop regression test.
* **Safe.** Any malformed metadata is skipped; the worst case is an email that
  simply isn't threaded (it keeps all its atoms, just without thread context).
"""

from __future__ import annotations

import re
from typing import Any

from app.core.ids import stable_id
from app.core.schemas import EvidenceAtom
from app.core.suppression_ledger import note_folded_into

_GIST_MAX = 160

# Lines that are pure social padding or signature boilerplate carry no scope
# meaning, so they make a useless "in reply to" gist. We skip them and fall
# through to the first substantive sentence. Universal — no per-deal vocab.
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|dear|good\s+(morning|afternoon|evening)|greetings|team)\b[\s,!:-]*",
    re.IGNORECASE,
)
_CLOSING_RE = re.compile(
    r"^(thanks?|thank\s+you|thx|regards|best|best\s+regards|sincerely|cheers|"
    r"warm\s+regards|kind\s+regards|respectfully|talk\s+soon|cordially|"
    r"sent\s+from\s+my|get\s+outlook|let\s+me\s+know|please\s+let\s+me\s+know)"
    r"\b[\s,!.:-]*$",
    re.IGNORECASE,
)
# Signature / contact-block lines (phone, title, dept, address fragments).
_SIGNATURE_RE = re.compile(
    r"(^\+?\d[\d\s().-]{6,}\d$)|(\b(?:office|cell|mobile|direct|tel|fax|ext)\b\s*[:#]?\s*\+?\d)"
    r"|(@[\w.-]+\.\w{2,}$)|(\bwww\.)|(https?://)",
    re.IGNORECASE,
)


class _Union:
    """Tiny union-find over artifact ids for thread grouping."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression.
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic: smaller id wins as root.
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def _is_email_header(atom: EvidenceAtom) -> bool:
    v = atom.value if isinstance(atom.value, dict) else {}
    return v.get("kind") == "email_header" and isinstance(v.get("email_thread_meta"), dict)


def _gist_for_artifact(atoms: list[EvidenceAtom]) -> str:
    """Best one-line summary of a message: the first substantive, non-quoted
    body line. Falls back to any non-header line, then to the subject."""
    candidates: list[tuple[int, str]] = []
    fallback: list[tuple[int, str]] = []
    for atom in atoms:
        v = atom.value if isinstance(atom.value, dict) else {}
        if v.get("kind") == "email_header":
            continue
        if v.get("kind") in {"attachment", "attachment_marker", "email_attachment"}:
            continue
        # A person record read from the signature is who wrote, not what.
        if str(getattr(atom.atom_type, "value", atom.atom_type)) == "stakeholder":
            continue
        text = (atom.raw_text or "").strip()
        if not text or not any(c.isalnum() for c in text):
            continue
        order = int(v.get("message_index", 0) or 0)
        if v.get("quoted"):
            fallback.append((order, text))
        else:
            candidates.append((order, text))
    pool = candidates or fallback
    if not pool:
        return ""
    pool.sort(key=lambda t: t[0])

    # Prefer the first SUBSTANTIVE line: skip greetings ("Hi Hiran,"),
    # sign-offs ("Thanks,"), and signature/contact lines. A bare greeting
    # gist is useless context for a reply. If everything is padding (rare),
    # fall back to the first line so we never return empty.
    def _is_padding(line: str) -> bool:
        stripped = line.strip()
        if _GREETING_RE.match(stripped):
            # A greeting prefix only — but "Hi, can you confirm 36?" is real.
            # "Hi Hiran," (greeting + name) is padding; "Hi, please send the
            # SOW" is substantive. Treat the remainder as padding when it's
            # empty or just a short name (<=2 capitalized alpha words).
            remainder = _GREETING_RE.sub("", stripped).strip().rstrip(",.!:")
            if len(remainder) < 3:
                return True
            words = remainder.split()
            if len(words) <= 2 and all(w.isalpha() and w[:1].isupper() for w in words):
                return True
            return False
        if _CLOSING_RE.match(stripped):
            return True
        if _SIGNATURE_RE.search(stripped):
            return True
        # Pure name line (1-3 capitalized words, no verb-ish content).
        if len(stripped) <= 24 and stripped.replace(".", "").replace(",", "").isalpha():
            words = stripped.split()
            if len(words) <= 3 and all(w[:1].isupper() for w in words if w):
                return True
        return False

    gist = ""
    for _, line in pool:
        if not _is_padding(line):
            # Strip a leading greeting prefix if the substantive content
            # follows it on the same line ("Hi Hiran, please confirm 36").
            cleaned = _GREETING_RE.sub("", line).strip() or line.strip()
            gist = cleaned
            break
    if not gist:
        gist = pool[0][1]
    return gist[: _GIST_MAX - 1] + "\u2026" if len(gist) > _GIST_MAX else gist


def thread_emails(
    atoms: list[EvidenceAtom], *, project_id: str = ""
) -> tuple[list[EvidenceAtom], dict[str, Any]]:
    """Group email artifacts into conversations and stamp thread context.

    Returns ``(atoms, summary)``. ``atoms`` is the same list, same objects,
    same ids — only ``value["email_thread"]`` is added on email atoms. The
    summary is telemetry: thread / message counts and a per-thread digest.
    """
    # 1) Per-artifact email metadata (only .eml emit a header atom w/ meta).
    meta_by_artifact: dict[str, dict[str, Any]] = {}
    atoms_by_artifact: dict[str, list[EvidenceAtom]] = {}
    for atom in atoms:
        aid = atom.artifact_id
        atoms_by_artifact.setdefault(aid, []).append(atom)
        if _is_email_header(atom):
            meta = dict(atom.value["email_thread_meta"])
            meta_by_artifact[aid] = meta

    if not meta_by_artifact:
        return atoms, {"thread_count": 0, "threaded_message_count": 0, "multi_message_threads": 0, "threads": []}

    # 2) Union-find grouping. RFC headers first, subject_norm as the safety net.
    uf = _Union()
    msgid_to_artifact: dict[str, str] = {}
    for aid, meta in meta_by_artifact.items():
        uf.add(aid)
        mid = (meta.get("message_id") or "").strip()
        if mid and mid not in msgid_to_artifact:
            msgid_to_artifact[mid] = aid

    # parent_by_artifact: the TRUE message this one answers, resolved from
    # In-Reply-To (preferred) or the nearest ancestor in References (last entry
    # is the most-recent ancestor). Used for parent-accurate reply context so a
    # branching thread doesn't mislabel "in reply to" as the chronologically
    # previous email. Falls back to chronological prev when headers are absent.
    parent_by_artifact: dict[str, str] = {}
    # Artifacts whose thread membership was decided by RFC headers. The subject
    # fallback must not move these -- see below.
    header_linked: set[str] = set()

    # Every reference token -> every artifact that cites it in In-Reply-To or
    # References, whether or not that token also happens to BE another
    # artifact's own Message-ID.
    #
    # msgid_to_artifact-only resolution assumed References always names an
    # ancestor EMAIL present in this compile -- true RFC 5322 semantics, where
    # a reply's References is a chain of prior Message-IDs. HubSpot instead
    # stamps every message in a conversation with the SAME synthetic anchor
    # (References: <hs-thread-…@hubspot.invalid>), identical across all of
    # them and matching no individual email's Message-ID. That token was never
    # a key in msgid_to_artifact, so it never resolved, header_linked was never
    # set, and grouping fell through to the subject fallback -- which then
    # failed too, because "Fw: Time Clock Installs…" and "RE: 010215 Time
    # Clock Installs…" are not the same normalised subject. All 11 attachments
    # landed on a message severed from the six-message discussion that
    # delivered them.
    #
    # The fix generalises rather than special-cases HubSpot: union any two
    # artifacts that cite the SAME reference token, resolvable or not. A real
    # ancestor Message-ID still works exactly as before (two replies to the
    # same email cite it and are unioned); a shared synthetic anchor now works
    # too, by the identical mechanism, with no vendor name in the logic.
    refs_by_artifact: dict[str, list[str]] = {}
    citers_by_token: dict[str, list[str]] = {}
    for aid, meta in meta_by_artifact.items():
        refs: list[str] = []
        if meta.get("in_reply_to"):
            refs.append(str(meta["in_reply_to"]).strip())
        refs.extend(str(r).strip() for r in (meta.get("references") or []))
        refs = [r for r in refs if r]
        refs_by_artifact[aid] = refs
        for ref in refs:
            citers_by_token.setdefault(ref, []).append(aid)

    for aid, refs in refs_by_artifact.items():
        for ref in refs:
            other = msgid_to_artifact.get(ref)
            if other and other != aid:
                uf.union(aid, other)
                header_linked.add(aid)
                header_linked.add(other)
        for ref in refs:
            co_citers = citers_by_token.get(ref) or []
            if len(co_citers) < 2:
                continue
            for other in co_citers:
                if other != aid:
                    uf.union(aid, other)
                    header_linked.add(aid)
                    header_linked.add(other)

    for aid, meta in meta_by_artifact.items():
        # First resolvable ancestor (In-Reply-To beats References; among
        # References the last is nearest) becomes the parent for context.
        in_reply_first = (str(meta.get("in_reply_to") or "").strip(),)
        refs_nearest_first = tuple(
            str(r).strip() for r in reversed(meta.get("references") or [])
        )
        for ref in in_reply_first + refs_nearest_first:
            other = msgid_to_artifact.get(ref)
            if other and other != aid:
                parent_by_artifact[aid] = other
                break

    # Subject fallback: union messages sharing a non-empty normalised subject,
    # for the ones RFC headers could not place. Within one deal compile this
    # reunites a back-and-forth whose .eml export stripped the References chain.
    #
    # It is a FALLBACK and was not behaving as one: it unioned every subject
    # match unconditionally, so two different conversations that happen to share
    # a line -- "Site Survey" on two deals -- merged even when their headers
    # said otherwise, and headers could never win.
    #
    # A message already placed by its headers is left alone. One that has none
    # still gets the safety net, which is the case the fallback exists for.
    subject_groups: dict[str, list[str]] = {}
    for aid, meta in meta_by_artifact.items():
        if aid in header_linked:
            continue
        subj = (meta.get("subject_norm") or "").strip()
        if subj:
            subject_groups.setdefault(subj, []).append(aid)
    for group in subject_groups.values():
        first = group[0]
        for other in group[1:]:
            uf.union(first, other)

    # 3) Collect threads: root -> [artifact_ids].
    threads: dict[str, list[str]] = {}
    for aid in meta_by_artifact:
        threads.setdefault(uf.find(aid), []).append(aid)

    # Stable encounter order for undated tie-breaks.
    encounter_index = {aid: i for i, aid in enumerate(meta_by_artifact)}

    summary_threads: list[dict[str, Any]] = []
    multi = 0
    for members in threads.values():
        # 4) Order messages: dated chronologically first, then undated in
        # encounter order — fully deterministic.
        def _sort_key(aid: str) -> tuple[int, float, int]:
            ep = float(meta_by_artifact[aid].get("date_epoch") or 0.0)
            has_date = 0 if ep > 0 else 1
            return (has_date, ep, encounter_index[aid])

        ordered = sorted(members, key=_sort_key)
        member_set = set(ordered)
        size = len(ordered)
        if size > 1:
            multi += 1

        # Deterministic thread id from the earliest message id / subject / ids.
        root_meta = meta_by_artifact[ordered[0]]
        root_key = (
            (root_meta.get("message_id") or "").strip()
            or (root_meta.get("subject_norm") or "").strip()
            or "|".join(sorted(ordered))
        )
        thread_id = stable_id("thr", project_id, root_key)

        gist_by_artifact = {
            aid: _gist_for_artifact(atoms_by_artifact.get(aid, [])) for aid in ordered
        }
        subject = (
            root_meta.get("subject")
            or root_meta.get("subject_norm")
            or ""
        )

        senders: list[str] = []
        for pos, aid in enumerate(ordered):
            meta = meta_by_artifact[aid]
            sender = (meta.get("sender") or "").strip()
            if sender:
                senders.append(sender)
            replied_to: dict[str, str] | None = None
            context = ""
            # Parent-accurate: the message this one actually answers
            # (In-Reply-To/References), falling back to the chronologically
            # previous message in the thread when no header parent resolved.
            parent_aid = parent_by_artifact.get(aid)
            if parent_aid not in member_set:
                parent_aid = None
            via = "in_reply_to"
            if parent_aid is None and pos > 0:
                parent_aid = ordered[pos - 1]
                via = "chronological"
            if parent_aid is not None:
                prev_meta = meta_by_artifact[parent_aid]
                prev_sender = (prev_meta.get("sender") or "").strip()
                prev_gist = gist_by_artifact.get(parent_aid, "")
                replied_to = {
                    "sender": prev_sender,
                    "gist": prev_gist,
                    "date": (prev_meta.get("date_raw") or "").strip(),
                    "via": via,
                }
                if prev_gist:
                    who = prev_sender or "previous message"
                    context = f'In reply to {who}: "{prev_gist}"'

            thread_block: dict[str, Any] = {
                "thread_id": thread_id,
                "thread_index": pos + 1,
                "thread_size": size,
                "subject": subject,
                "subject_norm": (meta.get("subject_norm") or "").strip(),
                "sender": sender,
                "to": [str(x).strip() for x in (meta.get("to") or [])][:12],
                "cc": [str(x).strip() for x in (meta.get("cc") or [])][:12],
                "date": (meta.get("date_raw") or "").strip(),
                "gist": gist_by_artifact.get(aid, ""),
            }
            if replied_to is not None:
                thread_block["replied_to"] = replied_to
            if context:
                thread_block["context"] = context

            # 5) Additive stamp on EVERY atom of this artifact. We touch only
            # value["email_thread"] — no id / type / text change → no drops.
            # The file-level block describes the FILE (its top message). A
            # quoted block inside the file is an older message with its own
            # author and time; its atoms say so, and say which message they
            # answer (the next-older block in the same file) — live 010300:
            # Carl's requirements were stamped "sender: patrick@purtera-it.com"
            # and the audit could not tell who was replying to whom.
            _blocks = _message_blocks(atoms_by_artifact.get(aid, []))
            for atom in atoms_by_artifact.get(aid, []):
                if isinstance(atom.value, dict):
                    tb = dict(thread_block)
                    mi = atom.value.get("message_index")
                    try:
                        mi = int(mi) if mi is not None else None
                    except (TypeError, ValueError):
                        mi = None
                    if mi is not None and mi in _blocks:
                        me = _blocks[mi]
                        tb["message"] = {"index": mi, "author": me["author"], "sent_at": me["sent_at"], "quoted": mi > 0}
                        older = _blocks.get(mi + 1)
                        if older:
                            tb["in_reply_to"] = {"author": older["author"], "sent_at": older["sent_at"], "gist": older["gist"]}
                        newer = _blocks.get(mi - 1) if mi > 0 else None
                        if newer:
                            tb["answered_by"] = {"author": newer["author"], "sent_at": newer["sent_at"]}
                        # 1 = the earliest message in this file's history.
                        tb["position_in_file"] = len(_blocks) - mi
                    atom.value["email_thread"] = tb

        _number_thread_messages(
            ordered, meta_by_artifact,
            {aid: _message_blocks(atoms_by_artifact.get(aid, [])) for aid in ordered},
            atoms_by_artifact,
        )

        summary_threads.append(
            {
                "thread_id": thread_id,
                "subject": subject,
                "size": size,
                "senders": senders,
            }
        )

    summary_threads.sort(key=lambda t: (-int(t["size"]), str(t["subject"])))
    summary = {
        "thread_count": len(threads),
        "threaded_message_count": len(meta_by_artifact),
        "multi_message_threads": multi,
        "threads": summary_threads,
    }
    return atoms, summary


def _number_thread_messages(
    ordered: list[str],
    meta_by_artifact: dict[str, dict[str, Any]],
    blocks_by_artifact: dict[str, dict[int, dict[str, str]]],
    atoms_by_artifact: dict[str, list[EvidenceAtom]],
) -> int:
    """One chronological numbering for every MESSAGE of a thread.

    ``thread_index`` numbers FILES. A message that exists only as a quote --
    010003's Adobe Sign notice at 1:04 PM, Sarah's 1:19 PM email -- has no
    file, so it had no number, and the thread appeared to start partway
    through. Each message is identified by sender address and send minute
    (the zone-free stamp ``dedup_quoted_history`` already matches quoted
    headers on), so a quoted copy of a message that also has its own file
    shares that file's number. Writes ``message.thread_position`` (1 =
    earliest) and ``message.thread_message_count`` on every stamped atom.
    """
    from app.parsers.email_parser import _parse_date_epoch

    def _epoch(raw: str) -> float:
        ep = _parse_date_epoch(raw or "")
        if ep:
            return ep
        st = _minute_stamp(raw or "")
        if not st:
            return 0.0
        from datetime import datetime, timezone

        day, minute = st.split("|")
        # A quote's local clock, read as UTC: good to the day, which is all a
        # tie with a dated file needs.
        return datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() + int(minute) * 60

    def _ident(sender: str, sent: str) -> tuple[str, str] | None:
        a, m = _address(sender), _minute_stamp(sent)
        return (a, m) if a and m else None

    entries: list[dict[str, Any]] = []
    by_ident: dict[tuple[str, str], dict[str, Any]] = {}
    slot: dict[tuple[str, int], dict[str, Any]] = {}
    for fi, aid in enumerate(ordered):
        meta = meta_by_artifact.get(aid, {})
        own = {"epoch": float(meta.get("date_epoch") or 0.0), "order": (fi, 0)}
        entries.append(own)
        slot[(aid, 0)] = own
        ident = _ident(str(meta.get("sender") or ""), str(meta.get("date_raw") or ""))
        if ident:
            for st in _minute_stamps_around(str(meta.get("date_raw") or "")):
                by_ident.setdefault((ident[0], st), own)
    for fi, aid in enumerate(ordered):
        for mi, rec in sorted((blocks_by_artifact.get(aid) or {}).items()):
            if mi == 0:
                continue
            ident = _ident(rec.get("author") or "", rec.get("sent_at") or "")
            hit = by_ident.get(ident) if ident else None
            if hit is None:
                # Older quotes sit deeper in the file: higher index, earlier.
                hit = {"epoch": _epoch(rec.get("sent_at") or ""), "order": (fi, -mi)}
                entries.append(hit)
                if ident:
                    by_ident[ident] = hit
            slot[(aid, mi)] = hit
    entries.sort(key=lambda e: (e["epoch"] if e["epoch"] else float("inf"), e["order"]))
    for n, e in enumerate(entries, start=1):
        e["position"] = n
    total = len(entries)
    for aid in ordered:
        for atom in atoms_by_artifact.get(aid, []):
            v = atom.value if isinstance(atom.value, dict) else None
            tb = v.get("email_thread") if v else None
            msg = tb.get("message") if isinstance(tb, dict) else None
            if not isinstance(msg, dict):
                continue
            e = slot.get((aid, int(msg.get("index") or 0)))
            if e is not None:
                msg["thread_position"] = e["position"]
                msg["thread_message_count"] = total
    return total


def _message_blocks(atoms: list[EvidenceAtom]) -> dict[int, dict[str, str]]:
    from app.parsers.email_parser import _is_greeting_line

    """message_index -> {author, sent_at, gist} for one email artifact, read off
    the atoms' own author stamps (the block's From:/Sent:)."""
    out: dict[int, dict[str, str]] = {}
    for a in atoms:
        v = a.value if isinstance(a.value, dict) else None
        if not v or v.get("message_index") is None:
            continue
        try:
            mi = int(v.get("message_index"))
        except (TypeError, ValueError):
            continue
        rec = out.setdefault(mi, {"author": "", "sent_at": "", "gist": ""})
        kind = str(v.get("kind") or "")
        # A quoted block's own "From: / Sent:" header names its author even
        # when every line under it is chatter ("Let's go!!" and a name): the
        # block used to come out authorless, and the line was read as part
        # of the message it answers.
        _author = v.get("author") or (v.get("sender") if kind == "quoted_message_header" else None)
        _sent = v.get("authored_at") or (v.get("sent_at") if kind == "quoted_message_header" else None)
        if str(_author or "").strip().lower() == "unknown":
            _author = None
        if not rec["author"] and _author:
            rec["author"] = str(_author)
        if not rec["sent_at"] and _sent:
            rec["sent_at"] = str(_sent)
        text = str(a.raw_text or "").strip()
        # A body line is the gist; a message that is only a pleasantry
        # ("Thank you for the opportunity!") falls back to that line.
        if kind == "email_body_line" and len(text.split()) >= 4 and not _is_greeting_line(text):
            if not rec["gist"] or rec.get("_fallback"):
                rec["gist"] = text[:120]
                rec.pop("_fallback", None)
        elif kind == "email_body_context" and not rec["gist"] and len(text.split()) >= 3:
            rec["gist"] = text[:120]
            rec["_fallback"] = "1"
    for rec in out.values():
        rec.pop("_fallback", None)
    return out


def _norm_key(atom: EvidenceAtom) -> str:
    """Collapse key for quoted-history matching: normalized text, whitespace-
    folded. Quote markers were already stripped at parse time.

    A list item is keyed under its list label ("provided by us:|relay"): the
    item alone is often one word, too short to collapse on text, but the
    same item under the same label is the same fact. Live 010289: the ask's
    "Relay" came back quoted in 9 replies."""
    txt = (getattr(atom, "normalized_text", "") or getattr(atom, "raw_text", "") or "").strip()
    key = re.sub(r"\s+", " ", txt).lower()
    v = atom.value if isinstance(atom.value, dict) else {}
    if v.get("list_item") and v.get("list_label") and key:
        return f"{str(v['list_label']).strip().lower()}|{key}"
    return key


# A quoted line this short is too generic to safely collapse on text alone
# ("yes", "ok", "thanks") — keep it. Real quoted history that inflates the
# atom stream is full sentences well above this.
_MIN_DEDUP_LEN = 12


_ADDR_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def _address(sender: str) -> str:
    m = _ADDR_RE.search(sender or "")
    return m.group(0).lower() if m else ""


def _minute_stamp(raw: str) -> str:
    """Day + minute of a send time, in whatever zone it was written.

    A quote shows the recipient's local time ("Wednesday, September 2, 2026
    10:38 AM") and the original's header shows another zone ("Wed, 02 Sep
    2026 14:38:56 +0000"): the hour differs, the minute and the date (within
    a day) do not. ``YYYY-MM-DD|MM`` is matched with the date loosened below.
    """
    from datetime import datetime
    from email.utils import parsedate_to_datetime

    raw = (raw or "").strip()
    if not raw:
        return ""
    dt = None
    try:
        dt = parsedate_to_datetime(raw)
    except Exception:
        dt = None
    if dt is None:
        # Gmail's "Mon, Jul 10, 2026 at 9:04 AM" (quote attribution).
        _plain = re.sub(r"\s+at\s+", " ", raw).replace(" , ", ", ")
        for fmt in ("%A, %B %d, %Y %I:%M %p", "%A, %B %d, %Y %H:%M", "%B %d, %Y %I:%M %p", "%m/%d/%Y %I:%M %p",
                    "%a, %b %d, %Y %I:%M %p", "%a, %b %d, %Y, %I:%M %p", "%a, %b %d, %Y %H:%M",
                    "%b %d, %Y %I:%M %p", "%m/%d/%y %I:%M %p", "%m/%d/%y, %I:%M %p", "%a, %d %b %Y %H:%M"):
            try:
                dt = datetime.strptime(_plain, fmt)
                break
            except ValueError:
                continue
    if dt is None:
        return ""
    return f"{dt.date().isoformat()}|{dt.minute:02d}"


def _minute_stamps_around(raw: str) -> set[str]:
    """The original's stamp and its neighbours a day either side: a zone
    shift can move the date by one, never the minute."""
    from datetime import date, timedelta

    s = _minute_stamp(raw)
    if not s:
        return set()
    day, minute = s.split("|")
    d = date.fromisoformat(day)
    return {f"{(d + timedelta(days=k)).isoformat()}|{minute}" for k in (-1, 0, 1)}


def dedup_quoted_history(
    atoms: list[EvidenceAtom], *, project_id: str = ""
) -> tuple[list[EvidenceAtom], list[EvidenceAtom]]:
    """Drop a QUOTED line in a reply when the same content already exists in the
    thread as authored (non-quoted) text or an earlier quoted copy.

    In a long back-and-forth every reply re-quotes the whole history, so the
    same sentence is emitted once per reply (the #010045 9,452-atom flood). The
    authored original is always kept; only its redundant quoted echoes are
    diverted. Returns ``(kept, dropped)``; the compiler routes ``dropped`` into
    the suppression ledger, so nothing is truly lost.

    Guarantees:
    * Never drops an authored (non-quoted) atom.
    * Never drops a quoted line whose content is unique to the thread (e.g. a
      quote of an external email not otherwise present).
    * Per-thread scoped — a quote only collapses against its OWN conversation,
      never across deals or unrelated threads.
    * Order-preserving for the kept list.
    """
    def _thread_of(atom: EvidenceAtom):
        v = atom.value if isinstance(atom.value, dict) else {}
        et = v.get("email_thread")
        return et if isinstance(et, dict) else None

    def _is_quoted(atom: EvidenceAtom) -> bool:
        v = atom.value if isinstance(atom.value, dict) else {}
        return bool(v.get("quoted"))

    # 1) Per-thread set of AUTHORED content keys (the originals we must keep and
    # that make a quoted echo redundant).
    authored_keys: dict[str, set[str]] = {}
    # The atom each key, header or message names: the survivor a dropped
    # echo is folded into.
    survivor_of: dict[tuple, EvidenceAtom] = {}
    for atom in atoms:
        et = _thread_of(atom)
        if et is None or _is_quoted(atom):
            continue
        key = _norm_key(atom)
        if len(key) >= _MIN_DEDUP_LEN:
            authored_keys.setdefault(et["thread_id"], set()).add(key)
            survivor_of.setdefault(("k", et["thread_id"], key), atom)

    # 1b) The messages that exist in the thread as their OWN email, by sender
    # and the minute they were sent. A quoted "From: X | Sent: Y" routing
    # atom exists so attribution survives when the original is missing; when
    # the original is right here it is pure repetition. Live 010289: 13 of 51
    # atoms were quoted headers of messages the deal already held.
    originals: dict[str, set[tuple[str, str]]] = {}
    for atom in atoms:
        et = _thread_of(atom)
        v = atom.value if isinstance(atom.value, dict) else {}
        if et is None or v.get("kind") != "email_header":
            continue
        addr = _address(str(v.get("from") or ""))
        for stamp in _minute_stamps_around(str(v.get("date") or "")) if addr else ():
            originals.setdefault(et["thread_id"], set()).add((addr, stamp))
            survivor_of.setdefault(("h", (addr, stamp)), atom)

    # 1c) The same, deal-wide, keyed by WHO wrote the line and WHEN. A reply
    # filed under another thread (a new subject, no In-Reply-To) still quotes
    # a message the deal holds as its own file; those lines are that file's,
    # not the reply's (live 010003: 20 lines of earlier mail sat under three
    # later emails). Matching the author and the minute keeps two people who
    # wrote the same sentence apart.
    authored_by: dict[str, set[tuple[str, str]]] = {}
    for atom in atoms:
        if _thread_of(atom) is None or _is_quoted(atom):
            continue
        key = _norm_key(atom)
        ident = _message_identity(atom) if len(key) >= _MIN_DEDUP_LEN else None
        if ident is not None:
            for stamp in ident[2]:
                authored_by.setdefault(key, set()).add((ident[1], stamp))
                survivor_of.setdefault(("m", key, ident[1], stamp), atom)
    all_originals: set[tuple[str, str]] = set().union(*originals.values()) if originals else set()

    def _held_message_line(atom: EvidenceAtom, key: str) -> EvidenceAtom | None:
        """The held message's own line this quoted atom repeats, if any."""
        who = authored_by.get(key)
        if not who:
            return None
        ident = _message_identity(atom)
        if ident is None:
            return None
        for st in ident[2]:
            if (ident[1], st) in who:
                return survivor_of.get(("m", key, ident[1], st))
        return None

    # 2) Walk atoms in thread order; drop a quoted atom whose key matches an
    # authored original OR an earlier-kept quoted copy in the same thread.
    #
    # "Earlier" is the thread's send order, not the order the files happened
    # to be listed in: the EARLIEST file that quotes a message owns it, and
    # every later quoted copy is the repetition. Walking in list order kept
    # the copy in whichever reply was listed first -- live 010003, sixteen
    # lines of earlier emails (and their signatures) sat on a late reply,
    # "You guys are the best! Thank you!" among them.
    def _file_rank(item: tuple[int, EvidenceAtom]) -> tuple[int, int]:
        i, atom = item
        et = _thread_of(atom) or {}
        try:
            ti = int(et.get("thread_index")) if et.get("thread_index") is not None else 10**6
        except (TypeError, ValueError):
            ti = 10**6
        return (ti, i)

    # The same words twice in ONE message are two lines (a "5:00-6:00 PM"
    # slot under each of three days), never a repeat: a quoted line is keyed
    # by its words AND how many times they already came up in its own
    # message of its own file, so the n-th copy in a reply folds onto the
    # n-th line of the earliest quote, and never onto a sibling line.
    occurrence: dict[int, int] = {}
    _occ_lines: dict[tuple, dict] = {}

    def _line_start(atom: EvidenceAtom) -> int | None:
        refs = getattr(atom, "source_refs", None) or []
        loc = (getattr(refs[0], "locator", None) or {}) if refs else {}
        line = loc.get("line_start") if isinstance(loc, dict) else None
        return line if isinstance(line, int) else None

    for atom in sorted(atoms, key=lambda a: (_line_start(a) is None, _line_start(a) or 0)):
        v = atom.value if isinstance(atom.value, dict) else {}
        if _thread_of(atom) is None or not v.get("quoted"):
            continue
        msg = (_thread_of(atom) or {}).get("message")
        mi = msg.get("index") if isinstance(msg, dict) else None
        if mi is None:
            mi = v.get("message_index")
        slot = (str(getattr(atom, "artifact_id", "") or ""), mi, _norm_key(atom),
                " ".join((getattr(atom, "raw_text", "") or "").split()).lower())
        line = _line_start(atom)
        # Two atoms off the SAME line are one line and share its count.
        seen_lines = _occ_lines.setdefault(slot, {})
        occurrence[id(atom)] = seen_lines.setdefault(line if line is not None else id(atom), len(seen_lines))

    seen_quoted: dict[str, set[tuple[str, int]]] = {}
    seen_headers: dict[str, set[tuple[str, str]]] = {}
    kept: list[EvidenceAtom] = []
    dropped: list[EvidenceAtom] = []
    for _i, atom in sorted(enumerate(atoms), key=_file_rank):
        et = _thread_of(atom)
        v = atom.value if isinstance(atom.value, dict) else {}
        if et is not None and v.get("kind") == "quoted_message_header":
            tid = et["thread_id"]
            key = (_address(str(v.get("sender") or "")), _minute_stamp(str(v.get("sent_at") or "")))
            if key[0] and key[1] and (key in originals.get(tid, ()) or key in seen_headers.get(tid, ())
                                      or key in all_originals):
                note_folded_into(atom, survivor_of.get(("h", key)) or survivor_of.get(("sh", tid, key)))
                dropped.append(atom)
                continue
            seen_headers.setdefault(tid, set()).add(key)
            survivor_of.setdefault(("sh", tid, key), atom)
            kept.append(atom)
            continue
        if et is None or not _is_quoted(atom):
            kept.append(atom)
            continue
        key = _norm_key(atom)
        if len(key) < _MIN_DEDUP_LEN:
            kept.append(atom)
            continue
        tid = et["thread_id"]
        held = None if key in authored_keys.get(tid, ()) else _held_message_line(atom, key)
        if key in authored_keys.get(tid, ()) or held is not None:
            # echo of an authored original (in this thread, or the same
            # author's message held as a file under another thread)
            note_folded_into(atom, survivor_of.get(("k", tid, key)) or held)
            dropped.append(atom)
            continue
        seen = seen_quoted.setdefault(tid, set())
        nth = (key, occurrence.get(id(atom), 0))
        if nth in seen:  # duplicate quoted copy across replies
            note_folded_into(atom, survivor_of.get(("q", tid, nth)))
            dropped.append(atom)
            continue
        seen.add(nth)
        survivor_of.setdefault(("q", tid, nth), atom)
        kept.append(atom)

    gone = {id(a) for a in dropped}
    kept = [a for a in atoms if id(a) not in gone]
    return kept, dropped


def _name_identity(author: str) -> str:
    """``name:patrick kelly`` for an author written with no address
    (Outlook's ``From: Patrick Kelly``), so its quoted lines still key on
    who wrote them."""
    name = re.sub(r"<[^>]*>", " ", author or "")
    name = " ".join(re.sub(r"[^a-z ]+", " ", name.lower()).split())
    if not name or name in {"unknown", "none"} or len(name.split()) > 5:
        return ""
    return "name:" + name


def _message_identity(atom: EvidenceAtom) -> tuple[str, str, set[str]] | None:
    """``(thread_id, author address, minute stamps)`` of the message a line
    belongs to, or ``None`` when the thread stamp cannot say."""
    v = atom.value if isinstance(atom.value, dict) else {}
    et = v.get("email_thread")
    if not isinstance(et, dict) or not et.get("thread_id"):
        return None
    msg = et.get("message") if isinstance(et.get("message"), dict) else {}
    quoted = bool(v.get("quoted"))
    author = str(msg.get("author") or v.get("author") or ("" if quoted else et.get("sender")) or "")
    sent = str(msg.get("sent_at") or v.get("authored_at") or ("" if quoted else et.get("date")) or "")
    addr = _address(author) or _name_identity(author)
    if not addr or not sent:
        return None
    if quoted:
        stamp = _minute_stamp(sent)
        return (str(et["thread_id"]), addr, {stamp} if stamp else set())
    return (str(et["thread_id"]), addr, _minute_stamps_around(sent))


#: Admission-reject reasons that mark a line as signature chrome (see
#: ``EmailParser._admission_reject_atom``): the same words in every message
#: their author signs.
_SIGNATURE_CHROME_REASONS = frozenset({"signature", "identity_only", "link_only", "quote_attribution"})


def _chrome_scope(atom: EvidenceAtom) -> str:
    v = atom.value if isinstance(atom.value, dict) else {}
    et = v.get("email_thread")
    if isinstance(et, dict) and et.get("thread_id"):
        return "t:" + str(et["thread_id"])
    return "a:" + str(getattr(atom, "artifact_id", "") or "")


def dedup_quoted_chatter(
    chatter: list[EvidenceAtom], *, context: list[EvidenceAtom] = ()
) -> tuple[list[EvidenceAtom], list[EvidenceAtom]]:
    """One atom per chatter line per MESSAGE, however often it is quoted.

    Greetings, sign-offs and cheers ("Hi Megan,", "Let's go!!") are held out
    of every head, so :func:`dedup_quoted_history` never saw them: each reply
    that quoted a message minted its "Hi Megan," again. They are too short to
    collapse on text alone -- Chase's "Hi Megan," is not Patrick's -- so the
    key is the text AND the message it belongs to (author address plus the
    minute it was sent, zone-shift tolerant). A quoted copy is dropped when the
    message's own email holds the line, or an earlier copy was kept; the kept
    atom is credited to the message that first authored it.
    """
    def _key(atom: EvidenceAtom) -> str:
        return _norm_key(atom)

    authored: set[tuple[str, str, str, str]] = set()
    # The atom each key names: the survivor a dropped copy is folded into.
    survivor_of: dict[tuple, EvidenceAtom] = {}
    for atom in list(context) + list(chatter):
        v = atom.value if isinstance(atom.value, dict) else {}
        if v.get("quoted"):
            continue
        ident = _message_identity(atom)
        k = _key(atom)
        if ident is None or not k:
            continue
        tid, addr, stamps = ident
        msg = (v.get("email_thread") or {}).get("message") or {}
        who = {addr}
        _nm = _name_identity(str(msg.get("author") or v.get("author") or (v.get("email_thread") or {}).get("sender") or ""))
        if _nm:
            who.add(_nm)
        for st in stamps:
            for w in who:
                authored.add((tid, w, st, k))
                survivor_of.setdefault(("a", tid, w, st, k), atom)

    # Signature chrome -- a name, a title, a phone, a separator rule -- is
    # the same line in every message its author signs, so a quoted copy
    # collapses on the thread and the words alone: kept once, on the message
    # that authored it when the deal holds that message. Live 010003: a
    # seller's "Patrick Kelly" / "770.769.7311" were 72 atoms each, quoted
    # under "From: Patrick Kelly" headers with no address (and under Gmail
    # "On ... wrote:" quotes with no author at all), which the per-message
    # key below could never resolve.
    authored_chrome: set[tuple[str, str]] = set()
    for atom in list(context) + list(chatter):
        v = atom.value if isinstance(atom.value, dict) else {}
        if v.get("quoted"):
            continue
        k = _key(atom)
        if k:
            authored_chrome.add((_chrome_scope(atom), k))
            survivor_of.setdefault(("c", _chrome_scope(atom), k), atom)
    seen_chrome: set[tuple[str, str]] = set()

    # The messages the deal holds as their OWN email, by thread, sender and
    # minute (as dedup_quoted_history keys its quoted headers). A quoted
    # message's "On <date> <name> wrote:" line opens a quote of one of them;
    # that email already heads its own section, so the line is repetition.
    # Live 010003: each reply kept a one-line section of its quoted history.
    originals: set[tuple[str, str, str]] = set()
    for atom in context:
        v = atom.value if isinstance(atom.value, dict) else {}
        et = v.get("email_thread")
        if v.get("kind") != "email_header" or not isinstance(et, dict) or not et.get("thread_id"):
            continue
        addr = _address(str(v.get("from") or ""))
        for st in _minute_stamps_around(str(v.get("date") or "")) if addr else ():
            originals.add((str(et["thread_id"]), addr, st))
            survivor_of.setdefault(("h", str(et["thread_id"]), addr, st), atom)

    seen: set[tuple[str, str, str, str]] = set()
    kept: list[EvidenceAtom] = []
    dropped: list[EvidenceAtom] = []
    for atom in chatter:
        v = atom.value if isinstance(atom.value, dict) else {}
        if v.get("quoted") and str(v.get("reason") or "") == "quote_attribution":
            ident = _message_identity(atom)
            if ident is not None and any((ident[0], ident[1], st) in originals for st in ident[2]):
                note_folded_into(atom, next((survivor_of[("h", ident[0], ident[1], st)] for st in ident[2]
                                             if ("h", ident[0], ident[1], st) in survivor_of), None))
                dropped.append(atom)
                continue
        if v.get("quoted") and str(v.get("reason") or "") in _SIGNATURE_CHROME_REASONS:
            k = _key(atom)
            ck = (_chrome_scope(atom), k)
            if k and (ck in authored_chrome or ck in seen_chrome):
                note_folded_into(atom, survivor_of.get(("c",) + ck) or survivor_of.get(("sc",) + ck))
                dropped.append(atom)
                continue
            if k:
                seen_chrome.add(ck)
                survivor_of.setdefault(("sc",) + ck, atom)
            kept.append(atom)
            continue
        ident = _message_identity(atom) if v.get("quoted") else None
        k = _key(atom)
        if ident is None or not k or not ident[2]:
            kept.append(atom)
            continue
        tid, addr, stamps = ident
        st = next(iter(stamps))
        sig = (tid, addr, st, k)
        if sig in authored or sig in seen:
            note_folded_into(atom, survivor_of.get(("a",) + sig) or survivor_of.get(("s",) + sig))
            dropped.append(atom)
            continue
        seen.add(sig)
        survivor_of.setdefault(("s",) + sig, atom)
        kept.append(atom)
    return kept, dropped


#: Reject reasons that mark a line as an author's sign-off block: the same
#: words under every message that author writes.
_REPEATING_CHROME_REASONS = frozenset({"signature", "identity_only", "link_only", "footer"})


def mark_repeated_signature_copies(chatter: list[EvidenceAtom]) -> int:
    """An author's signature in a later email is a copy of the first one.

    A signature is a reject with its reason; the same block under each of an
    author's emails is the same line again (live 010003: Patrick Kelly's six
    signature lines under every email he sent, twelve rejects to label). The
    earliest email keeps the line; each later one keeps its own atom, flagged
    ``cross_doc_copy`` and pointing at the first through ``duplicate_of``, as
    every other line two documents share. Authored lines only -- quoted
    copies are dedup_quoted_chatter's. Returns how many were marked.
    """
    from app.core.cross_doc_copies import COPY_FLAG

    def _when(atom: EvidenceAtom) -> tuple[int, float, int]:
        et = (atom.value or {}).get("email_thread") if isinstance(atom.value, dict) else None
        et = et if isinstance(et, dict) else {}
        try:
            from email.utils import parsedate_to_datetime

            t = parsedate_to_datetime(str(et.get("date") or "")).timestamp()
            return (0, t, 0)
        except Exception:
            ti = et.get("thread_index")
            return (1, float(ti) if isinstance(ti, (int, float)) else 1e12, 0)

    first: dict[tuple[str, str], EvidenceAtom] = {}
    marked = 0
    for i, atom in sorted(enumerate(chatter), key=lambda p: (_when(p[1]), p[0])):
        v = atom.value if isinstance(atom.value, dict) else {}
        if v.get("quoted") or str(v.get("reason") or "") not in _REPEATING_CHROME_REASONS:
            continue
        et = v.get("email_thread") if isinstance(v.get("email_thread"), dict) else {}
        msg = et.get("message") if isinstance(et.get("message"), dict) else {}
        who = _address(str(msg.get("author") or v.get("author") or et.get("sender") or ""))
        k = _norm_key(atom)
        if not who or not k:
            continue
        canon = first.get((who, k))
        if canon is None:
            first[(who, k)] = atom
            continue
        if str(canon.artifact_id) == str(atom.artifact_id):
            continue
        flags = list(atom.review_flags or [])
        if COPY_FLAG in flags:
            continue
        atom.review_flags = flags + [COPY_FLAG]
        v["duplicate_of"] = {"atom_id": str(canon.id), "artifact_id": str(canon.artifact_id),
                             "stage": "repeated_signature"}
        atom.value = v
        marked += 1
    return marked


__all__ = ["thread_emails", "dedup_quoted_history", "dedup_quoted_chatter", "mark_repeated_signature_copies"]
