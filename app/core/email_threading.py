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

from typing import Any

from app.core.ids import stable_id
from app.core.schemas import EvidenceAtom

_GIST_MAX = 160


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

    for aid, meta in meta_by_artifact.items():
        refs: list[str] = []
        if meta.get("in_reply_to"):
            refs.append(str(meta["in_reply_to"]).strip())
        refs.extend(str(r).strip() for r in (meta.get("references") or []))
        for ref in refs:
            other = msgid_to_artifact.get(ref)
            if other and other != aid:
                uf.union(aid, other)

    # Subject fallback: union all messages sharing a non-empty normalised
    # subject. Within one deal compile this reliably reunites a back-and-forth
    # whose .eml exports stripped the References chain (common from HubSpot).
    subject_groups: dict[str, list[str]] = {}
    for aid, meta in meta_by_artifact.items():
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
            if pos > 0:
                prev_aid = ordered[pos - 1]
                prev_meta = meta_by_artifact[prev_aid]
                prev_sender = (prev_meta.get("sender") or "").strip()
                prev_gist = gist_by_artifact.get(prev_aid, "")
                replied_to = {
                    "sender": prev_sender,
                    "gist": prev_gist,
                    "date": (prev_meta.get("date_raw") or "").strip(),
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
                "date": (meta.get("date_raw") or "").strip(),
                "gist": gist_by_artifact.get(aid, ""),
            }
            if replied_to is not None:
                thread_block["replied_to"] = replied_to
            if context:
                thread_block["context"] = context

            # 5) Additive stamp on EVERY atom of this artifact. We touch only
            # value["email_thread"] — no id / type / text change → no drops.
            for atom in atoms_by_artifact.get(aid, []):
                if isinstance(atom.value, dict):
                    atom.value["email_thread"] = thread_block

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


__all__ = ["thread_emails"]
