"""The labeling page's deal export, as the training file ``loop.load_deal`` reads.

The page export (``dealId``, ``documents``, ``atoms``, ``labels``,
``judgments``, ``myLinks``) is what every labeling thread pulls and checks.
The trainer needs the same rows in its own contract: every atom with its
``label_key``, the time its source entered the deal, its document kind,
section and sender, plus the person's links under ``links``.

Every line's sender (``speaker_side`` ours / theirs / partner / vendor,
``speaker_role``, ``speaker_org``), its document's ``doc_direction`` and the
deal's ``stage_at`` when it arrived come from ``c3_record.sender``, the rules
the parser's compile also reads them by: the line's ``saidBy``, else the
message it was written in, else a call transcript's speaker, else its
document's sender (an email's From, a note's author, and for an attachment
the email or note that carried it, the page's ``origin``).

When a line entered the deal is what the foresight mask, the direction of
``contradicts`` and the hindsight targets read, so it is never left blank:

1. the atom's own source time (``docOrigin.sent_at``: when the email, note or
   upload reached the deal);
2. else its document's ``origin.sent_at``;
3. else, for a meeting, note or email, the document's ``authoredAt`` (a
   meeting enters the deal when it happens);
4. else the time of the document before it in arrival order. A file's
   ``authoredAt`` is when someone wrote it, often years earlier, and says
   nothing about when the deal saw it;
5. else (a line with no document, such as a PM's answer typed on the
   labeling page) the latest time in the deal: it was written after
   everything the deal had seen.

Only the person's rows travel (``labels``, ``judgments``, ``myLinks``); the
assistant's mirrored proposals never do. No source text is written anywhere
but the output file, which belongs outside the repository.

    python -m ml.c3.export --page /outside/export043.json --deal-id 010043 \\
        --output /outside/train/010043.json [--deal-answers answers.json]

``--manifest`` appends the output's sha256 to a split manifest so a run can
check it trained the files that were frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

#: Document kinds whose ``authoredAt`` is the moment they entered the deal.
_EVENT_KINDS = ("meeting", "note", "email")


def _doc_times(documents: list[dict[str, Any]]) -> dict[str, str]:
    """Arrival time per artifact, filled forward in the export's arrival order
    (the deal record's rules, ``record.document_times``)."""
    from .record import document_times

    return {k: str(v) for k, v in document_times(
        {"doc_id": str(d.get("artifactId") or ""), "kind": d.get("kind"),
         "sent_at": (d.get("origin") if isinstance(d.get("origin"), dict) else {}).get("sent_at"),
         "authored_at": d.get("authoredAt")}
        for d in documents).items()}


def _sender_inputs(atom: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any]:
    """What ``c3_record.sender`` reads off one line: its own speaker, the
    message it was written in, and a call transcript's speaker (the line's
    section on the page)."""
    out: dict[str, Any] = {}
    if isinstance(atom.get("saidBy"), dict):
        out["said_by"] = atom["saidBy"]
    if isinstance(atom.get("saidTo"), list):
        out["said_to"] = [p for p in atom["saidTo"] if isinstance(p, dict)]
    msg = atom.get("message") if isinstance(atom.get("message"), dict) else {}
    if msg.get("author"):
        out["message_author"] = str(msg["author"])
    if str(doc.get("artifactType") or "") == "transcript":
        name = _section(atom.get("section"))
        if name:
            out["speaker_name"] = name
    return out


def _origin(doc: dict[str, Any]) -> dict[str, Any]:
    """The page's ``origin`` (a file's is its carrier's); a note nobody is
    recorded as sending takes its author, when the author has an address."""
    o = dict(doc.get("origin") if isinstance(doc.get("origin"), dict) else {})
    if not o.get("sender_email") and str(doc.get("kind") or "") == "note":
        from c3_record.sender import address

        email = address(doc.get("authorEmail")) or address(doc.get("author"))
        if email:
            o.update(sender_email=email, sender_name=str(doc.get("author") or "") or None, via="note")
    return o


def _record_documents(documents: list[dict[str, Any]], doc_time: dict[str, str]) -> list[dict[str, Any]]:
    """The documents as ``c3_record.annotate`` reads them: kind, arrival and
    who brought each."""
    return [{"doc_id": str(d.get("artifactId") or ""),
             "kind": str(d.get("docType") or d.get("kind") or "").strip().lower(),
             "arrival": doc_time.get(str(d.get("artifactId") or "")), "origin": _origin(d),
             "role": str(d.get("docRole") or "")}
            for d in documents]


def _section(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " > ".join(str(v) for v in value if str(v).strip())
    return str(value or "")


def is_page_export(payload: Any) -> bool:
    """The labeling page's shape: camelCase deal id and the person's links apart."""
    return isinstance(payload, dict) and "myLinks" in payload and isinstance(payload.get("atoms"), list)


def from_page_export(payload: dict[str, Any], deal_id: str, *, company: str = "purtera",
                     deal_answers: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """One training file: ``{deal_id, company, atoms, labels, judgments, links, deal_answers}``."""
    if not is_page_export(payload):
        raise ValueError("Not a labeling page export (needs atoms and myLinks)")
    if not str(deal_id or "").strip():
        raise ValueError("A training file needs a deal_id")
    from c3_record import table_geometry

    documents = [d for d in payload.get("documents") or [] if isinstance(d, dict)]
    doc_time = _doc_times(documents)
    kinds = {str(d.get("artifactId")): str(d.get("kind") or "") for d in documents}
    doc_of = {str(d.get("artifactId")): d for d in documents}
    by_file = {str(d.get("filename")): str(d.get("artifactId")) for d in documents if d.get("filename")}
    atoms: list[dict[str, Any]] = []
    keys: set[str] = set()
    last_in: dict[tuple[str, str], int] = {}
    for i, a in enumerate(payload["atoms"]):
        key = str(a.get("labelKey") or "")
        if not key:
            raise ValueError(f"Atom {a.get('atomId')} has no labelKey")
        if key in keys:
            raise ValueError(f"Two atoms share labelKey {key}")
        keys.add(key)
        art = str(a.get("artifactId") or "")
        origin = a.get("docOrigin") if isinstance(a.get("docOrigin"), dict) else {}
        when = origin.get("sent_at") or doc_time.get(art) or ""
        section = _section(a.get("section"))
        atoms.append({
            "label_key": key, "atom_id": str(a.get("atomId") or ""), "text": str(a.get("text") or ""),
            "entered_at": str(when), "doc_id": art,
            "doc_kind": str(a.get("docType") or kinds.get(art) or "").strip().lower(),
            "section": section, "order": i, "speaker_role": "", "speaker_side": "",
            **_sender_inputs(a, doc_of.get(art, {})),
            # Where the line sits in its table: the model input ``table_row``
            # (c3_record.table_geometry), the same rule the parser's compile uses.
            **table_geometry(a.get("tableRef")),
        })
        last_in[(art, section)] = i
        last_in[(art, "")] = i
    latest = max((a["entered_at"] for a in atoms if a["entered_at"]), default=None)
    for a in atoms:
        if not a["entered_at"]:
            if latest is None:
                raise ValueError(f"No arrival time for atom {a['atom_id']}")
            a["entered_at"] = latest
    # Who sent each line and where the deal stood when it arrived: the same
    # rules the parser's compile uses (c3_record.sender).
    from c3_record import annotate

    annotate(atoms, _record_documents(documents, doc_time))
    labels = []
    for lb in payload.get("labels") or []:
        row = dict(lb)
        if str(row.get("origin") or "").lower() == "labeler" and row.get("label_key") not in keys:
            # A hand-added (Missed) line has no atom: place it in its own
            # document, after the last parser line of its section.
            art = by_file.get(str(row.get("filename") or ""), "")
            section = _section(row.get("section"))
            at = last_in.get((art, section), last_in.get((art, ""), len(atoms) - 1))
            row.setdefault("doc_id", art)
            row.setdefault("doc_kind", str(row.get("doc_type") or kinds.get(art) or "").lower())
            row.setdefault("entered_at", atoms[at]["entered_at"] if atoms else doc_time.get(art))
            row.setdefault("order", at + 0.5)
            row["section"] = section
        labels.append(row)
    return {
        "deal_id": str(deal_id), "company": company,
        "source": {"page_deal_uuid": payload.get("dealId"), "compile_id": payload.get("compileId"),
                   "generated_at": payload.get("generatedAt")},
        "atoms": atoms, "labels": labels,
        "judgments": list(payload.get("judgments") or []),
        "links": list(payload.get("myLinks") or []),
        "deal_answers": list(deal_answers or []),
    }


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--page", required=True, help="labeling page export JSON")
    p.add_argument("--deal-id", required=True, help="the deal number, e.g. 010043")
    p.add_argument("--output", required=True)
    p.add_argument("--deal-answers", help="JSON list of the deal's saved Deal answers")
    p.add_argument("--company", default="purtera")
    p.add_argument("--manifest", help="split manifest to record the output's sha256 in")
    p.add_argument("--split", choices=("train", "validation", "holdout"))
    args = p.parse_args()
    payload = json.loads(Path(args.page).read_text(encoding="utf-8"))
    answers = json.loads(Path(args.deal_answers).read_text(encoding="utf-8")) if args.deal_answers else []
    if isinstance(answers, dict):
        answers = [answers]
    doc = from_page_export(payload, args.deal_id, company=args.company, deal_answers=answers)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    tmp.replace(out)
    digest = sha256(out)
    print(json.dumps({"deal_id": args.deal_id, "output": str(out), "sha256": digest,
                      "atoms": len(doc["atoms"]), "labels": len(doc["labels"]),
                      "judgments": len(doc["judgments"]), "links": len(doc["links"]),
                      "deal_answers": len(doc["deal_answers"])}))
    if args.manifest:
        if not args.split:
            raise SystemExit("--manifest needs --split")
        mp = Path(args.manifest)
        manifest = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
        for name in ("train", "validation", "holdout"):
            manifest.setdefault(name, [])
            manifest[name] = [e for e in manifest[name] if e.get("deal_id") != args.deal_id]
        manifest[args.split].append({"deal_id": args.deal_id, "path": str(out), "snapshot_sha256": digest,
                                     "compile_id": payload.get("compileId")})
        mp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
