"""Who brought each document into the deal, and which way it travelled.

A PM labelling atoms has to know whether a file is something WE sent (our
proposal, our SOW: our output, not evidence) or something THEY sent (the
customer's site list, the reseller's BOM: evidence). The file itself never
says; the message that carried it does.

``doc_origin`` is that answer, on every document and on every atom::

    {
      "direction": "purtera_outbound" | "customer_inbound" | "internal",
      "label": "Sent by PurTera (Patrick Kelly) to customer · Jun 9 email",
      "sender_name": "Patrick Kelly" | None,
      "sender_email": "patrick@purtera-it.com" | None,
      "sent_at": "2026-06-09T15:00:00Z" | None,
      "via": "email" | "note" | "upload",
      "source": how it was decided (see below),
      ...
    }

Sources, most trusted first:

* ``carried_by`` -- Purpulse stamps it on each manifest artifact from an exact
  HubSpot join (the file's id in a message's ``attachmentIds``). When the
  carrying email is itself in this envelope its own MIME From and forwarded
  chain refine the person: a customer document forwarded in by our own staff
  is the customer's.
* ``hubspot_note`` -- the note a file was attached to (note_attachments).
* the document is itself a message (email) or a note.
* ``delivered_by`` -- the lifecycle's timestamp match of a delivering message.
* nothing: uploaded internally. An unattributed file must not look attributed.

Direction is read from the SENDER's domain, never HubSpot's direction flag
(which describes the message's relationship to the deal record). Resellers
and partners (CDW, SHI...) are the other side of the table, so they read as
``customer_inbound``; deal_parties keeps the finer ``role_guess``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from app.core.deal_parties import address_of, domain_of
from app.core.internal_author import INTERNAL_EMAIL_DOMAINS

OUTBOUND = "purtera_outbound"
INBOUND = "customer_inbound"
INTERNAL = "internal"

#: Keys copied onto every atom. The label is what a reader sees; the rest is
#: what a model can learn from.
ATOM_KEYS = ("direction", "label", "sender_name", "sender_email", "sent_at", "via")


def is_internal_address(raw: Any) -> bool:
    dom = domain_of(str(raw or ""))
    return bool(dom) and any(dom == d or dom.endswith(f".{d}") for d in INTERNAL_EMAIL_DOMAINS)


def _name_in_header(raw: Any) -> str:
    """'Patrick Kelly <p@x.com>' -> 'Patrick Kelly'; a bare address -> ''."""
    s = str(raw or "").strip()
    if "<" not in s:
        return ""
    name = s.split("<", 1)[0].strip().strip('"').strip()
    return " ".join(name.split()) if name and "@" not in name else ""


def direction_of(sender: Any, recipients: list[Any] | None = None, *, fallback: str | None = None) -> str | None:
    """Which way a message went, from who sent it."""
    addr = address_of(str(sender or ""))
    if not addr:
        return fallback
    if not is_internal_address(addr):
        return INBOUND
    to = [address_of(str(r)) for r in (recipients or [])]
    to = [t for t in to if t]
    if to:
        return INTERNAL if all(is_internal_address(t) for t in to) else OUTBOUND
    return fallback or OUTBOUND


def company_of(email: Any) -> str:
    """'sarah@cdw.com' -> 'CDW'; 'bernie@sodexo.com' -> 'Sodexo'."""
    dom = domain_of(str(email or ""))
    if not dom:
        return ""
    head = dom.split(".")[0]
    return head.upper() if len(head) <= 4 else head[:1].upper() + head[1:]


def _short_date(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    dt = None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime

            dt = parsedate_to_datetime(s)
        except Exception:
            dt = None
    if dt is None:
        return ""
    return f"{dt.strftime('%b')} {dt.day}"


def label_for(origin: dict[str, Any]) -> str:
    direction = origin.get("direction")
    via = origin.get("via") or "upload"
    email = origin.get("sender_email") or ""
    who = origin.get("sender_name") or email
    when = _short_date(origin.get("sent_at"))
    tail = f"{when} {via}" if when else via
    if via == "upload" or not (who or direction):
        return "Uploaded internally"
    fwd = origin.get("forwarded_by")
    fwd_part = f" · forwarded by {fwd}" if fwd else ""
    if direction == OUTBOUND:
        return f"Sent by PurTera ({who}) to customer{fwd_part} · {tail}" if who else f"Sent by PurTera to customer · {tail}"
    if direction == INBOUND:
        org = company_of(email)
        inner = ", ".join(x for x in (who, org) if x and x != email) or who
        return f"Received from customer ({inner}){fwd_part} · {tail}"
    if via == "note":
        return f"Uploaded internally by {who} · {tail}" if who else f"Uploaded internally · {tail}"
    return f"Internal email ({who}) · {tail}" if who else f"Internal email · {tail}"


def _roster_names(envelope: dict[str, Any] | None) -> dict[str, str]:
    roster = envelope.get("deal_roster") if isinstance(envelope, dict) else None
    people = roster.get("people") if isinstance(roster, dict) else None
    out: dict[str, str] = {}
    for p in people or []:
        if isinstance(p, dict) and p.get("email") and p.get("name"):
            out[str(p["email"]).lower()] = str(p["name"])
    return out


def _person_name(header: Any, email: str, explicit: Any, names: dict[str, str]) -> str | None:
    return (
        _name_in_header(header)
        or (str(explicit).strip() if explicit else "")
        or names.get(email.lower(), "")
        or None
    )


def _first_delivered_ts(doc: dict[str, Any]) -> str | None:
    delivered = (doc.get("lifecycle") or {}).get("delivered") if isinstance(doc.get("lifecycle"), dict) else None
    for d in delivered or []:
        if isinstance(d, dict) and d.get("ts"):
            return str(d["ts"])
    return None


def origin_for(
    doc: dict[str, Any],
    by_filename: dict[str, dict[str, Any]],
    names: dict[str, str] | None = None,
) -> dict[str, Any]:
    names = names or {}
    cb = doc.get("carried_by") if isinstance(doc.get("carried_by"), dict) else None

    if cb and cb.get("kind") in ("email", "note"):
        via = str(cb.get("kind"))
        carrier = doc if cb.get("self") else by_filename.get(str(cb.get("filename") or ""))
        header = ""
        if via == "email" and isinstance(carrier, dict):
            header = str(((carrier.get("email_thread") or {}).get("sender")) or "")
        email = address_of(header) or address_of(str(cb.get("sender_email") or ""))
        recipients = list(cb.get("recipients") or [])
        out: dict[str, Any] = {
            "via": via,
            "sender_email": email or None,
            "sender_name": _person_name(header, email, cb.get("sender_name"), names),
            "sent_at": cb.get("sent_at") or (carrier or {}).get("authored_at"),
            "direction": (
                direction_of(email, recipients, fallback=cb.get("direction"))
                if via == "email" and email else cb.get("direction")
            ),
            "carrier": cb.get("filename"),
            "carrier_id": cb.get("id"),
            "source": "carried_by",
        }
        # A forward from our own staff of something a customer wrote is the
        # customer's: the originator of the chain is the sender that matters.
        origin = str((carrier or {}).get("originated_by") or "")
        o_addr = address_of(origin)
        if via == "email" and o_addr and o_addr != email and not is_internal_address(o_addr) and is_internal_address(email):
            out["forwarded_by"] = out["sender_name"] or email
            out["sender_email"] = o_addr
            out["sender_name"] = _person_name(origin, o_addr, None, names)
            out["direction"] = INBOUND
        out["label"] = label_for(out)
        return out

    note = doc.get("hubspot_note") if isinstance(doc.get("hubspot_note"), dict) else None
    if note:
        email = address_of(str(note.get("author_email") or ""))
        out = {
            "via": "note",
            "sender_email": email or None,
            "sender_name": (str(note.get("author") or "").strip() or None) if "@" not in str(note.get("author") or "") else names.get(email) or None,
            "sent_at": note.get("created_at"),
            "direction": INBOUND if email and not is_internal_address(email) else INTERNAL,
            "carrier_id": note.get("hubspot_note_id"),
            "source": "hubspot_note",
        }
        out["label"] = label_for(out)
        return out

    thread = doc.get("email_thread") if isinstance(doc.get("email_thread"), dict) else {}
    if str(doc.get("artifact_type") or "") == "email" or thread.get("sender"):
        header = str(thread.get("sender") or doc.get("sender_email") or "")
        email = address_of(header)
        if email:
            hs = str(doc.get("direction") or "")
            out = {
                "via": "email",
                "sender_email": email,
                "sender_name": _person_name(header, email, None, names),
                "sent_at": doc.get("authored_at") or thread.get("date"),
                "direction": direction_of(email, fallback=OUTBOUND if hs == "outbound" else INTERNAL if hs == "internal" else None),
                "source": "message",
            }
            out["label"] = label_for(out)
            return out

    na = doc.get("note_author") if isinstance(doc.get("note_author"), dict) else None
    if na and (na.get("email") or na.get("name")):
        email = address_of(str(na.get("email") or ""))
        out = {
            "via": "note",
            "sender_email": email or None,
            "sender_name": na.get("name") or names.get(email) or None,
            "sent_at": na.get("date") or doc.get("authored_at"),
            "direction": INBOUND if email and not is_internal_address(email) else INTERNAL,
            "source": "note",
        }
        out["label"] = label_for(out)
        return out

    delivered = address_of(str(doc.get("delivered_by") or ""))
    if delivered:
        out = {
            "via": "email",
            "sender_email": delivered,
            "sender_name": _person_name(doc.get("delivered_by"), delivered, None, names),
            "sent_at": _first_delivered_ts(doc),
            # An internal deliverer with unknown recipients may be a forward
            # among ourselves: only the other side is asserted.
            "direction": INBOUND if not is_internal_address(delivered) else INTERNAL,
            "source": str(doc.get("delivered_by_source") or "delivered_by"),
        }
        if doc.get("forwarded_by"):
            out["forwarded_by"] = str(doc.get("forwarded_by"))
        out["label"] = label_for(out)
        return out

    return {
        "via": "upload",
        "sender_email": None,
        "sender_name": None,
        "sent_at": None,
        "direction": INTERNAL,
        "source": "none",
        "label": "Uploaded internally",
    }


def annotate_doc_origin(documents: list[dict[str, Any]], envelope: dict[str, Any] | None = None) -> int:
    """Stamp ``doc_origin`` on every document and on each envelope atom.
    Returns how many documents were stamped. Never raises."""
    try:
        names = _roster_names(envelope)
        by_filename = {str(d.get("filename") or ""): d for d in documents or [] if isinstance(d, dict)}
        by_artifact: dict[str, dict[str, Any]] = {}
        n = 0
        for d in documents or []:
            if not isinstance(d, dict):
                continue
            d["doc_origin"] = origin_for(d, by_filename, names)
            by_artifact[str(d.get("artifact_id") or "")] = d["doc_origin"]
            n += 1
        for a in (envelope or {}).get("atoms") or []:
            if not isinstance(a, dict):
                continue
            o = by_artifact.get(str(a.get("artifact_id") or ""))
            if o is not None:
                a["doc_origin"] = {k: o.get(k) for k in ATOM_KEYS}
        return n
    except Exception:  # pragma: no cover - provenance must never fail a compile
        return 0


__all__ = ["annotate_doc_origin", "origin_for", "label_for", "direction_of", "OUTBOUND", "INBOUND", "INTERNAL"]
