"""Read the picture, not just the link to it.

``linked_pictures`` finds the drawing and says where it is. Nothing then looks
at it. On deal 010288 that produced exactly one atom for a wiring diagram --
"Diagram: https://nam13.safelinks..." -- 700 characters of Outlook wrapper
around a PNG, while the drawing itself carried a title block, a two-colour
supply legend, nineteen labelled components and a software compatibility list,
none of which reached the parse. The email said "the 'Installer Supplied
Components' are not accurate", and there was nothing in the deal for that
sentence to be about.

The vision path this service already has is PDF-shaped: ``pdf_image_vision``
wants an ``image_marker`` with a saved crop, a page index and a ``.pdf``
source. A picture linked from an email body has a URL and no bytes anywhere,
so it could never enter. This stage is the missing front half.

WHAT THE MODEL IS ASKED FOR, AND WHAT IT IS NOT. OCR reads the sheet first, so
every label the model can talk about is a line that is really printed on it --
it cannot invent a component and it cannot miss one. The model's whole job is
to say what ROLE each of those lines plays. It is never asked what colour
anything is: on this drawing colour assigns every part to a company, and asked
directly, the model put "Cat5e/6 cable (up to 6')" in the vendor's colour when
it is printed in the installer's. Colour is measured from the pixels instead
(see ``linked_picture_ink``), keyed off the legend in the drawing's own ink.

Design, matching the rest of the service:
  * Abstain-first. No endpoint, no fetch, no OCR, bad JSON, any error -> emit
    nothing. Byte-identical to today when the flag is off.
  * Additive. Never removes or rewrites an input atom; the only mutation is a
    receipt on the atom that carried the link, so a skip stays traceable.
  * Nothing printed is silently dropped. A label in a legend colour that the
    model did not account for is emitted anyway and flagged, because the cost
    of a missed line on a bill of materials is a part nobody buys.

A drawing is a statement by whoever drew it, so its atoms carry extractor
authority and land in review -- never as fact we authored.
"""
from __future__ import annotations

import base64
import ipaddress
import json
import logging
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable

from app.core.ids import stable_id
from app.core.normalizers import normalize_text
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)

logger = logging.getLogger(__name__)

VERSION = "linked_picture_vision_v1"

#: A drawing is worth one model call and a few seconds. These bound the damage
#: a malicious or merely enormous link can do to a compile.
MAX_BYTES = 8 * 1024 * 1024
MAX_PICTURES = 8
FETCH_TIMEOUT_S = 15
MAX_REDIRECTS = 3
#: Below this a "picture" is a tracking pixel or a spacer, not a drawing.
MIN_BYTES = 3000

_IMAGE_MIME = re.compile(r"^image/(png|jpe?g|gif|webp|bmp|tiff?)\b", re.I)
#: A line that ends this way is a heading for the lines under it, not a fact on
#: its own. "Supported Software Includes:" split from "ABC Ignite" gives two
#: atoms that each say nothing.
_OPENS_LIST = re.compile(r"[:–—-]\s*$")
_BULLET = re.compile("^[\\s•·▪‣*.‐‑‒–—―-]+")
#: The same marks where OCR leaves them BETWEEN the items of a list it read
#: as one line: "Supported Software Includes: . ABC Ignite * Peak Pro" is
#: the sheet's own software list with its bullets transcribed as punctuation.
_INLINE_BULLET = re.compile("\\s+[•·▪‣*.‐‑‒–—―-]\\s+")


def enabled() -> bool:
    return os.environ.get("SOWSMITH_LINKED_PICTURE_VISION", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def connections_enabled() -> bool:
    """Emit the model's reading of what connects to what. See ``statements``."""
    return os.environ.get("SOWSMITH_LINKED_PICTURE_TOPOLOGY", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# ── fetching something a stranger chose ─────────────────────────────
#
# The URL comes out of a document a sender controls, and this runs server-side
# inside our network. Everything below exists so that a link in an email cannot
# make the parser reach somewhere we would not go on purpose.


class UnsafeURL(Exception):
    """The link points somewhere a document is not allowed to send us."""


def _assert_public(url: str) -> str:
    """https, a real hostname, and an address that is not on our own network."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() != "https":
        raise UnsafeURL(f"not https: {parts.scheme!r}")
    host = parts.hostname or ""
    if not host:
        raise UnsafeURL("no host")
    try:
        infos = socket.getaddrinfo(host, parts.port or 443, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise UnsafeURL(f"will not resolve: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        # Loopback, link-local, private ranges, carrier NAT and the cloud
        # metadata endpoint all live behind one of these flags.
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
            raise UnsafeURL(f"resolves to a non-public address ({ip})")
    return url


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects by hand so every hop is checked, not just the first."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def fetch_picture(url: str) -> tuple[bytes, str] | None:
    """Return (bytes, mime) for a public image URL, or None. Never raises."""
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        for _hop in range(MAX_REDIRECTS + 1):
            _assert_public(url)
            req = urllib.request.Request(url, headers={
                "User-Agent": "parser-os/1.0 (+deal artifact reader)",
                "Accept": "image/*",
            })
            try:
                resp = opener.open(req, timeout=FETCH_TIMEOUT_S)
            except urllib.error.HTTPError as exc:
                if exc.code in (301, 302, 303, 307, 308):
                    url = urllib.parse.urljoin(url, exc.headers.get("Location") or "")
                    continue
                raise
            with resp:
                mime = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
                if not _IMAGE_MIME.match(mime):
                    logger.info("linked_picture_vision: %s is %r, not an image", url, mime)
                    return None
                declared = resp.headers.get("Content-Length")
                if declared and int(declared) > MAX_BYTES:
                    logger.info("linked_picture_vision: %s is %s bytes, too big", url, declared)
                    return None
                # Read one byte past the cap so an undeclared oversize body is
                # caught rather than streamed into memory unbounded.
                body = resp.read(MAX_BYTES + 1)
            if len(body) > MAX_BYTES or len(body) < MIN_BYTES:
                return None
            return body, mime.lower()
        logger.info("linked_picture_vision: too many redirects for %s", url)
    except UnsafeURL as exc:
        logger.warning("linked_picture_vision: refusing %s -- %s", url, exc)
    except Exception as exc:  # noqa: BLE001 - a bad link must not cost the parse
        logger.info("linked_picture_vision: could not fetch %s -- %s", url, exc)
    return None


# ── asking the model what each printed line IS ──────────────────────

PROMPT = """Below are the text lines OCR read off an engineering or installation drawing
attached to a sales deal, numbered in reading order. Say what role each line plays.

Return ONLY a JSON object:

{
  "is_drawing": true|false,     // false if this is a logo, photo, signature or screenshot
  "vendor": "",                 // whose drawing it is, if a line says so
  "drawing_ref": "",            // the drawing/sheet number and revision, verbatim ("" if none)
  "roles": {"0": "role", ...},  // EVERY line index -> exactly one role, see below
  "connections": [              // only where the drawn line is unambiguous
    {"from": "", "to": "", "via": ""}
  ]
}

The roles:
  "title"      part of the title block -- what this sheet depicts
  "legend"     a key entry: it defines what a colour or symbol on the sheet MEANS
               ("Huzzard supplied Components", "by others", "existing")
  "component"  a label naming a physical part, cable or device drawn on the sheet
  "note"       a note, callout, list heading or list item printed on the sheet
  "brand"      a company name, logo text, web address or sheet footer
  "decoration" an arrow, stray character or anything carrying no meaning

Rules:
  * Every index in the list below must appear exactly once in "roles".
  * Use the line's own words for "from"/"to"/"via" in connections.
  * Do NOT report colours. You are not being asked what colour anything is.
  * Leave a field "" and a list [] rather than guessing at a revision or a vendor.

LINES:
"""


def ask_roles(image_b64: str, lines: list[dict[str, Any]], *,
              mime: str = "image/png") -> dict[str, Any] | None:
    """One vision call: the picture plus the OCR lines. Returns the parsed
    object, or None on anything unusable."""
    from app.core.llm_client import complete_vision

    listing = "\n".join(f"{i}: {ln['content']}" for i, ln in enumerate(lines))
    try:
        reply = complete_vision(PROMPT + listing, image_b64, mime=mime, max_tokens=3000)
    except Exception as exc:  # noqa: BLE001
        logger.info("linked_picture_vision: vision call failed -- %s", exc)
        return None
    if not reply:
        return None
    text = reply.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)  # models fence anyway
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        got = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return got if isinstance(got, dict) else None


# ── turning a read sheet into statements ────────────────────────────


def _clean(v: Any) -> str:
    return " ".join(str(v or "").split())


#: A label has to be a word. OCR reads arrowheads and tick marks as "1", "E"
#: and "-", and each one would otherwise become a line item on a bill of
#: materials.
def _is_a_label(text: str) -> bool:
    stripped = _BULLET.sub("", _clean(text))
    letters = sum(c.isalpha() for c in stripped)
    return letters >= 2 and len(stripped) >= 3


def _label_text(text: str) -> str:
    return _BULLET.sub("", _clean(text)).strip()


def _note_text(text: str) -> str:
    """A note with its list bullets turned back into a list."""
    out = _INLINE_BULLET.sub(", ", _label_text(text))
    return re.sub(r":\s*,\s*", ": ", out).strip()


def _box(polygon: list[float]) -> tuple[float, float, float, float]:
    xs, ys = polygon[0::2], polygon[1::2]
    return min(xs), min(ys), max(xs), max(ys)


#: What a wrapped label looks like, measured on 010288's sheet: rows of one
#: label sit 0.00-0.18 of a line-height apart, separate callouts 0.55 and up.
#: The gap between those two populations is the whole rule.
WRAP_MAX_GAP_RATIO = 0.35
WRAP_MIN_OVERLAP = 0.6


def merge_wrapped_labels(lines: list[dict[str, Any]],
                         hues: list[int | None]) -> list[list[int]]:
    """Group OCR lines that are one label wrapped onto two rows.

    "Serial / Com Cable" is printed on two rows and OCR returns two lines, so
    without this the drawing yields an atom saying it shows "Serial /" and
    another saying it shows "Com Cable" -- a line torn off its label, which is
    worth nothing to anybody. Two rows are one label when they are the same
    colour, sit within a fraction of a line-height of each other, and overlap
    horizontally. "Couplers" and "Adapter Cable" are also stacked and also
    orange, and they stay apart because the sheet puts half a line between
    them.
    """
    groups: list[list[int]] = []
    used: set[int] = set()
    for i, line in enumerate(lines):
        if i in used:
            continue
        group = [i]
        used.add(i)
        x0, _y0, x1, y1 = _box(line["polygon"])
        height = max(1.0, y1 - _y0)
        for j in range(i + 1, len(lines)):
            if j in used or hues[j] != hues[i]:
                continue
            nx0, ny0, nx1, ny1 = _box(lines[j]["polygon"])
            gap = ny0 - y1
            if gap < 0 or gap > height * WRAP_MAX_GAP_RATIO:
                continue
            overlap = min(x1, nx1) - max(x0, nx0)
            if overlap <= 0 or overlap < WRAP_MIN_OVERLAP * min(x1 - x0, nx1 - nx0):
                continue
            group.append(j)
            used.add(j)
            x0, x1, y1 = min(x0, nx0), max(x1, nx1), ny1
        groups.append(group)
    return groups


def _joined_notes(notes: list[str]) -> list[str]:
    """Fold a list heading into the items under it.

    "Supported Software Includes:" / "ABC Ignite" / "Peak Pro" is one fact
    written on three lines. Split, each line is worthless: a heading with
    nothing under it and two product names with nothing above them.
    """
    out: list[str] = []
    pending: list[str] = []
    for raw in notes:
        text = _clean(raw)
        if not text:
            continue
        if pending and _BULLET.match(raw):
            pending.append(_BULLET.sub("", text))
            continue
        if pending:
            out.append(pending[0] + " " + ", ".join(pending[1:]) if len(pending) > 1 else pending[0])
            pending = []
        if _OPENS_LIST.search(text):
            pending = [text]
        else:
            out.append(text)
    if pending:
        out.append(pending[0] + " " + ", ".join(pending[1:]) if len(pending) > 1 else pending[0])
    return out


def statements(read: dict[str, Any]) -> list[tuple[str, str, AtomType]]:
    """(fact_kind, sentence, type) for everything the drawing states.

    A legend entry is a statement about how the sheet must be READ, so it is
    kept separate from the components it governs: when a legend turns out to be
    wrong -- as this one did -- the atom that is wrong has to be nameable on
    its own.
    """
    out: list[tuple[str, str, AtomType]] = []
    title = _clean(read.get("title"))
    ref, vendor = _clean(read.get("drawing_ref")), _clean(read.get("vendor"))
    if title or ref:
        bits = [b for b in (title, ref and f"drawing {ref}", vendor and f"by {vendor}") if b]
        out.append(("title", "The drawing is " + ", ".join(bits) + ".", AtomType.deal_metadata))

    for meaning in read.get("legend") or []:
        out.append(("legend",
                    f"The drawing's legend has an entry for {meaning}.",
                    AtomType.deal_metadata))

    for comp in read.get("components") or []:
        label, means = _clean(comp.get("label")), _clean(comp.get("means"))
        if not label:
            continue
        if means:
            sentence = f"The drawing shows {label}, in the colour its legend calls {means}."
        else:
            sentence = (f"The drawing shows {label}, in no colour the legend defines -- "
                        f"the sheet does not say who supplies it.")
        out.append(("component", sentence, AtomType.bom_line))

    for text in read.get("notes") or []:
        out.append(("note", f"Printed on the drawing: {_clean(text)}", AtomType.scope_item))

    # TOPOLOGY IS OFF BY DEFAULT, and this is the one place the stage declines
    # to say what it saw. Every other fact here is anchored: the words come
    # from OCR and the colours from the pixels. Which line on the sheet runs to
    # which is neither -- it is the model tracing wires, and on 010288 it put
    # the barcode reader straight into the power supply, which is not what the
    # drawing shows. A wrong connection reads exactly like a right one, so the
    # default is to emit nothing rather than something a PM would have to
    # re-derive from the picture anyway.
    if connections_enabled():
        for row in read.get("connections") or []:
            a, b = _clean(row.get("from")), _clean(row.get("to"))
            if not a or not b:
                continue
            via = _clean(row.get("via"))
            out.append(("connection",
                        f"The drawing connects {a} to {b}" + (f" via {via}" if via else "") + ".",
                        AtomType.scope_item))
    return out


def read_picture(body: bytes, mime: str) -> dict[str, Any] | None:
    """OCR the sheet, ask the model what each line is, measure the colours.

    Returns a normalised reading, or None when anything essential is missing.
    """
    from io import BytesIO

    from app.core import linked_picture_ink as inkmod
    from app.core.doc_intel_ocr import read_lines_with_polygons

    lines = read_lines_with_polygons(body)
    if not lines:
        logger.info("linked_picture_vision: no OCR lines; abstaining")
        return None
    try:
        from PIL import Image
        image = Image.open(BytesIO(body)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        logger.info("linked_picture_vision: cannot open image -- %s", exc)
        return None

    got = ask_roles(base64.b64encode(body).decode(), lines, mime=mime)
    if not got or not got.get("is_drawing"):
        return {"is_drawing": bool(got and got.get("is_drawing"))}

    roles = got.get("roles") or {}
    def role_of(i: int) -> str:
        return str(roles.get(str(i)) or roles.get(i) or "").strip().lower()

    legend_idx = {i for i in range(len(lines)) if role_of(i) == "legend"}
    legend_lines = [lines[i] for i in sorted(legend_idx)]
    references = inkmod.legend_reference_hues(image, legend_lines or lines)

    # OCR reads arrowheads and leader dashes as "1", "E" and "-". Drop them
    # before merging, or a stray dash under a label becomes part of its name.
    keep = [i for i, ln in enumerate(lines) if _is_a_label(ln["content"])]
    kept = [lines[i] for i in keep]
    hues = [inkmod.ink_hue(image, ln["polygon"]) for ln in kept]

    components: list[dict[str, str]] = []
    notes: list[str] = []
    for group in merge_wrapped_labels(kept, hues):
        head = keep[group[0]]
        if head in legend_idx:
            continue
        text = _label_text(" ".join(kept[g]["content"] for g in group))
        if not text:
            continue
        # WHAT MAKES A COMPONENT IS THE INK, NOT THE MODEL. A label printed in
        # a legend colour is a part the sheet assigns to a company; that is
        # what the legend is for. Asked instead to sort lines into roles, the
        # model moved "Gender Changers" and "PC with Access Control Software"
        # from component to note between two runs of the same image. The
        # measurement does not drift.
        hit = inkmod.classify(image, kept[group[0]]["polygon"], references)
        if hit is not None:
            components.append({"label": text, "means": hit[1]})
        elif role_of(head) == "note":
            notes.append(_note_text(text))

    title = " ".join(ln["content"] for i, ln in enumerate(lines) if role_of(i) == "title")
    return {
        "is_drawing": True,
        "title": title,
        "drawing_ref": got.get("drawing_ref") or "",
        "vendor": got.get("vendor") or "",
        "legend": sorted(set(references.values())),
        "legend_hues": references,
        "components": components,
        "notes": _joined_notes(notes),
        "connections": got.get("connections") or [],
        "ocr_lines": len(lines),
    }


def _emit(*, source: Any, url: str, fact_kind: str, text: str,
          atom_type: AtomType, confidence: float) -> EvidenceAtom | None:
    text = (text or "").strip()
    if not text:
        return None
    artifact_id = getattr(source, "artifact_id", "") or ""
    refs = getattr(source, "source_refs", None) or []
    filename = (getattr(refs[0], "filename", "") if refs else "") or ""
    atom_id = stable_id("atm", artifact_id, VERSION, url, fact_kind, text[:80])
    src = SourceRef(
        id=stable_id("src", atom_id),
        artifact_id=artifact_id,
        artifact_type=ArtifactType.image,
        filename=filename,
        locator={"image_url": url, "extraction": VERSION, "fact_kind": fact_kind},
        extraction_method=VERSION,
        parser_version=VERSION,
    )
    return EvidenceAtom(
        id=atom_id,
        project_id=getattr(source, "project_id", "") or "",
        artifact_id=artifact_id,
        atom_type=atom_type,
        raw_text=text,
        normalized_text=normalize_text(text),
        value={
            "via": "linked_picture_vision",
            "fact_kind": fact_kind,
            "image_url": url,
            "media_type": "image",
            "source_atom_id": getattr(source, "id", ""),
        },
        entity_keys=[],
        source_refs=[src],
        receipts=[],
        # A model reading somebody else's drawing is an extractor, not a party.
        authority_class=AuthorityClass.machine_extractor,
        confidence=confidence,
        confidence_raw=confidence,
        calibrated_confidence=confidence,
        review_status=ReviewStatus.needs_review,
        review_flags=["linked_picture_vision", f"fact_kind:{fact_kind}"],
        parser_version=VERSION,
    )


#: A legend is printed large and the sheet is keyed to it; a component label is
#: read by OCR and coloured by measurement, so it is firmer than a model guess
#: but still a machine read; a connection is the model's reading of which line
#: goes where, which is the easiest thing on a drawing to get wrong.
_CONFIDENCE = {
    "title": 0.72, "legend": 0.72, "note": 0.66, "component": 0.64, "connection": 0.45,
}


def atoms_from_linked_pictures(atoms: Iterable[Any]) -> list[EvidenceAtom]:
    """Read every linked picture and return NEW atoms. Never raises."""
    if not enabled():
        return []
    pool = list(atoms or [])
    if not pool:
        return []

    out: list[EvidenceAtom] = []
    seen: set[str] = set()
    for source in pool:
        if len(seen) >= MAX_PICTURES:
            break
        try:
            value = getattr(source, "value", None)
            if not isinstance(value, dict):
                continue
            url = str(value.get("image_url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)

            got = fetch_picture(url)
            if not got:
                value["picture_read"] = "unreachable"
                continue
            body, mime = got
            read = read_picture(body, mime)
            if not read:
                value["picture_read"] = "unreadable"
                continue
            if not read.get("is_drawing"):
                # A logo or a signature image is a real answer: it says the
                # link was not evidence, so nobody goes looking for it again.
                value["picture_read"] = "not_a_drawing"
                continue

            made = 0
            for fact_kind, text, atom_type in statements(read):
                atom = _emit(source=source, url=url, fact_kind=fact_kind, text=text,
                             atom_type=atom_type,
                             confidence=_CONFIDENCE.get(fact_kind, 0.5))
                if atom is not None:
                    out.append(atom)
                    made += 1
            value["picture_read"] = "read"
            value["picture_atoms"] = made
        except Exception as exc:  # noqa: BLE001
            logger.info("linked_picture_vision: %s", exc)
            continue
    if out:
        logger.info("linked_picture_vision: %d atoms from %d picture(s)", len(out), len(seen))
    return out


__all__ = [
    "atoms_from_linked_pictures",
    "ask_roles",
    "enabled",
    "fetch_picture",
    "read_picture",
    "statements",
    "UnsafeURL",
]
