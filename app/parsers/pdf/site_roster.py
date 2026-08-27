"""Recovering a site roster a PDF renders as a table or as plain text.

Two independent routes to the same answer, because a roster arrives either as ruled cells fitz can find or as text that merely looks tabular. Both produce ``physical_site`` rows.
"""

from __future__ import annotations

from app.core.schemas import AtomType
from app.core.schemas import AuthorityClass
from app.core.schemas import EvidenceAtom
from app.parsers.pdf._shared import _make_atom
from app.parsers.pdf._shared import _table_rows_repaired
from pathlib import Path
from typing import Any
import re


def _fitz_site_roster_fallback(
    *,
    pdf_path: Path,
    project_id: str,
    artifact_id: str,
    parser_version: str,
    already_emitted: set[str | None] | None = None,
) -> list[EvidenceAtom]:
    """Use ``fitz.find_tables()`` to catch site rosters the structured
    pipeline missed.

    Returns a list of structured ``physical_site`` entity atoms. Never
    raises — on any error (fitz unavailable, PDF unreadable, no tables)
    returns an empty list.
    """
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:
        return []
    try:
        from app.parsers.site_roster_extractor import (
            extract_site_roster,
            looks_like_site_roster,
        )
    except Exception:
        return []

    already_emitted = already_emitted or set()
    out: list[EvidenceAtom] = []
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return []
    try:
        # Pull the document-level surrounding text once so the
        # extractor can spot ``kind=physical_site`` declarations.
        page_texts: list[str] = []
        try:
            for p in doc:
                try:
                    page_texts.append(p.get_text() or "")
                except Exception:
                    continue
        except Exception:
            page_texts = []
        document_text = "\n".join(page_texts)

        for page_index, page in enumerate(doc):
            try:
                tables_finder = page.find_tables()
            except Exception:
                continue
            tables = list(getattr(tables_finder, "tables", []) or [])
            if not tables:
                continue
            for table_index, table in enumerate(tables):
                try:
                    extracted = _table_rows_repaired(page, table)
                except Exception:
                    continue
                if not extracted or len(extracted) < 2:
                    continue
                header = [(c or "") for c in extracted[0]]
                body = extracted[1:]
                rows: list[dict[str, Any]] = []
                for r in body:
                    if not r:
                        continue
                    rows.append({
                        header[i] if i < len(header) and header[i] else f"col_{i}": (
                            # Collapse internal whitespace in cell values so a
                            # word that wrapped across two display lines (e.g.
                            # "ATL-WEST-0\n2") renders as a single token.
                            " ".join((c or "").split())
                        )
                        for i, c in enumerate(r)
                    })
                if not rows:
                    continue
                # Build column header list, then route through
                # site_roster_extractor.
                columns = [
                    header[i] if i < len(header) and header[i] else f"col_{i}"
                    for i in range(len(header) if header else (len(rows[0]) if rows else 0))
                ]
                try:
                    is_roster = looks_like_site_roster(
                        columns=columns, rows=rows, surrounding_text=document_text
                    )
                except Exception:
                    is_roster = False
                if not is_roster:
                    # v53.5 BACKUP: route through the universal column
                    # schema registry. When site_roster_extractor's gate
                    # rejects the table (false negative — e.g. column
                    # headers don't match the canonical four-pattern set
                    # but the data IS a site roster), the schema registry
                    # may still recognize "Site ID + Facility name +
                    # Street address" and emit physical_site atoms.
                    try:
                        from app.core.table_schema_registry import (
                            identify_schema, emit_atoms_for_schema,
                        )
                        sn = identify_schema(columns)
                        if sn == "site_roster":
                            schema_atoms = []
                            for ri, _row in enumerate(rows):
                                row_vals = [
                                    _row.get(c, "") if isinstance(_row, dict)
                                    else (_row[i] if i < len(_row) else "")
                                    for i, c in enumerate(columns)
                                ]
                                schema_atoms.extend(emit_atoms_for_schema(
                                    schema_name=sn,
                                    columns=columns,
                                    row=row_vals,
                                    row_idx=ri,
                                    table_idx=table_index,
                                    project_id=project_id,
                                    artifact_id=artifact_id,
                                    filename=pdf_path.name,
                                    parser_version=parser_version,
                                ))
                            for sa in schema_atoms:
                                # Skip if already emitted by a structural path
                                _sid = (sa.value or {}).get("id") if sa.value else None
                                if _sid and _sid in already_emitted:
                                    continue
                                if _sid:
                                    already_emitted.add(_sid)
                                out.append(sa)
                    except Exception:
                        pass
                    continue
                try:
                    roster_rows = extract_site_roster(
                        columns=columns, rows=rows, surrounding_text=document_text
                    )
                except Exception:
                    roster_rows = []
                # Bbox from fitz table -> base locator
                try:
                    bbox = table.bbox
                    locator_base = {
                        "page": int(page_index),
                        "block_kind": "table",
                        "bbox": list(bbox),
                        "extraction": "site_roster_fitz_fallback_v1",
                    }
                except Exception:
                    locator_base = {"page": int(page_index), "extraction": "site_roster_fitz_fallback_v1"}
                for site_row in roster_rows:
                    sid = (site_row.site_id or "").strip()
                    # Normalize whitespace inside the ID (PDF wrap
                    # artifacts: "ATL-WEST-0 2" -> "ATL-WEST-02")
                    if sid and " " in sid:
                        compact = re.sub(r"\s+", "", sid)
                        # Only collapse when the compact form still
                        # looks like a site ID — keeps "Building C"
                        # type values from getting smushed.
                        from app.parsers.site_roster_extractor import _SITE_ID_SHAPE_RE
                        if _SITE_ID_SHAPE_RE.match(compact):
                            sid = compact
                    if sid in already_emitted:
                        continue
                    already_emitted.add(sid)
                    canon_id = sid or site_row.facility_name or ""
                    if not canon_id:
                        continue
                    site_text = " | ".join(
                        f"{k}: {v}"
                        for k, v in [
                            ("site_id", sid or site_row.site_id),
                            ("facility", site_row.facility_name),
                            ("address", site_row.street_address),
                            ("mdf_idf", site_row.mdf_idf),
                            ("access", site_row.access_window),
                            ("escort", site_row.escort_owner),
                            ("contact", site_row.contact),
                            ("phone", site_row.phone),
                            ("email", site_row.email),
                            ("notes", site_row.notes),
                        ]
                        if v
                    )
                    locator = {
                        **locator_base,
                        "row_index": site_row.row_index,
                        "table_index": table_index,
                    }
                    out.append(
                        _make_atom(
                            text=site_text or canon_id,
                            project_id=project_id,
                            artifact_id=artifact_id,
                            filename=pdf_path.name,
                            parser_version=parser_version,
                            # v53.2 ROOT-CAUSE FIX: must be physical_site so
                            # downstream code (semantic_dedup, build_site_readiness
                            # canonical_set, find_authoritative_site_phrases) can
                            # find these as the canonical roster. Previously
                            # labeled AtomType.entity with value.kind="physical_site"
                            # — produced physical_site_atoms=0 envelope-wide.
                            atom_type=AtomType.physical_site,
                            authority_class=AuthorityClass.contractual_scope,
                            confidence=site_row.confidence,
                            locator=locator,
                            value={
                                "kind": "physical_site",
                                "id": sid or site_row.site_id,  # canonical id (drives canonical_set)
                                "site_id": sid or site_row.site_id,
                                "name": site_row.facility_name,  # also as `name` for cross-doc joins
                                "facility_name": site_row.facility_name,
                                "address": site_row.street_address,
                                "street_address": site_row.street_address,
                                "mdf_idf": site_row.mdf_idf,
                                "access_window": site_row.access_window,
                                "escort_owner": site_row.escort_owner,
                                "contact": site_row.contact,
                                "phone": site_row.phone,
                                "email": site_row.email,
                                "city_state": site_row.city_state,
                                # Organisational territory, NOT geography.
                                "region": getattr(site_row, "region", None),
                                "sqft": site_row.sqft,
                                "occupancy": site_row.occupancy,
                                "notes": site_row.notes,
                                "extras": dict(site_row.extra_fields),
                            },
                        )
                    )
    finally:
        try:
            doc.close()
        except Exception:
            pass

    # v53.8/v53.10 TEXT-BASED EXTRACTION: always attempt when the doc
    # text declares a roster section. Even if fitz-table extraction
    # produced some rows, the text scan catches IDs the table extractor
    # missed (truncated cells, columns mis-aligned in reportlab PDFs).
    # The `already_emitted` set prevents duplicate emission.
    try:
        already_emitted = already_emitted or set()
        # Add IDs from any already-emitted atoms in `out` (this call's
        # own atoms) so the text extractor doesn't re-emit them.
        for a in out:
            v = getattr(a, "value", None) or {}
            if isinstance(v, dict):
                sid = v.get("id") or v.get("site_id")
                if sid:
                    already_emitted.add(sid)
        text_atoms = _text_based_site_roster_extract(
            pdf_path=pdf_path,
            project_id=project_id,
            artifact_id=artifact_id,
            parser_version=parser_version,
            already_emitted=already_emitted,
        )
        # v53.12: explicit stderr log so we can see in cloud worker logs
        # whether the text extractor fired and how many atoms it produced.
        import sys as _sys_v512
        try:
            print(
                f"v53_text_roster: {pdf_path.name} fitz={len(out)} text={len(text_atoms)}",
                file=_sys_v512.stderr,
            )
        except Exception:
            pass
        out.extend(text_atoms)
    except Exception as _exc_v512:
        try:
            import sys as _sys_v512x
            print(f"v53_text_roster_FAIL: {pdf_path.name}: {_exc_v512}", file=_sys_v512x.stderr)
        except Exception:
            pass

    return out

def _text_based_site_roster_extract(
    *,
    pdf_path: Path,
    project_id: str,
    artifact_id: str,
    parser_version: str,
    already_emitted: set[str | None],
) -> list[EvidenceAtom]:
    """v53.8: scan PDF text for site-ID-shaped tokens when no roster
    table parsed. Triggers ONLY when document text explicitly declares
    a site roster. Catches reportlab-rendered PDFs where fitz can't
    detect the table layout but the IDs are visible in extracted text.

    Universal — works for any deal whose roster section declares
    site IDs in a recognizable shape.
    """
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:
        return []
    try:
        from app.parsers.site_roster_extractor import _SITE_ID_SHAPE_RE
    except Exception:
        return []
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return []

    out: list[EvidenceAtom] = []
    try:
        page_texts: list[str] = []
        for p in doc:
            try:
                page_texts.append(p.get_text() or "")
            except Exception:
                continue
        document_text = "\n".join(page_texts)

        # Gate: only fire when the page/document explicitly declares a
        # physical-site roster. The previous count-only fallback ("5+
        # site-shaped tokens") emitted payment/MSA/project IDs from
        # commercial and contracting packets as physical_site atoms. A
        # table/document must identify itself as a roster before this
        # function is allowed to mint canonical sites.
        text_lower = document_text.lower()
        compact_text = re.sub(r"\s+", " ", text_lower)
        explicit_roster = any(s in compact_text for s in [
            "kind=physical_site", "kind = physical_site",
            "physical site roster", "authoritative physical site",
            "authoritative site roster",
        ])
        table_roster = (
            "site roster" in compact_text
            and ("site id" in compact_text or "site no" in compact_text or "facility code" in compact_text)
            and ("facility name" in compact_text or "street address" in compact_text or "administrative site" in compact_text)
        )
        numeric_roster = (
            ("site no" in compact_text or "site no." in compact_text)
            and ("administrative site" in compact_text or "school site" in compact_text)
            and "lat, long" in compact_text
            and "zip" in compact_text
        )
        declares_roster = explicit_roster or table_roster or numeric_roster
        if not declares_roster:
            return []

        # Numeric public-sector rosters (APS Attachment B style) often
        # extract as one cell per line rather than a fitz table. Parse the
        # repeated sequence: site_no, site name (possibly wrapped), street,
        # city, zip, lat/long. This is schema-driven from the header, not a
        # customer-specific school list.
        if numeric_roster:
            raw_lines = [ln.strip() for ln in document_text.split("\n") if ln.strip()]
            header_noise = {
                "attachment b", "site", "no.", "site no.", "administrative site",
                "school site", "street", "city", "zip", "lat, long",
            }
            lines = [ln for ln in raw_lines if ln.strip().lower() not in header_noise]
            street_re = re.compile(
                r"^(?:\d+[A-Za-z-]*\s+|P\.?O\.?\s+Box\s+|#?N/?A\b)",
                re.IGNORECASE,
            )
            latlong_re = re.compile(r"^(?:-?\d{1,3}\.\d+\s*,\s*-?\d{1,3}\.\d+|#?N/?A)$", re.IGNORECASE)
            zip_re = re.compile(r"^(?:\d{5}(?:-\d{4})?|#?N/?A)$", re.IGNORECASE)
            i = 0
            while i < len(lines):
                if not re.fullmatch(r"\d{1,4}", lines[i]):
                    i += 1
                    continue
                site_no = lines[i]
                j = i + 1
                name_parts: list[str] = []
                while j < len(lines) and not street_re.match(lines[j]) and not re.fullmatch(r"\d{1,4}", lines[j]):
                    if not latlong_re.match(lines[j]) and not zip_re.match(lines[j]):
                        name_parts.append(lines[j])
                    j += 1
                if not name_parts or j + 3 >= len(lines) or not street_re.match(lines[j]):
                    i += 1
                    continue
                street = lines[j].strip()
                k = j + 1
                city_parts: list[str] = []
                while k < len(lines) and not zip_re.match(lines[k]) and not re.fullmatch(r"\d{1,4}", lines[k]):
                    city_parts.append(lines[k].strip())
                    k += 1
                    if len(city_parts) >= 4:
                        break
                zip_code = lines[k].strip() if k < len(lines) else ""
                lat_long = lines[k + 1].strip() if k + 1 < len(lines) else ""
                if not city_parts or not zip_re.match(zip_code) or not latlong_re.match(lat_long):
                    i += 1
                    continue
                city = " ".join(city_parts).strip()
                name = " ".join(name_parts).strip()
                sid = site_no
                if sid not in already_emitted:
                    already_emitted.add(sid)
                    text = f"Site No. {site_no} | {name} | {street} | {city} | {zip_code} | {lat_long}"
                    out.append(
                        _make_atom(
                            text=text,
                            project_id=project_id,
                            artifact_id=artifact_id,
                            filename=pdf_path.name,
                            parser_version=parser_version,
                            atom_type=AtomType.physical_site,
                            authority_class=AuthorityClass.contractual_scope,
                            confidence=0.90,
                            locator={"extraction": "site_roster_numeric_text_v54", "site_no": site_no},
                            value={
                                "kind": "physical_site",
                                "id": sid,
                                "site_id": sid,
                                "site_no": site_no,
                                "name": name,
                                "facility_name": name,
                                "administrative_site_name": name,
                                "address": street,
                                "street": street,
                                "street_address": street,
                                "city": city,
                                "zip": zip_code,
                                "lat_long": lat_long,
                            },
                        )
                    )
                i = k + 2

        # v53.12: known non-site prefixes that match the loose site-ID
        # regex but aren't actual sites. Universal — these are network
        # closets, days, system codes, etc.
        _NON_SITE_PREFIXES = {
            "MDF", "IDF", "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
            "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP",
            "OCT", "NOV", "DEC", "USB", "AC", "DC", "DC1", "ON", "OFF",
            "HVAC", "PoE", "POE", "AP", "AP1", "AP2", "ID", "PO", "QA",
            "URL", "GUI", "API", "SSO", "VPN", "WAN", "LAN", "PSU",
        }
        # Find every site-ID-shaped token in the text
        site_ids_seen: dict[str, str] = {}  # id → name guess
        for line in document_text.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            for token in line_stripped.split():
                clean = token.rstrip(":,;")
                if not _SITE_ID_SHAPE_RE.match(clean):
                    continue
                # v53.12: reject non-site prefixes
                prefix = clean.split("-", 1)[0].split("_", 1)[0].upper()
                if prefix in _NON_SITE_PREFIXES:
                    continue
                # Reject if doesn't contain at least 2 alpha chars AND
                # 1 digit (real site IDs typically have both — like
                # ATL-HQ-01, STORE-142). Pure alphabetic codes (FRI,
                # ATL-HQ without numeric suffix) are usually weak.
                # We allow ATL-HQ via the BOM catalog path; here we
                # only emit when there's a digit suffix proving it's
                # a numbered site row.
                if not re.search(r"\d", clean):
                    continue
                # v53.12b: require ≥2 alpha chars AND total length ≥6.
                # Filters out "W2", "B1", etc. — single-letter+digit
                # codes that aren't realistic site IDs.
                alpha_count = sum(1 for ch in clean if ch.isalpha())
                if alpha_count < 2 or len(clean) < 6:
                    continue
                # v56: require a trailing numeric suffix like "-NN" or "_NN".
                # Real numbered roster IDs always have one (ATL-HQ-01,
                # STORE-142). Facility names that incidentally match
                # the shape (ATL-AIR, ATL-WEST without a row number)
                # would otherwise leak through and be promoted later to
                # synthetic site_ids like OPTBOT-AIRPORT-LOGIST. By
                # requiring -NN we accept the authoritative roster
                # row IDs and reject everything else.
                if not re.search(r"[-_]\d+$", clean):
                    continue
                if clean in already_emitted or clean in site_ids_seen:
                    continue
                # Try to extract a facility name from the same line.
                # Pattern: "ATL-HQ-01 OPTBOT Atlanta HQ 1200 ..."
                try:
                    after = line_stripped[line_stripped.index(token) + len(token):].strip()
                except Exception:
                    after = ""
                after = after.lstrip(":,; -|\t")
                name_match = re.match(
                    r"^([A-Za-z][A-Za-z0-9\s\-&'.]{2,60}?)"
                    r"(?=\s+\d|\s+(?:Street|St\.|Avenue|Ave|Road|Rd|Blvd|Parkway|Pkwy|Drive|Dr\.)|$)",
                    after,
                )
                facility = (name_match.group(1).strip() if name_match else after[:50].strip())
                facility = re.sub(r"\s+[A-Z]$", "", facility).strip()
                # v53.12: reject facility names containing days/months — caught from
                # adjacent table cells in PDF text flow.
                facility_low = facility.lower()
                if any(w in facility_low for w in ["mon-fri", "mon-sat", "tue ", "wed ", "thu ", "fri ", "sat ", "sun "]):
                    facility = clean  # fallback to id-as-name
                site_ids_seen[clean] = facility or clean

        # Emit one physical_site atom per discovered ID
        for sid, facility_name in site_ids_seen.items():
            if sid in already_emitted:
                continue
            already_emitted.add(sid)
            text = f"{sid} | {facility_name}".strip(" |")
            out.append(
                _make_atom(
                    text=text,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=pdf_path.name,
                    parser_version=parser_version,
                    atom_type=AtomType.physical_site,
                    authority_class=AuthorityClass.contractual_scope,
                    confidence=0.78,
                    locator={"extraction": "site_roster_text_fallback_v53_8"},
                    value={
                        "kind": "physical_site",
                        "id": sid,
                        "site_id": sid,
                        "name": facility_name or sid,
                        "facility_name": facility_name or sid,
                    },
                )
            )
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return out
