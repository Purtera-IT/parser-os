"""Column schema registry — recognizes structured table schemas and splits rows
into properly typed atoms instead of one blob per row.

v49: Core architectural fix for 14 zero/weak extraction categories.
A vendor_line_item blob carries site_allocation, quantity, unit_price, sku
all compressed into one text. The classifier can only pick ONE type. This
registry detects the table schema from headers and emits the correct set
of typed atoms for each row so every fact gets its own typed atom.
"""
from __future__ import annotations

import re
from typing import Any

from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    EvidenceReceipt,
    ReviewStatus,
    SourceRef,
)

# ═══════════════════════════════════════════════════════════
# COLUMN HEADER NORMALIZATION
# ═══════════════════════════════════════════════════════════


def _norm_col(s: str) -> str:
    """Normalize a column header for matching: lowercase, collapse spaces/punct."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _col_matches(header: str, patterns: tuple[str, ...]) -> bool:
    """True when any pattern matches the normalized header on WORD boundaries.

    Word-boundary (not raw substring) so a short pattern like ``id`` matches
    "Requirement ID" but NOT "Validation Test" (where it was substring-matching
    inside "valIDation"). That false positive made the requirements schema fire
    on room-mix tables (Room Type / Count / Standard Build / Validation Test),
    scrambling room rows into requirement atoms. Patterns are normalized the same
    way as the header so multi-word and punctuated patterns ("requirement id",
    "sv-") still match their tokens.
    """
    h = _norm_col(header)
    for p in patterns:
        pn = _norm_col(p)
        if pn and re.search(r"\b" + re.escape(pn) + r"\b", h):
            return True
    return False


# ═══════════════════════════════════════════════════════════
# SCHEMA DEFINITIONS
# ═══════════════════════════════════════════════════════════

_SCHEMAS: list[tuple[str, tuple[tuple[str, ...], ...], int]] = [
    # ─── v53.2: Site roster (HIGHEST priority — placed first so it wins
    # over site_budget when both could match) ───
    # Universal site-roster shape across customer docs:
    #   OPTBOT:  Site ID | Facility name | Street address | MDF/IDF | Access window
    #   APS:     Site No. | Administrative Site | Street | City | Zip | Lat,Long
    #   Generic: Code | Location | Address | Type
    # Need 2 of 3 column groups (id-like + name-like + address-like) to fire
    # so a generic ("site" + "cost") site_budget table doesn't accidentally match.
    (
        "site_roster",
        (
            # ID-like column
            ("site id", "site no", "site #", "site code", "facility id", "location id", "school id", "no.", "no "),
            # Name-like column
            ("facility name", "site name", "location name", "school site", "administrative site", "facility", "campus"),
            # Address-like column
            ("street address", "street", "address"),
        ),
        2,
    ),
    # ─── Services line items (split from BOM in v52) ───
    # Distinguishing signal: explicit "service" or "sv-" prefix in id/desc.
    # Order MATTERS — services placed before bom so it wins when both match.
    (
        "service_line",
        (
            ("service id", "sv-", "svc-", "service item"),
            ("description", "service", "scope"),
            ("unit price", "unit cost", "fixed fee", "rate"),
        ),
        2,
    ),
    # ─── BOM / Hardware quote ───
    # Requires SKU/part anchor so Services rows (no SKU) don't match.
    (
        "bom",
        (
            ("sku", "part no", "part number", "model", "item id", "hw-", "sw-", "catalog"),
            ("qty", "quantity", "count", "units"),
            ("unit cost", "unit price", "list price", "each", "unit $"),
        ),
        2,
    ),
    # ─── Cutover / cutover plan ───
    (
        "cutover",
        (
            ("step", "t minus", "t-", "timing", "day", "cutover step"),
            ("owner", "responsible", "assigned", "who"),
            ("activity", "description", "action", "task", "what", "checklist item"),
        ),
        2,
    ),
    # ─── Per-site room mix (its OWN schema so greedy schemas like acceptance
    # / requirements don't grab it via "Validation Test" → "test"/"id"). A
    # room-mix row is Room Type | Count | Standard Build | Validation Test. ───
    (
        "site_room_mix",
        (
            ("room type", "room", "space type"),
            ("count", "rooms", "qty", "quantity"),
            ("standard build", "build", "equipment", "spec", "fit out"),
            ("validation test", "validation", "acceptance test"),
        ),
        2,
    ),
    # ─── Requirements table ───
    (
        "requirements",
        (
            ("req", "requirement id", "id", "ref"),
            ("description", "requirement", "shall", "must", "will provide"),
            ("priority", "criticality", "must have", "category", "type"),
        ),
        2,
    ),
    # ─── Signatory / approval block ───
    (
        "signatory",
        (
            ("name", "printed name", "authorized"),
            ("title", "position", "role"),
            ("signature", "sign", "approved by"),
            ("date",),
        ),
        2,
    ),
    # ─── Acceptance criteria ───
    # GAP C FIX: min_matches=1 because SOW uses 2-col "Acceptance Area | Criteria"
    # where BOTH columns hit the same pattern group. With min_matches=2 the schema
    # never fires. min_matches=1 catches any acceptance-style table.
    (
        "acceptance",
        (
            ("criterion", "test", "acceptance", "criteria", "check", "acceptance area"),
            ("method", "procedure", "how to verify", "test method"),
            ("pass", "threshold", "result", "expected"),
        ),
        1,
    ),
    # ─── Deliverables table ───
    (
        "deliverables",
        (
            ("deliverable", "artifact", "output", "document"),
            ("due", "date", "deadline", "by"),
            ("owner", "responsible", "accountable", "assigned to"),
        ),
        2,
    ),
    # ─── Integration checkpoints ───
    (
        "integration_checkpoints",
        (
            ("checkpoint", "ic", "integration", "test id"),
            ("system", "platform", "application", "target"),
            ("test", "validation", "verify", "confirm"),
        ),
        2,
    ),
    # ─── Site budget / site cost ───
    (
        "site_budget",
        (
            ("site", "location", "building"),
            ("budget", "subtotal", "cost", "total", "amount"),
        ),
        2,
    ),
    # ─── Compliance / classification ───
    (
        "compliance",
        (
            ("classification", "class", "data type", "category"),
            ("allowed", "destination", "handling", "restriction", "rule"),
        ),
        2,
    ),
    # ─── System mapping ───
    (
        "system_mapping",
        (
            ("source", "from system", "existing system", "legacy"),
            ("target", "destination", "new system", "replaces"),
            ("field", "mapping", "transform", "data element"),
        ),
        2,
    ),
    # ─── v50 new schemas ───
    # Task table (Detailed Tasks sheets, RFP task lists)
    (
        "task",
        (
            ("task", "task id", "activity id", "id", "ref", "#", "no"),
            ("owner", "assigned", "responsible", "lead", "who"),
            ("due", "date", "start", "end", "deadline", "target"),
        ),
        2,
    ),
    # Milestone / phase / wave table
    (
        "milestone",
        (
            ("phase", "milestone", "wave", "sprint", "stage", "iteration"),
            ("start", "begin", "kickoff", "start date", "from"),
            ("end", "due", "completion", "finish", "end date", "to"),
        ),
        2,
    ),
    # Risk register
    (
        "risk_register",
        (
            ("risk", "risk id", "raid id", "id", "issue"),
            ("probability", "likelihood", "chance"),
            ("impact", "severity", "consequence"),
            ("mitigation", "response plan", "treatment", "action"),
        ),
        2,
    ),
    # Stakeholder table
    (
        "stakeholder_table",
        (
            ("name", "stakeholder", "contact", "person", "owner"),
            ("role", "title", "position", "function"),
            ("email", "phone", "contact info", "@"),
        ),
        2,
    ),
]


def identify_schema(columns: list[str]) -> str | None:
    """Return schema name if the column list matches a known schema, else None.

    Matching: count how many of the schema's required-column tuples have at
    least ONE column header in the table that matches. Return the schema with
    the highest match count that meets its minimum.
    """
    if not columns:
        return None
    best_schema: str | None = None
    best_score = 0
    for schema_name, req_patterns, min_matches in _SCHEMAS:
        matches = sum(
            1 for pattern_group in req_patterns
            if any(_col_matches(col, pattern_group) for col in columns)
        )
        if matches >= min_matches and matches > best_score:
            best_score = matches
            best_schema = schema_name
    return best_schema


def _parse_site_allocation(text: str) -> list[dict[str, Any]]:
    """Parse a site allocation string like 'HQ:52;WEST:27;AIR:15' into list of dicts."""
    if not text or not text.strip():
        return []
    text = text.strip()
    pair_pattern = re.compile(r"([A-Z][A-Z0-9_\-]{1,20})\s*[:=]\s*(\d+)", re.IGNORECASE)
    pairs = pair_pattern.findall(text)
    if pairs:
        return [{"site": s.strip(), "qty": int(q)} for s, q in pairs]
    m = re.search(r"^\s*(\d+)\s*$", text)
    if m:
        return [{"site": "all", "qty": int(m.group(1))}]
    return []


def _find_col_value(row_dict: dict[str, str], patterns: tuple[str, ...]) -> str:
    """Find the value of the first column that matches any pattern."""
    for col, val in row_dict.items():
        if _col_matches(col, patterns):
            return (val or "").strip()
    return ""


def emit_atoms_for_schema(
    *,
    schema_name: str,
    columns: list[str],
    row: list[str],
    row_idx: int,
    table_idx: int,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str = "table_schema_v49",
    section_path: list[str] | None = None,
) -> list[EvidenceAtom]:
    """Given a recognized schema, header list, and one data row, emit typed atoms.

    ``section_path`` (the heading/sheet chain the source row lives under) is
    carried onto every emitted atom's locator so section/site attribution
    survives the schema-routing step instead of being reset to empty.
    """
    if not row or not columns:
        return []
    _section_path = list(section_path) if section_path else []

    row_dict = dict(zip(columns, row + [""] * max(0, len(columns) - len(row))))
    row_text = " | ".join(v for v in row if v.strip())
    if not row_text.strip():
        return []

    def _make_src(suffix: str) -> SourceRef:
        atom_id = stable_id("atm", artifact_id, "schema_row", table_idx, row_idx, suffix)
        return SourceRef(
            id=stable_id("src", atom_id),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.docx,
            filename=filename,
            locator={
                "table_index": table_idx,
                "row": row_idx,
                "schema": schema_name,
                "extraction": "table_schema_v49",
                "section_path": _section_path,
            },
            extraction_method="table_schema_v49",
            parser_version=parser_version,
        )

    def _atom(suffix: str, atom_type: AtomType, text: str, value: dict) -> EvidenceAtom:
        aid = stable_id("atm", artifact_id, "schema_row", table_idx, row_idx, suffix)
        src = _make_src(suffix)
        receipt = EvidenceReceipt(
            atom_id=aid,
            artifact_id=artifact_id,
            filename=filename,
            source_ref_id=src.id,
            replay_status="unsupported",
            extracted_snippet=text[:500],
            locator=src.locator,
            reason="post_source_replay_table_schema_atom",
            verifier_version=parser_version,
        )
        return EvidenceAtom(
            id=aid,
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=atom_type,
            raw_text=text[:4000],
            normalized_text=text.lower()[:4000],
            value=value,
            entity_keys=[],
            source_refs=[src],
            receipts=[receipt],
            authority_class=AuthorityClass.contractual_scope,
            confidence=0.85,
            confidence_raw=0.85,
            calibrated_confidence=0.85,
            review_status=ReviewStatus.auto_accepted,
            review_flags=[],
            parser_version=parser_version,
        )

    atoms: list[EvidenceAtom] = []

    if schema_name == "bom":
        item_id = _find_col_value(row_dict, ("item id", "hw-", "sw-", "id", "line", "#"))
        description = _find_col_value(row_dict, ("description", "product", "name", "part desc"))
        sku = _find_col_value(row_dict, ("sku", "part no", "part number", "model", "catalog"))
        qty_raw = _find_col_value(row_dict, ("qty", "quantity", "count", "units", "ea"))
        unit_cost_raw = _find_col_value(row_dict, ("unit cost", "unit price", "list price", "each", "unit $"))
        site_alloc_raw = _find_col_value(row_dict, ("site allocation", "site split", "per site", "allocation"))
        category = _find_col_value(row_dict, ("category", "type", "class"))

        qty = None
        m = re.search(r"(\d[\d,]*)", qty_raw)
        if m:
            try:
                qty = int(m.group(1).replace(",", ""))
            except ValueError:
                pass

        unit_cost = None
        m = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)", unit_cost_raw)
        if m:
            try:
                unit_cost = float(m.group(1).replace(",", ""))
            except ValueError:
                pass

        atoms.append(_atom(
            "bom_line",
            AtomType.bom_line,
            row_text,
            {
                "item_id": item_id, "description": description, "sku": sku,
                "qty": qty, "unit_cost": unit_cost, "category": category,
                "row_text": row_text,
            },
        ))

        for alloc in _parse_site_allocation(site_alloc_raw):
            site_code = alloc["site"]
            site_qty = alloc["qty"]
            alloc_text = f"{description or item_id or sku} | site: {site_code} | qty: {site_qty}"
            atoms.append(_atom(
                f"site_alloc_{site_code}",
                AtomType.site_allocation,
                alloc_text,
                {"site": site_code, "item_id": item_id, "description": description,
                 "sku": sku, "qty": site_qty},
            ))

        # GAP B FIX: lead_time_constraint from "Lead Time Days" column.
        lead_time_raw = _find_col_value(
            row_dict, ("lead time", "lead_time", "lead time days", "weeks", "procurement days")
        )
        if lead_time_raw:
            lt_days = None
            m = re.search(r"(\d+)", lead_time_raw)
            if m:
                lt_days = int(m.group(1))
            lt_text = f"{description or sku or item_id}: lead time {lead_time_raw}"
            atoms.append(_atom(
                "lead_time",
                AtomType.lead_time_constraint,
                lt_text,
                {"item_id": item_id, "description": description, "sku": sku,
                 "lead_time_raw": lead_time_raw, "lead_time_days": lt_days},
            ))

    elif schema_name == "service_line":
        svc_id = _find_col_value(row_dict, ("service id", "sv-", "svc-", "id", "#"))
        description = _find_col_value(row_dict, ("description", "service", "scope", "name"))
        unit = _find_col_value(row_dict, ("unit", "uom"))
        qty_raw = _find_col_value(row_dict, ("qty", "quantity", "count", "units", "ea"))
        unit_price_raw = _find_col_value(row_dict, ("unit price", "unit cost", "fixed fee", "rate", "list price"))
        ext_raw = _find_col_value(row_dict, ("extended", "ext cost", "total", "subtotal"))
        notes = _find_col_value(row_dict, ("notes", "remarks", "comment"))

        qty = None
        m = re.search(r"(\d[\d,]*)", qty_raw)
        if m:
            try:
                qty = int(m.group(1).replace(",", ""))
            except ValueError:
                pass
        unit_price = None
        m = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)", unit_price_raw)
        if m:
            try:
                unit_price = float(m.group(1).replace(",", ""))
            except ValueError:
                pass

        if svc_id or description:
            atoms.append(_atom(
                "service_line",
                AtomType.service_line,
                row_text,
                {"service_id": svc_id, "description": description, "unit": unit,
                 "qty": qty, "unit_price": unit_price, "extended_cost": ext_raw,
                 "notes": notes, "row_text": row_text},
            ))

    elif schema_name == "cutover":
        step_id = _find_col_value(row_dict, ("step", "id", "#", "no"))
        timing = _find_col_value(row_dict, ("t minus", "t-", "timing", "day", "schedule", "when"))
        owner = _find_col_value(row_dict, ("owner", "responsible", "assigned", "who", "role"))
        description = _find_col_value(row_dict, ("activity", "description", "action", "task", "what", "step desc", "checklist item"))
        if not description:
            description = row_text
        atoms.append(_atom(
            "cutover_step",
            AtomType.cutover_step,
            description or row_text,
            {"step_id": step_id, "timing": timing, "owner": owner, "description": description},
        ))

    elif schema_name == "site_room_mix":
        room_type = _find_col_value(row_dict, ("room type", "room", "space type"))
        count = _find_col_value(row_dict, ("count", "rooms", "qty", "quantity"))
        build = _find_col_value(row_dict, ("standard build", "build", "equipment", "spec", "fit out"))
        validation = _find_col_value(row_dict, ("validation test", "validation", "acceptance test", "test"))
        if room_type or count:
            atoms.append(_atom(
                "site_room_mix",
                AtomType.site_room_mix,
                f"{room_type} | {count}" if room_type else row_text,
                {"room_type": room_type, "count": count, "build_spec": build, "validation": validation},
            ))

    elif schema_name == "requirements":
        req_id = _find_col_value(row_dict, ("req", "id", "ref", "requirement id", "#"))
        description = _find_col_value(row_dict, ("description", "requirement", "shall", "must", "text"))
        if not description:
            description = row_text
        priority = _find_col_value(row_dict, ("priority", "criticality", "must have", "p1", "p2"))
        site = _find_col_value(row_dict, ("site", "location", "scope", "applicable to"))
        category = _find_col_value(row_dict, ("category", "type", "domain", "area"))
        atoms.append(_atom(
            "requirement",
            AtomType.requirement,
            description or row_text,
            {"req_id": req_id, "description": description, "priority": priority,
             "site": site, "category": category},
        ))

    elif schema_name == "signatory":
        name = _find_col_value(row_dict, ("name", "printed name", "authorized", "by"))
        title = _find_col_value(row_dict, ("title", "position", "role", "capacity"))
        signature = _find_col_value(row_dict, ("signature", "sign", "approved by"))
        date = _find_col_value(row_dict, ("date",))
        if name or signature:
            atoms.append(_atom(
                "signatory",
                AtomType.signatory,
                f"{name} | {title}" if name else signature,
                {"name": name, "title": title, "date": date, "raw": row_text},
            ))

    elif schema_name == "acceptance":
        criterion = _find_col_value(row_dict, ("criterion", "test", "acceptance", "criteria", "check", "item"))
        method = _find_col_value(row_dict, ("method", "procedure", "how to verify", "test method", "approach"))
        threshold = _find_col_value(row_dict, ("pass", "threshold", "result", "expected", "acceptable"))
        if criterion:
            atoms.append(_atom(
                "acceptance_criterion",
                AtomType.acceptance_criterion,
                row_text,  # full row, not just the area — carries context AND dedups
                {"criterion": criterion, "method": method, "threshold": threshold, "raw": row_text},
            ))

    elif schema_name == "deliverables":
        name = _find_col_value(row_dict, ("deliverable", "artifact", "output", "document", "item"))
        due = _find_col_value(row_dict, ("due", "date", "deadline", "by", "target"))
        owner = _find_col_value(row_dict, ("owner", "responsible", "accountable", "assigned to"))
        description = _find_col_value(row_dict, ("description", "detail", "notes", "acceptance"))
        if name:
            atoms.append(_atom(
                "deliverable",
                AtomType.deliverable,
                name,
                {"name": name, "due": due, "owner": owner, "description": description, "raw": row_text},
            ))

    elif schema_name == "integration_checkpoints":
        ic_id = _find_col_value(row_dict, ("checkpoint", "ic", "id", "#", "test id"))
        system = _find_col_value(row_dict, ("system", "platform", "application", "target", "integration"))
        test = _find_col_value(row_dict, ("test", "validation", "verify", "confirm", "steps"))
        criteria = _find_col_value(row_dict, ("criteria", "pass", "result", "expected", "acceptance"))
        if ic_id or test:
            atoms.append(_atom(
                "integration_checkpoint",
                AtomType.integration_checkpoint,
                f"{ic_id} | {system} | {test}" if ic_id else test,
                {"ic_id": ic_id, "system": system, "test": test, "criteria": criteria, "raw": row_text},
            ))

    elif schema_name == "site_budget":
        site = _find_col_value(row_dict, ("site", "location", "building", "facility"))
        budget = _find_col_value(row_dict, ("budget", "subtotal", "cost", "total", "amount"))
        if site and budget:
            atoms.append(_atom(
                "site_budget",
                AtomType.site_budget,
                f"{site} | {budget}",
                {"site": site, "budget": budget, "raw": row_text},
            ))

    elif schema_name == "compliance":
        classification = _find_col_value(row_dict, ("classification", "class", "data type", "category", "level"))
        rule = _find_col_value(row_dict, ("allowed", "destination", "handling", "restriction", "rule", "requirement"))
        if classification:
            atoms.append(_atom(
                "compliance_classification",
                AtomType.compliance_classification,
                f"{classification} | {rule}" if rule else classification,
                {"classification": classification, "rule": rule, "raw": row_text},
            ))

    elif schema_name == "system_mapping":
        source = _find_col_value(row_dict, ("source", "from system", "existing system", "legacy", "from"))
        target = _find_col_value(row_dict, ("target", "destination", "new system", "replaces", "to"))
        field = _find_col_value(row_dict, ("field", "mapping", "transform", "data element", "attribute"))
        if source or target:
            atoms.append(_atom(
                "system_mapping",
                AtomType.system_mapping,
                f"{source} -> {target}" if source and target else source or target,
                {"source": source, "target": target, "field": field, "raw": row_text},
            ))

    elif schema_name == "task":
        task_id = _find_col_value(row_dict, ("task id", "activity id", "id", "ref", "#", "no"))
        site = _find_col_value(row_dict, ("site", "location", "scope", "where"))
        phase = _find_col_value(row_dict, ("phase", "milestone", "wave", "sprint"))
        name = _find_col_value(row_dict, ("task", "activity", "description", "name", "title"))
        owner = _find_col_value(row_dict, ("owner", "assigned", "responsible", "lead", "who"))
        start = _find_col_value(row_dict, ("start", "begin", "from", "start date"))
        due = _find_col_value(row_dict, ("due", "end", "deadline", "target", "by", "due date", "end date"))
        dependency = _find_col_value(row_dict, ("dependency", "depends on", "predecessor", "prior", "blocked by"))
        status = _find_col_value(row_dict, ("status", "state", "progress"))
        if name or task_id:
            atoms.append(_atom(
                "task",
                AtomType.task,
                name or row_text,
                {"task_id": task_id, "site": site, "phase": phase, "name": name,
                 "owner": owner, "start": start, "due": due, "dependency": dependency,
                 "status": status, "raw": row_text},
            ))
            # Also emit a dependency atom when the row carries one — gives PM
            # an explicit Gantt-style predecessor link.
            if dependency:
                atoms.append(_atom(
                    "task_dep",
                    AtomType.dependency,
                    f"{name or task_id} depends on {dependency}",
                    {"dependent": name or task_id, "depends_on": dependency,
                     "dependency_type": "predecessor", "raw": row_text},
                ))

    elif schema_name == "milestone":
        phase_id = _find_col_value(row_dict, ("phase", "milestone", "wave", "sprint", "stage", "#"))
        name = _find_col_value(row_dict, ("name", "title", "description", "deliverable"))
        start = _find_col_value(row_dict, ("start", "begin", "kickoff", "start date", "from"))
        end = _find_col_value(row_dict, ("end", "due", "completion", "finish", "end date", "to"))
        owner = _find_col_value(row_dict, ("owner", "lead", "responsible", "accountable"))
        exit_criteria = _find_col_value(row_dict, ("exit", "criteria", "deliverable", "completion criteria"))
        atoms.append(_atom(
            "milestone",
            AtomType.milestone_phase,
            row_text,  # full row, not just the name — carries context AND dedups
            {"phase_id": phase_id, "name": name, "start": start, "end": end,
             "owner": owner, "exit_criteria": exit_criteria, "raw": row_text},
        ))

    elif schema_name == "risk_register":
        risk_id = _find_col_value(row_dict, ("risk id", "raid id", "id", "issue id", "#"))
        description = _find_col_value(row_dict, ("description", "risk", "issue", "summary", "text"))
        probability = _find_col_value(row_dict, ("probability", "likelihood", "chance", "p"))
        impact = _find_col_value(row_dict, ("impact", "severity", "consequence", "i"))
        owner = _find_col_value(row_dict, ("owner", "assigned", "raid owner"))
        mitigation = _find_col_value(row_dict, ("mitigation", "response plan", "treatment", "action"))
        if description or risk_id:
            # Full row, not just the description — carries context AND dedups
            # with the bound row. Mitigation stays as a value field (no separate
            # context-less mitigation atom).
            atoms.append(_atom(
                "risk_register",
                AtomType.risk,
                row_text,
                {"risk_id": risk_id, "description": description,
                 "probability": probability, "impact": impact,
                 "owner": owner, "mitigation": mitigation, "raw": row_text},
            ))

    elif schema_name == "site_roster":
        # v53.2: emit a physical_site atom per row. Captures EVERY site
        # in the authoritative roster table — drives the canonical
        # catalog used by the central site:* gate so ghost sites can't
        # leak through.
        site_id = _find_col_value(
            row_dict,
            ("site id", "site no", "site #", "site code", "facility id",
             "location id", "school id", "no.", "no "),
        )
        name = _find_col_value(
            row_dict,
            ("facility name", "site name", "location name", "school site",
             "administrative site", "facility", "campus"),
        )
        address = _find_col_value(row_dict, ("street address", "street", "address"))
        city = _find_col_value(row_dict, ("city",))
        state = _find_col_value(row_dict, ("state", "province"))
        zip_code = _find_col_value(row_dict, ("zip", "postal", "post code"))
        mdf_idf = _find_col_value(row_dict, ("mdf", "idf", "telecom", "wiring closet"))
        access_window = _find_col_value(
            row_dict, ("access window", "access hours", "hours", "operating hours"),
        )
        escort_owner = _find_col_value(
            row_dict, ("escort owner", "escort", "contact", "facility contact", "owner"),
        )
        latlong = _find_col_value(row_dict, ("lat", "long", "lat, long", "coordinates", "gps"))
        # Skip header-repeat rows and totals/sub-totals.
        # v53.7: also reject "all" / "various" / "none" / pure-numbers
        # placeholders that show up in site allocation columns of
        # spreadsheet tables.
        _bad = (name or site_id or "").strip().lower()
        if not _bad or _bad in {
            "total", "sum", "subtotal", "n/a", "tbd",
            "all", "various", "none", "unknown", "all sites",
            "all locations",
        }:
            return atoms
        # Also reject if id is a year token only (e.g. "2026").
        if site_id and site_id.strip().isdigit() and len(site_id.strip()) == 4:
            return atoms
        # Need at least site_id OR name OR address — drop pure-empty rows.
        if not (site_id or name or address):
            return atoms
        canonical = (site_id or name or "").strip()
        if not canonical:
            return atoms
        atoms.append(_atom(
            "physical_site",
            AtomType.physical_site,
            f"{canonical} | {name} | {address}".strip(" |"),
            {
                "kind": "physical_site",
                "id": canonical,
                "site_id": site_id or canonical,
                "name": name or canonical,
                "facility_name": name,
                "address": address,
                "street_address": address,
                "city": city, "state": state, "zip": zip_code,
                "mdf_idf": mdf_idf,
                "access_window": access_window,
                "escort_owner": escort_owner,
                "lat_long": latlong,
                "raw": row_text,
            },
        ))

    elif schema_name == "stakeholder_table":
        name = _find_col_value(row_dict, ("name", "stakeholder", "contact", "person", "full name"))
        title = _find_col_value(row_dict, ("title", "position"))
        role = _find_col_value(row_dict, ("role", "function", "responsibility"))
        email = _find_col_value(row_dict, ("email", "e-mail", "address"))
        phone = _find_col_value(row_dict, ("phone", "telephone", "tel"))
        org = _find_col_value(row_dict, ("org", "organization", "company", "side"))
        approval_domain = _find_col_value(row_dict, ("approval", "authority", "approves", "domain"))
        if name:
            atoms.append(_atom(
                "stakeholder",
                AtomType.stakeholder,
                f"{name} | {title or role}" if (title or role) else name,
                {"name": name, "title": title, "role": role, "email": email,
                 "phone": phone, "org": org, "approval_domain": approval_domain,
                 "raw": row_text},
            ))

    return atoms


__all__ = ["identify_schema", "emit_atoms_for_schema"]
