"""OrbitBrief project envelope — the "perfect compressible" LLM input.

OWNER: this module builds the envelope. parser-os is stage one; it
produces every ``orbitbrief.input.v2`` payload in blob, and Orbitbrief-Core
consumes that payload across the seam (``orbitbrief_core.seam``) without
ever re-parsing.

An earlier docstring here claimed this module was deprecated and had
"moved to Orbitbrief-Core". It had not. Core carried a 1,244-line copy
that nothing imported, ~700 lines diverged from this one, whose own
docstring claimed the opposite — each file telling the reader the other
was authoritative. Core's copy was deleted 2026-08-28; this is the only
envelope builder, and the seam model in ``orbitbrief_core.seam.envelope``
is the only consumer-side contract.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.document_lifecycle.dataset import lookup as _lifecycle_lookup
from app.core.document_lifecycle import timeline as _timeline
from app.core.document_lifecycle import deal_stage as _deal_stage
from app.core.document_lifecycle import scope as _scope
from app.core.document_lifecycle.kit_health import check_deal_kit as _check_deal_kit
from app.core.orbitbrief_core import (
    build_bill_of_materials,
    build_change_order_timeline,
    build_deal_financials,
    build_deal_header,
    build_pm_dashboard,
    build_project_vitals,
    build_scope_truth,
    build_site_readiness,
    build_sow_readiness_scorecard,
    build_srl_missing_checklist,
    build_stakeholder_load,
)
from app.core.schemas import (
    ArtifactType,
    CompileResult,
    EntityRecord,
    EvidenceAtom,
    EvidenceEdge,
    EvidencePacket,
    SourceRef,
)
from app.parsers.site_roster_extractor import capped_source_row
from app.parsers.structured_projection import (
    DERIVED_DIR_SUFFIX,
    STRUCTURED_FILENAME,
    structured_doc_to_markdown,
)

ENVELOPE_SCHEMA_VERSION = "orbitbrief.input.v2"
ENVELOPE_FILENAME = "orbitbrief.input.json"
ENVELOPE_MARKDOWN_FILENAME = "orbitbrief.input.md"
PARSER_MANIFEST_SIDECAR = ".parser_manifest.json"


# ────────────────────────── public API ───────────────────────────────────


def build_orbitbrief_envelope(
    *,
    project_dir: Path,
    compile_result: CompileResult,
) -> dict[str, Any]:
    """Build the in-memory envelope from a compile result.

    The compile result already carries every parsed atom, packet, and the
    manifest with parser routing — we just need to fuse those together
    with each artifact's structured projection (PDFs use their persisted
    ``structured.json``; non-PDF parsers get a synthesized projection
    from their atoms grouped by section path).
    """
    project_dir = Path(project_dir).resolve()
    manifest = compile_result.manifest
    atoms = list(compile_result.atoms or [])
    # Binary-region placeholders ("[Image awaiting OCR / vision …]") are
    # coverage bookkeeping, not evidence. One reached the PM as an
    # open_question (live 010215, R5). They leave the atom stream here and are
    # reported under ``coverage.unrecovered_regions`` so a silent zero and a
    # real zero never look the same: the region is still named, just not
    # asked as a question.
    unrecovered_regions: list[dict[str, Any]] = []
    _kept: list[EvidenceAtom] = []
    for _a in atoms:
        _v = _a.value if isinstance(getattr(_a, "value", None), dict) else {}
        if str(_v.get("kind") or "").endswith("_marker"):
            unrecovered_regions.append({
                "artifact_id": _a.artifact_id,
                "kind": _v.get("kind"),
                "region_ref": _v.get("region_ref"),
                "size_bytes": _v.get("size_bytes"),
                "text": _a.raw_text,
            })
        else:
            _kept.append(_a)
    atoms = _kept
    packets = list(compile_result.packets or [])
    entities = list(compile_result.entities or [])
    edges = list(compile_result.edges or [])

    atoms_by_artifact: dict[str, list[EvidenceAtom]] = defaultdict(list)
    for atom in atoms:
        atoms_by_artifact[atom.artifact_id].append(atom)

    # A6 graceful degradation: build a per-file outcome index from the
    # manifest's parser_routing so each document carries its own
    # status (ok / ok_empty / skipped_no_parser / failed_parse).
    # PM_HANDOFF builders read this to render a "Files processed"
    # table and avoid the silent failure where a parse error left the
    # file count looking normal but produced 0 evidence.
    outcome_by_artifact: dict[str, dict[str, Any]] = {}
    if manifest is not None:
        for routing_entry in (manifest.parser_routing or []):
            aid = routing_entry.get("artifact_id")
            outcome = routing_entry.get("outcome")
            if aid and isinstance(outcome, dict):
                outcome_by_artifact[aid] = outcome

    # WHEN each artifact was authored and WHOSE it is, plus the deal's own stage
    # transitions. Together these decide where a document sits in the deal's life
    # and therefore who may read it -- see document_lifecycle/deal_stage.py.
    provenance = _load_manifest_provenance(project_dir)
    _crm_ctx = _load_manifest_crm(project_dir) or {}
    stage_timeline = _crm_ctx.get("stage_timeline") if isinstance(_crm_ctx, dict) else None

    # Which keys mean THIS deal, for scope detection. A document naming only
    # these stays deal-scoped; one naming others is speaking for more than this
    # deal. See document_lifecycle/scope.py.
    _this_keys: list[str] = []
    _dn = _deal_number_from_crm(_crm_ctx)
    if _dn:
        _this_keys.append(_dn)
    for _d in (_scope.PO_RE.finditer(str(_crm_ctx.get("deal_name") or "")) if _crm_ctx else []):
        _this_keys.append(_d.group(1))

    documents: list[dict[str, Any]] = []
    artifact_iter = manifest.artifact_fingerprints if manifest is not None else []
    for fp in artifact_iter:
        artifact_atoms = atoms_by_artifact.get(fp.artifact_id, [])
        artifact_path = _resolve_artifact_path(project_dir, fp.filename)
        structured_projection = _structured_projection_for(
            artifact_path=artifact_path,
            artifact_type=fp.artifact_type,
            artifact_atoms=artifact_atoms,
            filename=fp.filename,
        )
        # Which lifecycle stage this document belongs to, and therefore who may
        # read it. Looked up by content hash from a precomputed table -- no model
        # call in the compile path. Absent for documents we have never classified,
        # and absent is not a guess: consumers quarantine rather than assume.
        lifecycle = _lifecycle_lookup(fp.sha256)
        # The timeline cut. A document that reached us AFTER we sent the quote or
        # the SOW cannot be part of why we priced it that way -- reading it as
        # quoting evidence is hindsight, and it trains a head on facts the head
        # will not have at the moment it has to answer. So each document carries
        # when it arrived and which side of the deal's own cut that falls on.
        #
        # Both fields are conservative by construction: no delivering message, no
        # cut, or an unparseable timestamp all yield ``after_cut: false``. Ruling
        # real evidence OUT is the expensive direction of this error, so the
        # ambiguous case stays admissible and stays visible.
        if lifecycle is not None:
            arrived = _timeline.delivered_at(lifecycle)
            lifecycle["delivered_at"] = arrived
            lifecycle["after_cut"] = _timeline.is_after_cut(
                compile_result.project_id, arrived,
            )
        prov = provenance.get(fp.filename, {})
        # HOW WIDE this document is. A rate card speaks for the customer, a
        # programme breakdown for several deals; neither is wrong, but a rollup
        # inside one is never THIS deal's number. Detected from the document's
        # own content -- never its filename.
        _delivered_text = " ".join(
            str(d.get("text") or "") for d in ((lifecycle or {}).get("delivered") or [])
        )
        _scope_info = _scope.detect_scope(
            # EvidenceAtom exposes raw_text, not text. Reading the wrong
            # attribute silently yielded empty strings, so no deal keys were
            # found and every document scored `deal` -- the Sodexo Breakdown
            # came back scope=deal with no foreign keys while the same atoms
            # analysed directly gave scope=program, 33068 and 34150.
            texts=[a.raw_text for a in artifact_atoms if getattr(a, "raw_text", None)],
            document_type=(lifecycle or {}).get("type"),
            this_deal_keys=_this_keys,
            delivering_text=_delivered_text,
        )
        # Part 2: resolve each row to the deal it belongs to, so a row that IS
        # this deal's is admitted rather than demoted with the rest of a
        # multi-deal document.
        _scope_rows = _scope.narrow_rows(
            [
                {
                    "atom_type": getattr(getattr(a, "atom_type", None), "value", None),
                    "text": getattr(a, "raw_text", "") or "",
                    "locator": (a.source_refs[0].locator if getattr(a, "source_refs", None) else {}),
                }
                for a in artifact_atoms
            ],
            scope=_scope_info["scope"],
            this_deal_keys=_this_keys,
        ) if _scope_info["scope"] != _scope.SCOPE_DEAL else []
        _scope_summary = _scope.summarise(
            [
                {
                    "atom_type": getattr(getattr(a, "atom_type", None), "value", getattr(a, "atom_type", None)),
                    "text": getattr(a, "raw_text", "") or "",
                }
                for a in artifact_atoms
            ],
            scope=_scope_info["scope"],
            this_deal_keys=_this_keys,
        )
        # A document that states its own date outranks the time we uploaded
        # it. Live 010300: two PSOWs dated March 2025 carried authored_at
        # 2026-09-03 (HubSpot file time), so a cut would have called them
        # today's evidence. Only an upload-grade precision is overridden, and
        # only by an EARLIER header date; an exact send time is left alone.
        _hdr = _document_header_date(artifact_atoms, _page_one_text(project_dir, fp.filename))
        if _hdr and prov.get("authored_at_precision") in (None, "edited") and (
            not prov.get("authored_at") or str(_hdr) < str(prov.get("authored_at"))
        ):
            prov = dict(prov)
            prov["ingested_at"] = prov.get("authored_at")
            prov["authored_at"] = _hdr
            prov["authored_at_precision"] = "document_header"
        deal_stage = _deal_stage.annotate(
            lifecycle,
            authored_at=prov.get("authored_at"),
            direction=prov.get("direction"),
            timeline=stage_timeline,
        )
        documents.append(
            {
                "artifact_id": fp.artifact_id,
                "filename": fp.filename,
                "artifact_type": fp.artifact_type.value,
                "sha256": fp.sha256,
                "lifecycle": lifecycle,
                # The deal's own life, not the file's name: when this arrived,
                # which stage was in force, whose it is, and who may read it.
                "authored_at": prov.get("authored_at"),
                "authored_at_precision": prov.get("authored_at_precision"),
                "direction": prov.get("direction"),
                "sender_domain": prov.get("sender_domain"),
                "deal_stage": deal_stage,
                "scope": {
                    **_scope_info,
                    **_scope_summary,
                    **({"rows": _scope_rows,
                        "covers_deals": sorted({r["belongs_to"] for r in _scope_rows if r["belongs_to"]}),
                        "atoms_admitted": sum(1 for r in _scope_rows if r["verdict"] == "admit"),
                        "atoms_demoted": sum(1 for r in _scope_rows if r["verdict"] == "context"),
                        **({"misfiled": _mf} if (_mf := _scope.misfiled_verdict(_scope_rows, this_deal_keys=_this_keys)) else {})}
                       if _scope_rows else {}),
                },
                # The conversation this message belongs to. Threading already
                # runs as a compile stage and stamps every atom, but only the
                # atoms -- so a reader above atom level could not group 33 email
                # files into the 6 conversations they actually are.
                "email_thread": _document_thread(artifact_atoms),
                # Who the forwarded chain STARTED with -- claimed ONLY when this
                # message actually carried something.
                #
                # A reply quotes the whole history, so the deepest quoted sender
                # is present in every message of a thread. Reading it off any
                # message made a plain reply from Quinton report "forwarding
                # Bernie Donnelly", which is false: he introduced nothing, he
                # answered. The question "whose document is this?" only arises
                # for a message that brought a document.
                "originated_by": (
                    _originating_sender(artifact_atoms, artifact_id=fp.artifact_id)
                    if (prov.get("attachment_ids") or [])
                    else None
                ),
                "attachment_ids": prov.get("attachment_ids") or [],
                # A Deal Kit that belongs to another deal is how that deal's
                # pricing walks into this quote. Reported on the document so a
                # PM sees it where the file is, not in a separate report.
                **(
                    {"kit_health": _check_deal_kit(
                        atoms=[{"atom_type": getattr(getattr(a, "atom_type", None), "value", None),
                                "text": getattr(a, "raw_text", "")} for a in artifact_atoms],
                        deal_number=_dn,
                    )}
                    if (lifecycle or {}).get("type") == "DEAL_KIT" else {}
                ),
                "sender_email": prov.get("sender_email"),
                "size_bytes": fp.size_bytes,
                "parser_name": fp.parser_name,
                "parser_version": fp.parser_version,
                "structured": structured_projection,
                "atom_ids": sorted(a.id for a in artifact_atoms),
                # A6 graceful degradation: per-file parse outcome.
                # ``status`` is one of ok / ok_empty / skipped_no_parser
                # / failed_parse. PM_HANDOFF reads this to surface
                # files that the engineer should manually inspect.
                "parse_outcome": outcome_by_artifact.get(
                    fp.artifact_id,
                    {"status": "unknown", "atom_count": len(artifact_atoms), "warning_count": 0},
                ),
            }
        )

    summary = _build_summary(
        atoms=atoms,
        packets=packets,
        documents=documents,
        entities=entities,
        edges=edges,
    )
    crm = _load_manifest_crm(project_dir)
    if crm:
        summary["crm"] = crm
    foreign = _foreign_artifacts(crm=crm, documents=documents)
    if foreign:
        summary["foreign_artifacts"] = foreign
    # v57.3.5: filter site:* entities + redirect ghost atom keys
    # BEFORE building the indexes — because orbitbrief-core's cluster
    # builder reads from ``envelope.indexes.atoms_by_entity_key`` to
    # decide how many atoms each canonical site has. If we filter
    # entities AFTER the index is built, the index still maps ghost
    # keys to lots of atom_ids while the canonical keys only have the
    # 1 physical_site atom each — so canonical clusters fail the >2
    # atoms gate in orbitbrief-core and get dropped from the dossier.
    # Move filter ABOVE _build_indexes so the redirect propagates.
    entities = _filter_site_entities_against_physical_atoms(entities, atoms)
    indexes = _build_indexes(atoms=atoms, entities=entities, edges=edges)
    drawings = _build_drawings_section(
        atoms=atoms,
        packets=packets,
        edges=edges,
        atoms_by_artifact=atoms_by_artifact,
        documents=documents,
    )
    envelope: dict[str, Any] = {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "project_id": compile_result.project_id,
        "compile_id": compile_result.compile_id,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": summary,
        "documents": documents,
        "atoms": [_compact_atom(a) for a in atoms],
        "packets": [_compact_packet(p) for p in packets],
        "entities": [_compact_entity(e, atoms_by_artifact, atoms) for e in entities],
        "edges": [_compact_edge(edge) for edge in edges],
        "indexes": indexes,
        "coverage": {"unrecovered_regions": unrecovered_regions},
    }
    # When this deal committed to an answer, and the verified events that say so.
    # Present for every deal so a consumer can tell "no cut" (still in discovery,
    # everything is evidence) from "never classified" (we have no timeline at
    # all): ``known`` is the difference, and it is the only honest way to read a
    # null ``quote_asof``. Each event carries the sentence it was extracted from.
    _cut = _timeline.quote_asof(compile_result.project_id)
    _events = _timeline.events(compile_result.project_id)
    envelope["deal_timeline"] = {
        "known": bool(_events) or _cut is not None,
        "quote_asof": _cut,
        "events": _events,
        "documents_after_cut": sum(
            1 for d in documents
            if isinstance(d.get("lifecycle"), dict) and d["lifecycle"].get("after_cut")
        ),
    }
    # OrbitBrief-Core deliverables — deterministic pre-aggregations so
    # the downstream LLM synthesis layer (and the PM cockpit) can render
    # the Monday-morning view, the SOW-readiness scorecard, and the
    # required-fields checklist directly without re-scanning atoms.
    envelope["pm_dashboard"] = build_pm_dashboard(
        atoms=atoms, packets=packets, edges=edges, entities=entities,
    )
    envelope["sow_readiness_scorecard"] = build_sow_readiness_scorecard(
        atoms=atoms, packets=packets, edges=edges, entities=entities,
    )
    envelope["srl_missing_checklist"] = build_srl_missing_checklist(
        atoms=atoms, documents=documents,
    )
    if compile_result.compile_capabilities:
        envelope["compile_capabilities"] = compile_result.compile_capabilities
    # Facet dashboard sections — the 7 PM sections (WORK/SITE/COMMERCIAL/...)
    # assigned by the contrastive facet head (held-out 0.925). Guess-free: atoms the
    # head can't confidently place go to `uncategorized`. Added ONLY when enabled +
    # the head is present, so OFF -> the key is absent -> byte-identical envelope.
    from app.core.facets import build_facet_sections as _build_facet_sections

    _facet_sections = _build_facet_sections(atoms)
    if _facet_sections.get("enabled"):
        envelope["facet_sections"] = _facet_sections
    # Trained service-router head: classify the deal SCOPE into its primary
    # managed-service pack so brief-gen's pack_prior runs the right brain (a TV
    # install no longer routes to datacenter). Guess-free: abstains -> primary
    # null. Added ONLY when enabled + head present, so OFF -> key absent.
    from app.core.service_router import build_service_routing as _build_service_routing

    # ``project_id`` IS the deal key: a deal-scoped PM correction stores
    # ``scope_key = project_id`` (routes_feedback._scope_from_chip), so passing
    # it here is what lets a shadow observation and a later correction be
    # joined on the same deal.
    _routing = _build_service_routing(
        atoms,
        documents,
        deal_id=compile_result.project_id,
        project_id=compile_result.project_id,
        base=None,
        base_observed=False,
    )
    # Emitted whenever there is anything to say, not only when a head is
    # loaded. With the head off -- every environment today -- the key still
    # carries the ``candidates`` a correction chip needs to offer a choice, and
    # the recorded input, so the observation travels with the deal instead of
    # being thrown away on the one path that is always taken.
    #
    # Safe to widen: compute_output_signature hashes the CompileResult (atoms,
    # entities, edges, packets), not the envelope, so no signature moves.
    if _routing.get("enabled") or _routing.get("candidates"):
        envelope["service_routing"] = _routing
    # S+++++ cockpit surfaces — authority-weighted scope truth,
    # chronological change-order audit, per-site readiness rollup,
    # per-stakeholder workload matrix, and a single 0-100 project
    # vitals number that blends every signal above into one
    # auditable cockpit-header score.
    # Phase-2 reconciliation: every contradicts-cluster resolved by the
    # authority lattice with full receipts, or surfaced unresolved when the
    # top tier ties. A verdict layer over the evidence -- no atom is mutated,
    # so replay keeps verifying. Additive key; the output signature hashes
    # the CompileResult, not the envelope.
    from app.core.reconcile import build_reconciliation as _build_reconciliation

    envelope["reconciliation"] = _build_reconciliation(atoms, edges)
    envelope["scope_truth"] = build_scope_truth(atoms=atoms, edges=edges)
    envelope["change_order_timeline"] = build_change_order_timeline(atoms=atoms)
    envelope["site_readiness"] = build_site_readiness(atoms=atoms, edges=edges)
    # Does the number of sites we RESOLVED match the number the deal SAYS it has?
    #
    # On 010215 the emails stated "10" nine times and a quantity entity of 10 was
    # extracted and kept, while the site layer resolved six addresses -- four of
    # them two addresses fused, one truncated, three sites missing entirely. Both
    # facts sat in this envelope and nothing compared them, so a PM had no way to
    # know the site list was short until a technician stood at the wrong address.
    #
    # This does not correct the count. It refuses to let the contradiction pass
    # unremarked, which is the only honest thing to do when two parts of the same
    # evidence disagree.
    # Two documents asserting two different addresses for one site is not a
    # duplicate to merge — it is a disagreement to report. On 010215 the Academy
    # of Early Learning SOW says 600 E Northside Ave and the customer's own
    # locations list says 111 Academy St; eight of ten sites agree exactly and
    # two do not. Picking a winner silently is how a technician reaches the wrong
    # school while the system reports full confidence.
    try:
        from app.core.site_evidence_conflict import (
            find_address_collisions,
            find_site_address_conflicts,
        )

        _claims = []
        for _a in atoms:
            if str(getattr(getattr(_a, "atom_type", None), "value", getattr(_a, "atom_type", "")))\
                    .endswith("physical_site"):
                _v = getattr(_a, "value", None) or {}
                if not isinstance(_v, dict):
                    continue
                _refs = getattr(_a, "source_refs", None) or []
                _claims.append({
                    "name": _v.get("name") or getattr(_a, "raw_text", ""),
                    "address": _v.get("address"),
                    "source": getattr(_refs[0], "filename", None) if _refs else None,
                    "authored_at": None,
                })
        # Both directions of the same disagreement: one site with two addresses,
        # and one address with two sites. Dedup no longer merges the second case
        # (distinct names veto it), so without this the collision would survive
        # into the envelope with nothing saying the two schools collide.
        _conf = list(find_site_address_conflicts(_claims))
        _conf += find_address_collisions(_claims)
        if _conf:
            envelope["site_evidence_conflicts"] = _conf
    except Exception as _sec_exc:
        import logging as _lg_sec

        _lg_sec.getLogger(__name__).warning("site_evidence_conflict failed: %s", _sec_exc)

    try:
        from app.core.site_count_reconcile import reconcile_site_count

        _sr = envelope.get("site_readiness") or {}
        envelope["site_count_reconciliation"] = reconcile_site_count(
            atoms, int(_sr.get("site_count") or 0)
        )
    except Exception as _scr_exc:  # never fail a compile over a cross-check
        import logging as _lg_scr

        _lg_scr.getLogger(__name__).warning("site_count_reconcile failed: %s", _scr_exc)

    # v49 FIX 6: enrich site_readiness rows with structured attributes
    # (address, mdf_idf, access_window, escort, users, rooms, notes,
    # aliases) from physical_site atoms. These come from the v48 site
    # roster extractor and v49 docx schema registry and are the single
    # source of truth for site metadata.
    try:
        import re as _re_v49
        def _atom_type_str(_a) -> str:
            _at = getattr(_a, "atom_type", None)
            return _at.value if hasattr(_at, "value") else str(_at or "")
        _sr = envelope.get("site_readiness") or {}
        _sites_list = _sr.get("sites") or []
        # site_readiness.sites is a LIST of dicts keyed by "site" field
        _by_slug: dict[str, dict] = {}
        for _entry in _sites_list:
            if isinstance(_entry, dict):
                _k = _entry.get("site") or _entry.get("site_key") or ""
                if _k:
                    _by_slug[_k] = _entry
        for _atom in atoms:
            if _atom_type_str(_atom) != "physical_site":
                continue
            _val = getattr(_atom, "value", None) or {}
            if not isinstance(_val, dict):
                continue
            _sid = _val.get("id") or _val.get("site_id") or ""
            if not _sid:
                continue
            _slug = f"site:{_re_v49.sub(r'[^a-z0-9]+', '_', _sid.lower()).strip('_')}"
            _entry = _by_slug.get(_slug)
            if _entry is None:
                continue
            for _attr in ("address", "mdf_idf", "access_window", "escort", "users", "rooms", "notes",
                          "facility_name", "street_address", "escort_owner", "contact", "phone", "email"):
                _v = _val.get(_attr)
                if _v and not _entry.get(_attr):
                    _entry[_attr] = _v
            _names = _val.get("names") or []
            if _names:
                _aliases = _entry.setdefault("aliases", [])
                for _n in _names:
                    if _n and _n not in _aliases:
                        _aliases.append(_n)
    except Exception as _v49_exc:
        import logging as _lg_v49
        _lg_v49.getLogger(__name__).warning("v49 site attribute passthrough failed: %s", _v49_exc)

    # The passthrough above can only name a site a physical_site atom
    # anchors. Sites the deal mentions only in prose become rows with no
    # name at all, so the dossier shows the PM ``site:prudential_center_
    # office`` where the document said "Prudential Center office". Recover
    # the readable form from evidence the compile already holds. Fills
    # blanks only — an all-anchored deal (Clayton's 437 roster sites) does
    # no work here and is left byte-identical.
    try:
        from app.core.site_naming import recover_site_display_names
        _sr_rows = ((envelope.get("site_readiness") or {}).get("sites") or [])
        _recovered = recover_site_display_names(
            sites=_sr_rows, atoms=atoms, documents=documents,
        )
        for _row in _sr_rows:
            _name = _recovered.get(_row.get("site") or "")
            if not _name:
                continue
            _row["facility_name"] = _name
            # Anchored rows carry their name in ``aliases`` too (the
            # passthrough copies ``value.names`` in). Match that shape so
            # every consumer sees prose sites the same way as roster ones.
            _row_aliases = _row.setdefault("aliases", [])
            if _name not in _row_aliases:
                _row_aliases.insert(0, _name)
        if _recovered:
            import logging as _lg_naming_ok
            _lg_naming_ok.getLogger(__name__).info(
                "site_naming recovered %d display name(s): %s",
                len(_recovered), ", ".join(sorted(_recovered)),
            )
    except Exception as _naming_exc:
        import logging as _lg_naming
        _lg_naming.getLogger(__name__).warning(
            "site display-name recovery failed: %s", _naming_exc
        )

    envelope["stakeholder_load"] = build_stakeholder_load(atoms=atoms)

    # Deal header / financials / BOM — PM-facing assembly of the
    # structured commercial atoms the xlsx parser emits. Each is omitted
    # when the deal carries no such data, so the envelope shape stays
    # stable for non-commercial projects. Never fatal.
    try:
        _deal_header = build_deal_header(atoms=atoms)
        if _deal_header.get("present"):
            envelope["deal_header"] = _deal_header
        _deal_financials = build_deal_financials(atoms=atoms)
        if _deal_financials.get("present"):
            envelope["deal_financials"] = _deal_financials
        _bom = build_bill_of_materials(atoms=atoms)
        if _bom.get("present"):
            envelope["bill_of_materials"] = _bom
    except Exception as _deal_exc:
        import logging as _lg_deal
        _lg_deal.getLogger(__name__).warning("deal section build failed: %s", _deal_exc)

    # Gap F — Truth Gate: grade every entity by independent-source
    # corroboration so single-sourced facts are visibly distinct from
    # facts three documents agree on. Deterministic, never fatal.
    try:
        from app.core.truth_gate import build_truth_gate
        envelope["truth_gate"] = build_truth_gate(
            atoms=atoms, entities=entities, edges=edges,
        )
    except Exception as _tg_exc:
        import logging as _lg_tg
        _lg_tg.getLogger(__name__).warning("truth_gate build failed: %s", _tg_exc)
        envelope["truth_gate"] = {}

    envelope["project_vitals"] = build_project_vitals(
        atoms=atoms,
        edges=edges,
        packets=packets,
        scorecard=envelope["sow_readiness_scorecard"],
        checklist=envelope["srl_missing_checklist"],
        site_readiness=envelope["site_readiness"],
        stakeholder_load=envelope["stakeholder_load"],
        scope_truth=envelope["scope_truth"],
    )
    # Drawings section is omitted entirely on non-schematic projects so
    # the envelope shape stays byte-identical for the existing test grid.
    if drawings["artifacts"]:
        envelope["drawings"] = drawings
    # CRM context (when the parser-manifest sidecar carries it) is
    # exposed at the top of the envelope so downstream consumers can
    # render deal name / opportunity ID / amount without re-reading
    # the manifest blob.
    if crm:
        envelope["crm"] = crm
    # What THIS run was asked for, so a consumer can distinguish a deliberately
    # cut corpus from a full one. `applied` is the discriminator: a null cutoff
    # on a full run and a null cutoff because nobody recorded one look the same
    # otherwise, which is the failure this field exists to end.
    _run_cutoff = _load_manifest_run_cutoff(project_dir)
    envelope["run_scope"] = {
        "as_of": _run_cutoff,
        "applied": _run_cutoff is not None,
        "documents_in_scope": len(documents),
    }
    _resolve_delivered_by(documents)
    # Must follow _resolve_delivered_by: the originator is what the inference
    # reads, and it does not exist until delivery has been matched.
    _direction_from_originator(documents, stage_timeline)
    _link_caption_notes(documents, envelope, stage_timeline)
    try:
        from app.core.document_parties import annotate_document_parties
        _page_texts = {
            str(d.get("artifact_id")): _page_one_text(project_dir, str(d.get("filename") or ""))
            for d in documents
            if str(d.get("filename") or "").lower().endswith(".pdf")
        }
        annotate_document_parties(documents, envelope, _page_texts)
    except Exception as _exc:  # pragma: no cover - never fail the envelope
        envelope.setdefault("warnings", []).append(f"document_parties failed: {type(_exc).__name__}: {_exc}")
    # Last, so it gates the corrected picture: a customer document that stopped
    # being called ours a moment ago must be readable by the models that had it.
    _annotate_reader_scope(documents, stage_timeline)
    _annotate_quoted_message_scope(envelope, documents, stage_timeline)
    _retype_produced_material_scope(envelope, documents)
    threads = _thread_index(documents)
    if threads:
        envelope["email_threads"] = threads
        _enrich_atom_threads(envelope.get("atoms") or [], threads)
    return envelope


#: A PurTera deal number as it prefixes an artifact filename:
#: ``010129-hs-email-...``, ``000116 - GHA -Thyssenkrupp...``, ``010162  Deal Kit.xlsx``.
#: Anchored deliberately — an unanchored six-digit search pulls numbers out of
#: UUID fragments (``...b878374bd7c1``) and screenshot stamps
#: (``Screenshot 2026-08-17 150656``), which is noise, not identity.
_ARTIFACT_DEAL_NUM_RE = re.compile(r"^(\d{6})\s*[-_\s]")

#: How far apart two deal numbers must be before the artifact is treated as
#: foreign. Adjacent numbers are overwhelmingly the same customer and project —
#: "010143 WSS Presbyterian" holding "010142 WSS Presbyterian", a renumber or a
#: sibling survey/install pair. Across the corpus 60 of 79 number mismatches
#: were adjacent like that; flagging them would make the signal 76% false and
#: teach everyone to ignore it. Only distant numbers indicate a document filed
#: against the wrong deal.
_FOREIGN_DEAL_DISTANCE = 2


def _deal_number_from_crm(crm: Mapping[str, Any] | None) -> str | None:
    """The deal's own number, taken from the CRM deal name it is filed under.

    ``context.crm.deal_name`` reads ``"010114 - CDW Checkout Wireless Wifi"``,
    so the number prefixes it. This is authoritative: it comes from the deal
    record rather than from the documents being checked.
    """
    if not isinstance(crm, Mapping):
        return None
    m = _ARTIFACT_DEAL_NUM_RE.match(str(crm.get("deal_name") or "").strip())
    return m.group(1) if m else None


def _foreign_artifacts(
    *, crm: Mapping[str, Any] | None, documents: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Artifacts on this deal whose filename claims a different deal.

    Nothing anywhere checked this, and the failure is silent and total: one deal
    held five artifacts belonging to three other opportunities — Marco's Pizza
    and Vodafone — and OrbitBrief produced a fluent, confident brief about
    Marco's Pizza under a GHA Technologies deal name. A PM reading it has no
    signal that anything is wrong. Incomplete is recoverable; confidently about
    the wrong company is not.

    Reported, never dropped. A related document is sometimes filed on purpose,
    and deleting evidence on a filename heuristic would be its own bug.
    """
    own = _deal_number_from_crm(crm)
    if not own:
        return []
    out: list[dict[str, Any]] = []
    for doc in documents or []:
        if not isinstance(doc, Mapping):
            continue
        name = str(doc.get("filename") or "").strip()
        m = _ARTIFACT_DEAL_NUM_RE.match(name)
        if not m or m.group(1) == own:
            continue
        if abs(int(m.group(1)) - int(own)) <= _FOREIGN_DEAL_DISTANCE:
            continue
        out.append({
            "filename": name,
            "claims_deal": m.group(1),
            "deal_number": own,
            "artifact_id": doc.get("artifact_id"),
            # Distance alone does not separate "another opportunity for the
            # same customer" from "a different company's project". Deal
            # 010106 "Dollar Tree DC7 WAP" holding 010091 "Dollar Tree DC7 WAP
            # Install" is a sibling; deal 010129 "GHA Assa Abbloy" holding
            # "Marcos New Store Installs" is not. When the deal's own account
            # name shows up in the filename, say so, so the reader can tell a
            # shared document from a misfiled one without opening it.
            "account_match": _account_match(crm, name),
        })
    return out


#: Words too generic to identify a customer by.
_ACCOUNT_STOPWORDS = frozenset({
    "technologies", "technology", "inc", "llc", "corp", "company", "group",
    "solutions", "services", "systems", "install", "installation", "project",
})


def _names_from_crm(crm: Mapping[str, Any]) -> list[str]:
    """Customer names worth recognising in a filename.

    Both the account and the deal name matter, because the account is often the
    reseller while the document names the end customer. Deal 010128's account
    is "CentricsIT" but its name is "010128 - CentricsIT Marcos - MOMS POS
    Installation", and the shared runbook on it is called "010013 Marcos New
    Store Installs" — recognisable from the deal name, invisible from the
    account alone.
    """
    out = [str(crm.get("account_name") or "")]
    deal = str(crm.get("deal_name") or "")
    # Drop the leading number and keep the descriptive half.
    out.append(re.sub(r"^\d{6}\s*[-_\s]\s*", "", deal))
    # 3 is deliberate: plenty of accounts are initialisms (CDW, GHA, SHI)
    # and dropping them made every one of their documents look foreign.
    return [n.strip().lower() for n in out if len(n.strip()) >= 3]


#: Filename furniture that names no customer: document kinds, version and date
#: markers, and the boilerplate every deal folder repeats.
_FILENAME_FURNITURE = frozenset({
    "deal", "kit", "cost", "breakdown", "quote", "quotation", "estimate",
    "final", "draft", "copy", "signed", "redlines", "redline", "notes", "note",
    "scope", "work", "statement", "proposal", "summary", "photos", "photo",
    "image", "images", "screenshot", "survey", "install", "installation",
    "rollout", "swap", "upgrade", "refresh", "migration", "docx", "xlsx",
    "pdf", "pptx", "jpeg", "email", "note", "attachment", "version",
})


def _distinctive_words(text: str) -> list[str]:
    """Words in a filename that could plausibly name a customer."""
    words = re.findall(r"[a-z]{4,}", text.lower())
    return [
        w for w in words
        if w not in _ACCOUNT_STOPWORDS
        and w not in _FILENAME_FURNITURE
        and not re.fullmatch(r"v\d+|\d+", w)
    ]


def _account_match(crm: Mapping[str, Any] | None, filename: str) -> str:
    """``same`` | ``different`` | ``unknown`` — whose customer this file names.

    The two-state version conflated "names a different customer" with "names no
    customer at all", and most misfiled documents are called ``Deal Kit.xlsx``.
    Five of seven flagged rows were generic filenames of that shape, reported as
    a different customer purely because the host customer's name was absent —
    which it was always going to be. Absence of a name is not evidence.
    """
    if not isinstance(crm, Mapping):
        return "unknown"
    # Underscores and hyphens are filename dress: "_" is a word character, so
    # the word-boundary search below could never see CDW inside "CDW_Quote".
    hay = re.sub(r"[_\-]+", " ", filename.lower())
    for name in _names_from_crm(crm):
        # Word-boundary, not substring: a three-letter account like CDW would
        # otherwise match inside an unrelated word.
        if re.search(r"\b" + re.escape(name) + r"\b", hay):
            return "same"
        words = [w for w in re.findall(r"[a-z]{4,}", name) if w not in _ACCOUNT_STOPWORDS]
        # Any distinctive word carrying over is enough: "Marcos" links
        # "CentricsIT Marcos - MOMS POS" to "Marcos New Store Installs".
        if words and any(w in hay for w in words):
            return "same"
    # Strip the leading deal number before asking whether anything is left that
    # could be a customer name at all.
    stem = re.sub(r"^\d{6}\s*[-_\s]*", "", filename)
    stem = re.sub(r"\.[a-z0-9]{2,5}$", "", stem, flags=re.IGNORECASE)
    return "different" if _distinctive_words(stem) else "unknown"





def _enrich_atom_threads(atoms: list[dict[str, Any]], threads: list[dict[str, Any]]) -> None:
    """Give every email atom the whole conversation, not just its own message.

    email_threading.py already stamps each atom with its position and the gist
    of the message it replies to. That answers "what is this a reply to". It does
    not answer "what is this conversation, who is in it, and is this the last
    word" -- which is what a reader needs to weigh a single line like
    "Yes, approved, go ahead with 36".

    Thread-level facts only, all deterministic: the conversation's name, who
    took part, when it ran, and whether this message is the latest in it. No
    summary is invented -- a generated gist of twenty messages would be an
    unfalsifiable claim sitting in the evidence set, which is the one thing this
    pipeline must not produce.

    Mutates in place; additive, so an atom that was never threaded is untouched.
    """
    by_id = {t["thread_id"]: t for t in threads}
    if not by_id:
        return
    for atom in atoms:
        structured = atom.get("structured")
        if not isinstance(structured, dict):
            continue
        block = structured.get("email_thread")
        if not isinstance(block, dict):
            continue
        thread = by_id.get(block.get("thread_id"))
        if not thread:
            continue
        block["thread_name"] = thread.get("name")
        block["participants"] = thread.get("participants") or []
        block["thread_first_message_at"] = thread.get("first_message_at")
        block["thread_last_message_at"] = thread.get("last_message_at")
        # "Is this the last word on it?" -- an approval that was later revised
        # reads very differently from one nobody answered.
        date = block.get("date")
        last = thread.get("last_message_at")
        block["is_latest_in_thread"] = bool(date and last and str(date) >= str(last))
        if thread.get("looks_split_with"):
            block["thread_looks_split_with"] = thread["looks_split_with"]



_EMAIL_ADDR_RE = re.compile(r"[A-Za-z0-9._%%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


_HEADER_DATE_LABEL_RE = re.compile(r"(?:^|\|)\s*(?:col_\d+:\s*)?(?:date|dated|effective date|issue date|revision date)\s*:", re.I | re.M)
_DATE_VALUE_RE = re.compile(
    r"(?:\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{4}"
    r"|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b)",
    re.I,
)


_PAGE_ONE_TEXT_CACHE: dict[str, str] = {}


def _page_one_text(project_dir: Any, filename: str) -> str:
    """Text of a PDF's first page from its own text layer (best effort, cached).

    The parser's structured projection drops the header table's cells once it
    recovers them as a table, so "Date: March 21, 2025" never reaches an atom.
    The file itself still says it.
    """
    key = f"{project_dir}::{filename}"
    if key in _PAGE_ONE_TEXT_CACHE:
        return _PAGE_ONE_TEXT_CACHE[key]
    text = ""
    try:
        if str(filename).lower().endswith(".pdf"):
            try:
                import pymupdf as fitz  # PyMuPDF
            except Exception:  # pragma: no cover - older wheel
                import fitz  # type: ignore
            root = Path(project_dir)
            cand = next(iter(root.rglob(filename)), None) if filename else None
            if cand is not None and cand.is_file():
                with fitz.open(str(cand)) as doc:
                    if len(doc):
                        text = doc[0].get_text() or ""
                        if len(text.strip()) < 40:
                            # No text layer (a scanned PSOW): read the page the
                            # way the parser does, at OCR resolution.
                            try:
                                from app.parsers._ocr_chain import ocr_pdf_page
                                text = str((ocr_pdf_page(doc[0]) or {}).get("text") or "")
                            except Exception:
                                pass
    except Exception:
        text = ""
    _PAGE_ONE_TEXT_CACHE[key] = text
    return text


def _document_header_date(artifact_atoms: list[Any], page_text: str | None = None) -> str | None:
    """The date a document states about itself, as ISO ``YYYY-MM-DD``, or None.

    Read from the first page's labelled fields ("Date: | March 21, 2025",
    "Date: March 05, 2025"): a label that IS a date label and a value that
    parses as a date. Signature dates, delivery dates and dates in prose are
    not the document's own date and are not read here. ``page_text`` is the
    first page's own text, consulted the same way when given.
    """
    try:
        from dateutil import parser as _dp
    except Exception:  # pragma: no cover
        return None
    best: str | None = None
    if page_text:
        for m in _HEADER_DATE_LABEL_RE.finditer(page_text):
            window = page_text[m.end(): m.end() + 120]
            vals = _DATE_VALUE_RE.findall(window)
            if not vals:
                continue
            try:
                dt = _dp.parse(vals[0].strip(), fuzzy=True, default=datetime(1900, 1, 1))
            except Exception:
                continue
            if 1990 <= dt.year <= 2100:
                iso = dt.strftime("%Y-%m-%d")
                if best is None or iso < best:
                    best = iso
        if best is None:
            # No readable label (OCR turned "Date:" into "foe:" on a scanned
            # PSOW) but the header BLOCK -- the first lines, before the body
            # prose -- holds exactly one full date standing on its own line.
            # A header field's value is the date the document states.
            head_lines = [ln.strip() for ln in page_text.splitlines()[:24]]
            own = [ln for ln in head_lines if _DATE_VALUE_RE.fullmatch(ln)]
            if len(own) == 1:
                try:
                    dt = _dp.parse(own[0], fuzzy=True, default=datetime(1900, 1, 1))
                    if 1990 <= dt.year <= 2100:
                        best = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
    for a in artifact_atoms or []:
        refs = getattr(a, "source_refs", None) or []
        loc = getattr(refs[0], "locator", None) if refs else None
        page = (loc or {}).get("page") if isinstance(loc, dict) else None
        if page not in (None, 0, 1):
            continue
        text = str(getattr(a, "raw_text", "") or "").replace("\n", " ")
        m = _HEADER_DATE_LABEL_RE.search(text)
        if not m:
            continue
        # The value is the LAST date-shaped token after the label: a table row
        # serialises as "col_0: Date: | <column heading>: March 21, 2025".
        vals = _DATE_VALUE_RE.findall(text[m.end():])
        if not vals:
            continue
        raw = vals[-1].strip()
        try:
            dt = _dp.parse(raw, fuzzy=True, default=datetime(1900, 1, 1))
        except Exception:
            continue
        if dt.year < 1990 or dt.year > 2100:
            continue
        iso = dt.strftime("%Y-%m-%d")
        if best is None or iso < best:
            best = iso
    return best


def _annotate_quoted_message_scope(
    envelope: dict[str, Any],
    documents: list[dict[str, Any]],
    timeline: Mapping[str, Any] | None,
) -> int:
    """Give a QUOTED message from an external author its own reader scope.

    Admissibility is decided per file, but an email is many authors: a reply
    we sent carries, quoted, the message it answers. Live 010300: the
    customer's entire ask ("170+ sites and growing", "Yealink phones on
    Nextiva", "two separate projects", "an A+ PM") existed only inside two
    OUTBOUND replies, filed `label`, and the Deal Kit could not see it.

    For each atom whose line came from a quoted block with an external
    author (the block's own From:, stamped by the email parser), re-decide
    admissibility as INBOUND at the carrier's stage and write the result on
    the atom as ``reader_scope`` plus ``decision_provenance``. Consumers that
    read scope off the document keep doing so; one that honours the atom's
    own scope (deal-ask) sees the customer's words. Internal authors are
    left alone: quoting ourselves does not make our words the customer's.

    Returns how many atoms were annotated.
    """
    from app.core.document_lifecycle import reader_scope as _rs
    from app.core.internal_author import extract_email_domain, INTERNAL_EMAIL_DOMAINS

    by_id = {str(d.get("artifact_id")): d for d in documents or []}
    atoms_all = envelope.get("atoms") or []

    def _name_stem(text: str) -> str:
        toks = [t for t in re.split(r"[^A-Za-z]+", str(text or "")) if t]
        return " ".join(t.lower() for t in toks[:2]) if len(toks) >= 2 else ""

    def _resolve_author(author: str, artifact_id: str, message_index: Any) -> str | None:
        """A quoted From: often shows only a display name ("From: Carl
        Painter Jr") -- Outlook drops the address for a sender in the same
        directory. Resolve it to an address the envelope already knows:
        first the signature read from the SAME quoted message (it carries the
        email), then any email document whose sender or signature matches the
        name. Live 010300: 4 of 5 quoted customer lines were skipped for this."""
        if extract_email_domain(author):
            return author
        stem = _name_stem(author)
        if not stem:
            return None
        same_msg, any_doc = [], []
        for b in atoms_all:
            if b.get("atom_type") != "stakeholder":
                continue
            sb = b.get("structured") if isinstance(b.get("structured"), dict) else {}
            em = str(sb.get("email") or "").strip()
            if not em or _name_stem(sb.get("name")) != stem:
                continue
            if str(b.get("artifact_id")) == artifact_id and sb.get("message_index") == message_index:
                same_msg.append(em)
            else:
                any_doc.append(em)
        return (same_msg or any_doc or [None])[0]

    n = 0
    for a in atoms_all:
        s = a.get("structured") if isinstance(a.get("structured"), dict) else {}
        if not s.get("quoted"):
            continue
        author = str(s.get("author") or "").strip()
        resolved = _resolve_author(author, str(a.get("artifact_id")), s.get("message_index")) if author else None
        domain = extract_email_domain(resolved) if resolved else None
        if not domain or domain in INTERNAL_EMAIL_DOMAINS:
            continue
        if resolved != author:
            s["author_resolved"] = resolved
        doc = by_id.get(str(a.get("artifact_id"))) or {}
        block = doc.get("deal_stage") if isinstance(doc.get("deal_stage"), dict) else {}
        stage = block.get("stage_at_arrival")
        adm, why = _deal_stage.admissibility(
            stage=stage, direction="inbound", classified_as=None, timeline=timeline,
        )
        if not adm:
            continue
        scope: dict[str, Any] = {}
        for consumer in _rs.consumers():
            ok, cwhy = _rs.visible_to(consumer, stage=stage, admissible_for=adm, timeline=timeline)
            scope[consumer] = {"visible": ok, "why": cwhy}
        doc_scope = doc.get("reader_scope") or {}
        if all((doc_scope.get(c) or {}).get("visible") == v.get("visible") for c, v in scope.items()):
            continue  # same answer as the file; nothing to override
        a["reader_scope"] = scope
        prov = dict(a.get("decision_provenance") or {})
        prov.update({
            "source": "quoted_message",
            "author": author,
            "admissible_for": adm,
            "rationale": f"quoted message authored by {domain}, decided as inbound: {why}",
        })
        a["decision_provenance"] = prov
        n += 1
    return n


def _retype_produced_material_scope(envelope: dict[str, Any], documents: list[dict[str, Any]]) -> int:
    """scope_item from a document the lifecycle filed as produced/vendor
    material becomes site_implementation_note.

    The Kronos install instructions (admissible_for=atlas) contributed 99
    scope_item atoms: "Route the power supply cable through the clamps" is a
    vendor's procedure step, not something the customer asked us to do. The
    lifecycle had already made the document-level call; the atom type just
    never followed it. Reads the decision it already made -- no vocabulary.
    """
    # Live 010215: the Kronos instructions arrived INBOUND before quoting, so
    # the stage x direction rule (correctly) filed them as `evidence` -- they
    # are what we quote FROM -- and an `admissible_for == atlas` test never
    # fired. The signal that survives that re-decision is the lifecycle's own
    # taxonomy verdict: stage DELIVERY / CLOSEOUT (install instructions,
    # runbooks, checklists, as-builts) and scope `global` (a standing reference,
    # not deal-specific by nature). Either says "procedure", whatever stage the
    # deal was in when it arrived.
    produced: dict[str, str] = {}
    for d in documents or []:
        aid = str(d.get("artifact_id"))
        lc = d.get("lifecycle") if isinstance(d.get("lifecycle"), dict) else {}
        ds = d.get("deal_stage") if isinstance(d.get("deal_stage"), dict) else {}
        if ds.get("admissible_for") == "atlas":
            produced[aid] = "document filed as atlas (produced/vendor material); its rows are procedure, not customer scope"
        elif str(lc.get("stage") or "").upper() in ("DELIVERY", "CLOSEOUT"):
            produced[aid] = f"lifecycle type {lc.get('type') or '?'} (stage {lc.get('stage')}): vendor/delivery material; its rows are procedure, not customer scope"
        elif str(((d.get("scope") or {}) if isinstance(d.get("scope"), dict) else {}).get("scope") or "") == "global":
            produced[aid] = "standing reference (global scope, not deal-specific); its rows are procedure, not customer scope"
    if not produced:
        return 0
    n = 0
    for a in envelope.get("atoms") or []:
        aid = str(a.get("artifact_id"))
        if aid in produced and a.get("atom_type") == "scope_item":
            a["atom_type"] = "site_implementation_note"
            prov = dict(a.get("decision_provenance") or {})
            prov.update({"source": "produced_material", "rationale": produced[aid]})
            a["decision_provenance"] = prov
            n += 1
    return n


def _direction_from_originator(
    documents: list[dict[str, Any]],
    timeline: Mapping[str, Any] | None = None,
) -> int:
    """Give a FILE the direction of whoever originated it, and re-decide admissibility.

    A file carries no direction of its own -- only email and notes do -- so with
    none available the classifier falls back to the name. On deal 010215 that
    read "SOW Smarthands Marion County SD ..." and returned ``label``: material
    we produced. Those ten documents are the CUSTOMER's, one per school, sent by
    Bernie Donnelly at Sodexo and forwarded in by Trent.

    That is not a cosmetic mislabel. ``admissibility`` makes type decisive over
    stage on purpose, so our own output can never be readmitted as evidence --
    which means a Deal Kit model would be denied the exact ten documents the
    real Deal Kit was built from, and the gate would look like it was working.

    The side is inferred from the ORIGINATOR, never from HubSpot's `direction`.
    HubSpot's flag is about a message's relationship to the deal record, not
    about who wrote it: the message carrying these was marked INCOMING while
    sent from our own address. The person at the top of the quoted chain is a
    fact; the flag is not.

    Only external -> ``inbound`` is asserted. An internal originator is left
    alone: we may be forwarding someone else's material, which is precisely the
    case here, and calling that "ours" would repeat the error in the other
    direction.

    Returns how many documents were re-decided.
    """
    from app.core.internal_author import extract_email_domain, INTERNAL_EMAIL_DOMAINS

    changed = 0
    for doc in documents:
        if doc.get("direction"):
            continue
        origin = doc.get("delivered_by")
        if not origin:
            continue
        domain = extract_email_domain(str(origin))
        if not domain or domain in INTERNAL_EMAIL_DOMAINS:
            continue

        doc["direction"] = "inbound"
        doc["direction_source"] = "originator of the delivering message"

        block = doc.get("deal_stage")
        if not isinstance(block, dict):
            continue
        # Re-decide with the direction we now have. `classified_as` is reset to
        # None deliberately: the name-based guess is what put the customer's
        # documents in `label`, and keeping it would win over the stage rule
        # again by the same route.
        adm, why = _deal_stage.admissibility(
            stage=block.get("stage_at_arrival"),
            direction="inbound",
            classified_as=None,
            timeline=timeline,
        )
        if adm and adm != block.get("admissible_for"):
            block["admissible_for"] = adm
            block["why"] = f"{why}; direction from {domain}, who originated it"
            block["changed_from_classifier"] = True
            lifecycle = doc.get("lifecycle")
            if isinstance(lifecycle, dict):
                lifecycle["admissible_for"] = adm
            changed += 1
    return changed


_CAPTION_WINDOW_SEC = 180


def _link_caption_notes(
    documents: list[dict[str, Any]],
    envelope: dict[str, Any],
    timeline: Mapping[str, Any] | None,
) -> int:
    """A HubSpot note written within seconds of a file upload captions it.

    A file uploaded to the deal has no sender and no direction, so the
    lifecycle cannot place it (live 010300: the Teams PSOW sat with
    ``admissible_for: None``). The person who uploaded it usually leaves a
    note at the same moment ("psow from current partner", 13 seconds before
    the file). Timing plus a caption-shaped body (short, no digits) links the
    two: the note's author delivered the file, the file takes the note's
    direction, and admissibility is re-decided with that direction. Only files
    with NO provenance are touched; a file an email delivered keeps that.

    Returns how many files were linked.
    """
    from datetime import datetime as _dt

    def _ts(v: Any) -> float | None:
        s = str(v or "").strip()
        if not s:
            return None
        try:
            return _dt.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except Exception:
            return None

    meta_by_doc: dict[str, dict[str, Any]] = {}
    for a in envelope.get("atoms") or []:
        s = a.get("structured") if isinstance(a.get("structured"), dict) else {}
        if s.get("source") != "hubspot_note" and s.get("field_name") not in ("hubspot_note_meta", "hubspot_note_provenance"):
            continue
        cur = meta_by_doc.setdefault(str(a.get("artifact_id")), {})
        for k in ("title", "author", "author_email"):
            if s.get(k) and not cur.get(k):
                cur[k] = s.get(k)
    notes = []
    for d in documents or []:
        aid = str(d.get("artifact_id"))
        meta = meta_by_doc.get(aid)
        if not meta:
            continue
        title = str(meta.get("title") or "").strip()
        if not title or len(title.split()) > 8 or re.search(r"\d|\$", title):
            continue  # a caption is short and says nothing numeric
        when = _ts(d.get("authored_at"))
        if when is None:
            continue
        notes.append((when, d, meta))
    if not notes:
        return 0
    n = 0
    for d in documents or []:
        if d.get("direction") or d.get("delivered_by"):
            continue
        if str(d.get("artifact_type") or "") in ("email", "txt", "md"):
            continue
        when = _ts(d.get("ingested_at") or d.get("authored_at"))
        if when is None:
            continue
        best = min(notes, key=lambda t: abs(t[0] - when))
        if abs(best[0] - when) > _CAPTION_WINDOW_SEC:
            continue
        _, note_doc, meta = best
        author = str(meta.get("author_email") or meta.get("author") or note_doc.get("sender_email") or "").strip()
        direction = str(note_doc.get("direction") or "").strip() or None
        if not author and not direction:
            continue  # nothing to inherit
        d["delivered_by"] = author or None
        d["delivered_by_source"] = "caption_note"
        d["caption"] = str(meta.get("title") or "")
        d["caption_note"] = note_doc.get("artifact_id")
        if direction:
            d["direction"] = direction
            d["direction_source"] = "caption note written within seconds of the upload"
            block = d.get("deal_stage")
            if isinstance(block, dict):
                adm, why = _deal_stage.admissibility(
                    stage=block.get("stage_at_arrival"), direction=direction,
                    classified_as=None, timeline=timeline,
                )
                if adm and adm != block.get("admissible_for"):
                    block["admissible_for"] = adm
                    block["why"] = f"{why}; direction from the note that captions this upload ({author or 'unknown author'})"
                    block["changed_from_classifier"] = True
                    lifecycle = d.get("lifecycle")
                    if isinstance(lifecycle, dict):
                        lifecycle["admissible_for"] = adm
        n += 1
    return n


def _annotate_reader_scope(
    documents: list[dict[str, Any]],
    timeline: Mapping[str, Any] | None,
) -> None:
    """Record, per document, which models were allowed to read it.

    Training a Deal Kit model on a finished deal is only honest if it sees what
    the person saw. A document that arrived after the kit was produced teaches
    the model from its own answer -- and on a manually-worked corpus that is the
    default outcome unless something stops it.

    Written onto the document rather than computed by each consumer, so the
    answer is one thing a person can audit rather than several that can drift.
    """
    from app.core.document_lifecycle import reader_scope as _rs

    for doc in documents:
        block = doc.get("deal_stage") or {}
        stage = block.get("stage_at_arrival")
        adm = block.get("admissible_for") or (doc.get("lifecycle") or {}).get("admissible_for")
        seen: dict[str, Any] = {}
        for consumer in _rs.consumers():
            ok, why = _rs.visible_to(
                consumer, stage=stage, admissible_for=adm, timeline=timeline
            )
            seen[consumer] = {"visible": ok, "why": why}
        doc["reader_scope"] = seen


def _resolve_delivered_by(documents: list[dict[str, Any]]) -> None:
    """Say WHO sent the message that delivered each file.

    A file carries no sender of its own. The lifecycle work recovered the
    message that delivered it, but only as ``{kind, text, ts}`` -- so the UI
    could say *which* email brought a document and not who wrote it, which is
    the difference between a SOW draft they sent and one we sent.

    The delivering message is itself a document in this envelope, and email
    documents carry a sender. Match on TIMESTAMP, which is exact and unique to
    the minute, rather than on subject text, which drifts across a thread.

    Falls back to the first address in the delivered text -- these are forwards
    and the sender's signature is usually in them ("Trent Torrence ...
    t@purtera-it.com"). Marked as a fallback so a consumer can tell a resolved
    sender from a scraped one.

    Mutates in place. A file with no delivering message is left alone: an
    unattributed document must not end up looking attributed.
    """
    # authored_at FIRST, and only then the thread date. Both name the same
    # instant, but `email_thread.date` is the raw RFC 2822 header --
    # "Wed, 12 Aug 2026 18:00:51 +0000" -- while the delivered stamp it is
    # compared against is ISO 8601. Slicing the RFC form to 16 characters yields
    # "Wed, 12 Aug 2026", which can never equal "2026-08-12T18:00", so this join
    # matched nothing and every file silently took the signature-scraping
    # fallback below. The fallback reads the FIRST address in a forward, which
    # is the forwarder -- the opposite of what this function is for.
    emails: list[tuple[str, dict[str, Any]]] = []
    for doc in documents:
        thread = doc.get("email_thread") or {}
        when = doc.get("authored_at") or thread.get("date")
        if when and thread.get("sender"):
            emails.append((str(when)[:19], doc))

    for doc in documents:
        delivered = ((doc.get("lifecycle") or {}).get("delivered")) or []
        # Skip documents that speak for themselves. The test is whether this IS
        # a message -- a message carries its own sender, a file never does --
        # NOT whether it happens to carry a thread block. Those differ: the
        # Academy of Early Learning SOW picked up a stray email_thread from a
        # thread it was never delivered on, and that block carries
        # `sender: quinton.james@cdw.com` -- who did not deliver it. Skipping on
        # either the block or its inner sender left the file with no
        # delivered_by, so the originator rescue could not fire and it alone
        # stayed filed as our own material while its nine siblings were
        # readmitted. A mis-threaded file must still get its sender resolved.
        if not delivered or doc.get("sender_email"):
            continue
        first = next((d for d in delivered if isinstance(d, dict) and d.get("ts")), None)
        if first is None:
            continue
        stamp = str(first.get("ts") or "")[:19]

        match = next((d for ts, d in emails if ts and stamp and ts[:16] == stamp[:16]), None)
        if match is not None:
            thread = match.get("email_thread") or {}
            # The originator beats the forwarder. HubSpot only knows the message
            # that carried the file into the deal; the person who actually sent
            # it is further up the quoted chain.
            origin = match.get("originated_by")
            doc["delivered_by"] = origin or thread.get("sender")
            if origin and origin != thread.get("sender"):
                doc["forwarded_by"] = thread.get("sender")
            # Deliberately NOT copying the delivering message's `direction`.
            # HubSpot's direction is about the message's relationship to the DEAL
            # record, not about who wrote it: the Marion County SOWs resolve to
            # t@purtera-it.com -- our own domain -- on a message HubSpot marked
            # INCOMING. Showing "They sent - t@purtera-it.com" would be a claim
            # this join cannot support. The person is a fact; the side is not.
            doc["delivered_by_source"] = "delivering message"
            continue

        found = _EMAIL_ADDR_RE.search(str(first.get("text") or ""))
        if found:
            doc["delivered_by"] = found.group(0)
            doc["delivered_by_source"] = "signature in the forwarded message"


def _thread_index(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The deal's conversations, newest activity first.

    A deal's email is not 33 files, it is 6 conversations. Without this a reader
    has to reconstruct that from filenames that carry only HubSpot ids.

    NAMING. The thread's name is the most frequent normalised subject, breaking
    ties toward the LONGEST -- on deal 010215 the same conversation appears as
    both "010215 time clock installs for marion county school district" and
    "time clock installs for marion county school district", and the one
    carrying the deal number is the more useful name.

    Every variant seen is kept in `subject_variants`. Subject drift is how two
    halves of one conversation end up as two threads, so it is reported rather
    than smoothed away -- see `looks_split`.
    """
    from collections import Counter, defaultdict

    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"subjects": Counter(), "docs": [], "senders": Counter(), "dates": [], "messages": []}
    )
    for doc in documents:
        block = doc.get("email_thread")
        if not isinstance(block, dict) or not block.get("thread_id"):
            continue
        g = groups[str(block["thread_id"])]
        if block.get("subject_norm"):
            g["subjects"][str(block["subject_norm"])] += 1
        g["docs"].append(doc.get("artifact_id"))
        if block.get("sender"):
            g["senders"][str(block["sender"])] += 1
        if block.get("date"):
            g["dates"].append(str(block["date"]))
        # Per message: who sent it and what it carried. "This thread discussed
        # the SOWs" and "THIS message is where the SOWs came from" are different
        # facts, and only the second tells you whose documents they are.
        g["messages"].append({
            "artifact_id": doc.get("artifact_id"),
            "sender": doc.get("sender_email") or block.get("sender"),
            "originated_by": doc.get("originated_by"),
            "date": block.get("date"),
            "subject": block.get("subject"),
            "direction": doc.get("direction"),
            "attachment_count": len(doc.get("attachment_ids") or []),
            # The ids themselves, not just how many. A count is a number nobody
            # can check: on deal 010215 one message advertised 11 attachments
            # and every one of them resolved to a file that had never been
            # mirrored. With the ids present the reader can be taken to the
            # actual documents, and the ones that are missing can say so.
            "attachment_ids": [str(x) for x in (doc.get("attachment_ids") or []) if x],
        })

    out: list[dict[str, Any]] = []
    for thread_id, g in groups.items():
        subjects = g["subjects"]
        name = max(subjects.items(), key=lambda kv: (kv[1], len(kv[0])))[0] if subjects else ""
        dates = sorted(g["dates"])
        out.append(
            {
                "thread_id": thread_id,
                "name": name,
                "message_count": len(g["docs"]),
                "artifact_ids": g["docs"],
                "participants": [s for s, _ in g["senders"].most_common()],
                "first_message_at": dates[0] if dates else None,
                "last_message_at": dates[-1] if dates else None,
                "subject_variants": sorted(subjects),
                "messages": sorted(g["messages"], key=lambda m: str(m.get("date") or "")),
                "attachments_carried": sum(m["attachment_count"] for m in g["messages"]),
            }
        )

    # Subject drift splits one conversation in two: someone replies having
    # stripped or added a prefix, and the References chain does not bridge it.
    # Flag the pair rather than merging -- a prefix rule over-fires, and a wrong
    # merge is harder to notice than a reported suspicion.
    for a in out:
        a["looks_split_with"] = [
            b["thread_id"]
            for b in out
            if b is not a and a["name"] and b["name"] and (a["name"].endswith(b["name"]) or b["name"].endswith(a["name"]))
        ]

    out.sort(key=lambda t: (t["last_message_at"] or ""), reverse=True)
    return out



def _originating_sender(
    artifact_atoms: list[Any], artifact_id: str | None = None
) -> str | None:
    """The person a forwarded chain STARTED with, not whoever forwarded it last.

    An attachment that arrives inside a forward belongs to whoever actually sent
    it. On deal 010215 the ten Marion County SOWs reached HubSpot only when
    Trent forwarded the chain in -- HubSpot associates the files with that one
    message and knows nothing earlier. Attributing the documents to him says the
    customer's own SOWs came from us.

    The chain is in the body. The parser splits a forward into message blocks,
    oldest last, so the highest message_index is the original:

        msg1  Patrick Kelly <patrick@purtera-it.com>       1:12 PM
        msg2  Trent Torrence <t@purtera-it.com>            1:5x PM
        msg3  Quinton James <quinton.james@cdw.com>       10:20 AM
        msg4  Donnelly, Bernie <Bernie.Donnelly@sodexo.com> 8:3x AM   <- the customer

    ``artifact_id`` scopes the walk to THIS document's own refs. Dedup merges
    atoms across documents, and a winner keeps the losers' source_refs -- so a
    message in the same thread contributes its refs here. Two messages quote the
    same history, and ``message_index`` counts from the top of whichever message
    it came from, so indices from two chains are not comparable. Mixing them made
    the delivering email report "Trent Torrence <t@purtera-it.com>" at index 2,
    a string that appears nowhere in its own chain: it was the reply's index 2,
    read as if it were this forward's. The customer's own SOWs were then
    attributed to us, and the ten documents were filed unreadable.
    """
    best_index = -1
    best_sender = None
    want = str(artifact_id or "")
    for atom in artifact_atoms or []:
        for ref in getattr(atom, "source_refs", None) or []:
            if want and str(getattr(ref, "artifact_id", "") or "") != want:
                continue
            locator = getattr(ref, "locator", None)
            if not isinstance(locator, dict):
                continue
            sender = str(locator.get("sender") or "").strip()
            index = locator.get("message_index")
            if not sender or sender.lower() == "unknown" or not isinstance(index, int):
                continue
            if index > best_index:
                best_index, best_sender = index, sender
    return best_sender


def _document_thread(artifact_atoms: list[Any]) -> dict[str, Any] | None:
    """The thread block for a whole email document, lifted from its atoms.

    email_threading.py groups messages by RFC 5322 Message-ID / In-Reply-To /
    References, falling back to a normalised subject -- headers first, because
    they are facts and subjects drift. It writes the result onto every atom, and
    nowhere else, so nothing above atom level could see it.

    Document-level fields only: which conversation, where in it, and what it is
    called. The per-atom `gist` of the message being replied to stays on the
    atoms, where it belongs -- it is context for one utterance, not for a file.
    """
    for atom in artifact_atoms or []:
        block = None
        structured = getattr(atom, "structured", None)
        if isinstance(structured, dict):
            block = structured.get("email_thread")
        if block is None:
            value = getattr(atom, "value", None)
            if isinstance(value, dict):
                block = value.get("email_thread")
        if isinstance(block, dict) and block.get("thread_id"):
            return {
                "thread_id": block.get("thread_id"),
                "thread_index": block.get("thread_index"),
                "thread_size": block.get("thread_size"),
                "subject": block.get("subject"),
                "subject_norm": block.get("subject_norm"),
                "sender": block.get("sender"),
                "date": block.get("date"),
            }
    return None


def _load_manifest_provenance(project_dir: Path) -> dict[str, dict[str, Any]]:
    """Per-artifact authored time and direction, keyed by filename.

    Purpulse writes these onto each manifest artifact because only Purpulse knows
    them: the authored time lives in HubSpot metadata under a different key per
    source, and ``attachments.created_at`` is INGEST time -- on deal 010215 every
    email row was created 2026-08-17 for a deal opened 2026-08-12, so using it
    files the whole discovery phase under the wrong stage.
    """
    path = Path(project_dir) / PARSER_MANIFEST_SIDECAR
    if not path.is_file():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for art in manifest.get("artifacts") or []:
        if not isinstance(art, dict):
            continue
        name = str(art.get("filename") or "").strip()
        if not name:
            continue
        md = art.get("metadata") if isinstance(art.get("metadata"), dict) else {}
        out[name] = {
            "authored_at": art.get("authored_at"),
            "authored_at_precision": art.get("authored_at_precision"),
            "direction": art.get("direction"),
            "sender_domain": art.get("sender_domain"),
            # Which files a message carried. HubSpot records it per email, and
            # it is the difference between "this thread discussed the SOWs" and
            # "this message is where the SOWs came from".
            "attachment_ids": [str(x) for x in (md.get("attachmentIds") or []) if x],
            "sender_email": md.get("senderEmail"),
        }
    return out


def _load_manifest_run_cutoff(project_dir: Path) -> str | None:
    """The as-of the RUN was given, from ``context.as_of`` on the manifest sidecar.

    This is not ``deal_timeline.quote_asof``. That is a fact about the DEAL --
    when it first committed to an answer -- and is true no matter how the
    compile was invoked. This is a fact about THIS RUN: the operator asked for
    "data up to 20 minutes before Decision Pending", and the manifest was cut to
    match before parser-os ever saw it.

    Nothing recorded it, and the absence was not harmless. A cut run and a full
    run produced envelopes that were indistinguishable, so the UI could not say
    which set it was showing: on 010215 the page displayed 11 of 69 documents
    under a selector reading "All data - no cutoff", and the 18 documents the cut
    had excluded rendered as "Awaiting parse" -- as though the parser had failed
    on them, rather than as the deliberate answer to the question that was asked.

    A run that cannot say what it was asked cannot be audited.
    """
    path = Path(project_dir) / PARSER_MANIFEST_SIDECAR
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        ctx = manifest.get("context")
        if not isinstance(ctx, dict):
            return None
        value = ctx.get("as_of") or ctx.get("quote_asof")
        text = str(value).strip() if value is not None else ""
        return text or None
    except (json.JSONDecodeError, OSError, TypeError):
        return None


def _load_manifest_crm(project_dir: Path) -> dict[str, Any] | None:
    """Read ``context.crm`` from the parser manifest sidecar when present."""
    path = Path(project_dir) / PARSER_MANIFEST_SIDECAR
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        ctx = manifest.get("context")
        if not isinstance(ctx, dict):
            return None
        crm = ctx.get("crm")
        return dict(crm) if isinstance(crm, dict) else None
    except (json.JSONDecodeError, OSError, TypeError):
        return None


def write_orbitbrief_envelope(
    *,
    project_dir: Path,
    envelope: dict[str, Any],
    out_dir: Path | None = None,
) -> tuple[Path, Path, Path | None]:
    """Write the envelope JSON, markdown, and (if SowSmith is installed) the SOW.

    Returns ``(json_path, markdown_path, sow_path_or_None)``. Defaults
    to writing under ``<project_dir>/.orbitbrief/``. Pass ``out_dir``
    to override.

    The ``sow.md`` file is rendered by the standalone ``sowsmith``
    package (https://github.com/Purtera-IT/SowSmith) if it's
    installed. If SowSmith isn't on the path, ``sow_path`` is
    returned as ``None`` and only the envelope JSON + markdown are
    written. This keeps parser-os usable with or without the
    downstream SOW generator on the same machine.

    Install SowSmith to enable in-process SOW rendering::

        pip install -e path/to/SowSmith

    Or render after the fact::

        sowsmith render <project>/.orbitbrief/orbitbrief.input.json
    """
    out_dir = Path(out_dir) if out_dir is not None else (Path(project_dir) / ".orbitbrief")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / ENVELOPE_FILENAME
    md_path = out_dir / ENVELOPE_MARKDOWN_FILENAME
    json_path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(envelope_to_markdown(envelope), encoding="utf-8")

    sow_path: Path | None = None
    try:
        from sowsmith import build_sow_markdown  # type: ignore[import-not-found]
    except ImportError:
        build_sow_markdown = None  # type: ignore[assignment]
    if build_sow_markdown is not None:
        sow_path = out_dir / "sow.md"
        sow_path.write_text(build_sow_markdown(envelope), encoding="utf-8")
    return json_path, md_path, sow_path


def envelope_to_markdown(envelope: dict[str, Any]) -> str:
    """Render the envelope as a single LLM-ready markdown document.

    The output is a concatenation of every document's structured
    markdown, separated by clear horizontal rules and tagged with the
    artifact id so anchors stay unique even across documents.
    """
    lines: list[str] = []
    lines.append("---")
    lines.append(f"schema: {envelope.get('schema_version', ENVELOPE_SCHEMA_VERSION)}")
    lines.append(f"project_id: {envelope.get('project_id', '')}")
    lines.append(f"compile_id: {envelope.get('compile_id', '')}")
    lines.append(f"generated_at: {envelope.get('generated_at', '')}")
    lines.append("---")
    lines.append("")

    summary = envelope.get("summary") or {}
    if summary:
        lines.append("# OrbitBrief Project Envelope")
        lines.append("")
        lines.append(
            f"_{summary.get('artifact_count', 0)} artifacts • "
            f"{summary.get('page_count', 0)} pages • "
            f"{summary.get('atom_count', 0)} atoms • "
            f"{summary.get('packet_count', 0)} packets_"
        )
        lines.append("")
        by_type = summary.get("by_atom_type") or {}
        if by_type:
            lines.append("**Atoms by type**")
            for atom_type, count in sorted(by_type.items(), key=lambda kv: -kv[1]):
                lines.append(f"- {atom_type}: {count}")
            lines.append("")

    # OrbitBrief-Core cockpit surfaces — rendered as markdown sections
    # so an LLM consuming the markdown form sees the pre-aggregated
    # signals (not just raw atoms / packets).
    lines.extend(_render_cockpit_surfaces_md(envelope))

    for doc in envelope.get("documents", []) or []:
        artifact_id = doc.get("artifact_id", "")
        filename = doc.get("filename", "")
        artifact_type = doc.get("artifact_type", "")
        lines.append("---")
        lines.append("")
        lines.append(f'<!-- artifact id="{artifact_id}" type="{artifact_type}" -->')
        lines.append(f"## File: {filename}")
        lines.append("")
        structured = doc.get("structured") or {}
        schema = (structured.get("schema_version") if isinstance(structured, dict) else "") or ""
        if schema and schema != "orbitbrief.atom_projection.v1":
            # Every "real" structured doc — PDF, XLSX, DOCX, transcript,
            # email, quote — uses the unified renderer.
            lines.append(structured_doc_to_markdown(structured))
        else:
            lines.append(_render_generic_structured_md(structured))
        lines.append("")

    entities = envelope.get("entities") or []
    if entities:
        lines.append("---")
        lines.append("")
        lines.append("## Entities (cross-artifact)")
        lines.append("")
        lines.append("| Type | Canonical | Aliases | Artifacts | Atoms |")
        lines.append("|---|---|---|---|---|")
        for entity in entities:
            lines.append(
                "| {type} | {name} | {aliases} | {arts} | {atoms} |".format(
                    type=entity.get("entity_type", ""),
                    name=entity.get("canonical_name", ""),
                    aliases=", ".join((entity.get("aliases") or [])[:6]),
                    arts=len(entity.get("artifact_ids") or []),
                    atoms=len(entity.get("source_atom_ids") or []),
                )
            )
        lines.append("")

    edges = envelope.get("edges") or []
    if edges:
        lines.append("---")
        lines.append("")
        lines.append("## Cross-references and contradictions")
        lines.append("")
        lines.append("| Edge type | From | To | Cross-artifact | Reason |")
        lines.append("|---|---|---|---|---|")
        for edge in edges:
            lines.append(
                "| {type} | {fa} | {ta} | {ca} | {reason} |".format(
                    type=edge.get("edge_type", ""),
                    fa=edge.get("from_atom_id", ""),
                    ta=edge.get("to_atom_id", ""),
                    ca="yes" if edge.get("cross_artifact") else "no",
                    reason=(edge.get("reason") or "").replace("|", "\\|"),
                )
            )
        lines.append("")

    drawings = envelope.get("drawings") or {}
    artifacts = drawings.get("artifacts") or []
    if artifacts:
        lines.append("---")
        lines.append("")
        lines.append("## Drawings")
        lines.append("")
        idx = drawings.get("indexes") or {}
        det_counts = idx.get("detections_by_target_key") or {}
        if det_counts:
            lines.append("**Detection counts across all drawings**")
            for target_key, count in sorted(det_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"- {target_key}: {count}")
            lines.append("")
        warn_counts = idx.get("warnings_by_type") or {}
        if warn_counts:
            lines.append("**Warnings across all drawings**")
            for wt, count in sorted(warn_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"- {wt}: {count}")
            lines.append("")
        for art in artifacts:
            lines.append(f"### {art.get('filename') or art.get('artifact_id') or 'drawing'}")
            lines.append("")
            qc_ids = art.get("quantity_conflict_packet_ids") or []
            if qc_ids:
                lines.append(
                    f"_{len(qc_ids)} quantity_conflict packet(s): "
                    + ", ".join(qc_ids)
                    + "_"
                )
                lines.append("")
            for page in art.get("pages", []) or []:
                p = page.get("page")
                sn = page.get("sheet_number") or "?"
                lines.append(f"#### Page {p} — Sheet {sn}")
                meta = page.get("sheet_metadata") or {}
                if meta:
                    parts: list[str] = []
                    for k in ("sheet_title", "project_name", "scale", "issue_date", "revision"):
                        v = meta.get(k)
                        if v:
                            parts.append(f"{k}={v}")
                    if parts:
                        lines.append("- " + " • ".join(parts))
                target_counts = page.get("target_counts") or {}
                if target_counts:
                    lines.append("- Target counts: " + ", ".join(
                        f"{k}={v}" for k, v in sorted(target_counts.items())
                    ))
                rooms = page.get("rooms") or []
                if rooms:
                    lines.append(
                        "- Rooms: "
                        + ", ".join(
                            f"{r.get('label')}{(' ' + r['number']) if r.get('number') else ''}"
                            for r in rooms
                        )
                    )
                notes = page.get("keyed_notes") or []
                if notes:
                    lines.append(f"- Keyed notes: {len(notes)}")
                schedules = page.get("schedule_rows") or []
                if schedules:
                    lines.append(f"- Schedule rows: {len(schedules)}")
                warnings = page.get("warnings") or []
                if warnings:
                    types = sorted({w.get("warning_type") for w in warnings if w.get("warning_type")})
                    lines.append("- Warnings: " + ", ".join(types))
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ────────────────────────── internals ────────────────────────────────────


def _render_cockpit_surfaces_md(envelope: dict[str, Any]) -> list[str]:
    """Render the OrbitBrief-Core cockpit fields as markdown sections.

    Surfaces rendered:
      * project_vitals          — header score with component breakdown
      * pm_dashboard            — blockers / contradictions / open Qs
      * sow_readiness_scorecard — dimension table + grade
      * srl_missing_checklist   — coverage + missing field list
      * scope_truth             — canonical scope table + contested
      * change_order_timeline   — chronological change order audit
      * site_readiness          — per-site rollup table
      * stakeholder_load        — workload + bottleneck matrix

    Missing surfaces are silently skipped — older compile results
    without OrbitBrief-Core fields still render their atoms section.
    """
    lines: list[str] = []

    vitals = envelope.get("project_vitals") or {}
    if vitals:
        lines.append("---")
        lines.append("")
        lines.append("## Project Vitals")
        lines.append("")
        lines.append(
            f"**Score:** {vitals.get('score_100', '—')} / 100  ·  "
            f"**Band:** `{vitals.get('band', '—')}`  ·  "
            f"**Top drivers:** {', '.join(vitals.get('top_drivers') or []) or '—'}  ·  "
            f"**Top detractors:** {', '.join(vitals.get('top_detractors') or []) or '—'}"
        )
        lines.append("")
        components = vitals.get("components") or []
        if components:
            lines.append("| Component | Weight | Score | Contribution |")
            lines.append("|---|---|---|---|")
            for c in components:
                lines.append(
                    f"| {c.get('name', '—')} | {c.get('weight', 0):.2f} | "
                    f"{c.get('raw_score', 0):.2f} | {c.get('contribution', 0):.3f} |"
                )
            lines.append("")

    dash = envelope.get("pm_dashboard") or {}
    if dash:
        lines.append("---")
        lines.append("")
        lines.append("## PM Dashboard")
        lines.append("")
        bl = dash.get("blockers") or []
        if bl:
            lines.append(f"**Blockers ({len(bl)})**")
            for b in bl[:15]:
                lines.append(f"- [{b.get('kind', '—')}] {(b.get('summary', '') or '')[:200]}")
            lines.append("")
        cd = dash.get("cross_doc_contradictions") or []
        if cd:
            lines.append(f"**Cross-doc contradictions ({len(cd)})**")
            for c in cd[:10]:
                lines.append(f"- {(c.get('reason') or '')[:200]}")
            lines.append("")
        co = dash.get("change_orders") or []
        if co:
            lines.append(f"**Change orders ({len(co)})**")
            for c in co[:10]:
                delta = c.get("change_delta")
                delta_str = f" ({delta.get('from')}→{delta.get('to')}, Δ{delta.get('delta'):+d})" if delta else ""
                lines.append(f"- {(c.get('text') or '')[:200]}{delta_str}")
            lines.append("")
        oq = dash.get("open_questions") or []
        if oq:
            lines.append(f"**Open questions ({len(oq)})**")
            for q in oq[:10]:
                lines.append(f"- {(q.get('text') or '')[:200]}")
            lines.append("")
        sla = dash.get("sla_summary") or []
        if sla:
            lines.append(f"**SLA targets ({len(sla)})**")
            for s in sla[:10]:
                targets = s.get("sla") or {}
                target_str = ", ".join(f"{k}={v}" for k, v in targets.items())
                lines.append(f"- {target_str}  ({(s.get('text') or '')[:80]})")
            lines.append("")
        money = dash.get("money_summary") or {}
        if money.get("total"):
            lines.append(f"**Commercial total:** ${money['total']:,.2f} ({len(money.get('atoms', []))} atoms)")
            lines.append("")

    sc = envelope.get("sow_readiness_scorecard") or {}
    if sc:
        lines.append("---")
        lines.append("")
        lines.append("## SOW Readiness Scorecard")
        lines.append("")
        lines.append(
            f"**Overall:** {sc.get('readiness_score', 0):.2f} / 1.00  ·  "
            f"**Grade:** `{sc.get('grade', '—')}`"
        )
        lines.append("")
        lines.append("| Dimension | Score |")
        lines.append("|---|---|")
        for dim, d in (sc.get("dimensions") or {}).items():
            lines.append(f"| {dim} | {d.get('score', 0):.2f} |")
        lines.append("")

    ck = envelope.get("srl_missing_checklist") or {}
    if ck:
        lines.append("---")
        lines.append("")
        lines.append("## SRL Coverage")
        lines.append("")
        lines.append(
            f"**Coverage:** {ck.get('present_count', 0)} / {ck.get('field_count', 0)} fields "
            f"({(ck.get('coverage', 0) or 0) * 100:.0f}%)"
        )
        lines.append("")
        by_cat = ck.get("by_category") or {}
        if by_cat:
            lines.append("| Category | Present / Total | Coverage |")
            lines.append("|---|---|---|")
            for cat, stats in sorted(by_cat.items()):
                lines.append(
                    f"| {cat} | {stats.get('present', 0)} / {stats.get('total', 0)} | "
                    f"{(stats.get('coverage', 0) or 0) * 100:.0f}% |"
                )
            lines.append("")
        missing = ck.get("missing") or []
        if missing:
            lines.append(f"**Missing fields ({len(missing)})**")
            for m in missing:
                lines.append(f"- `{m.get('field_id')}` — {m.get('label')}")
            lines.append("")

    st = envelope.get("scope_truth") or {}
    if st.get("devices"):
        lines.append("---")
        lines.append("")
        lines.append("## Scope Truth")
        lines.append("")
        lines.append(
            f"**{st.get('device_count', 0)}** devices across **{st.get('site_count', 0)}** sites  ·  "
            f"**{st.get('contested_count', 0)}** contested"
        )
        lines.append("")
        lines.append("| Device | Site | Quantity | Governing | Status |")
        lines.append("|---|---|---|---|---|")
        for d in st["devices"]:
            status = "⚠ contested" if d.get("is_contested") else "✓"
            lines.append(
                f"| {d.get('device', '—')} | {d.get('site', '—')} | "
                f"**{d.get('canonical_quantity', '—')}** | "
                f"`{d.get('governing_authority', '—')}` | {status} |"
            )
        lines.append("")

    ct = envelope.get("change_order_timeline") or {}
    if ct.get("entries"):
        lines.append("---")
        lines.append("")
        lines.append(f"## Change Order Timeline ({ct.get('entry_count', 0)} entries)")
        lines.append("")
        lines.append("| Kind | Delta | Approved | Text |")
        lines.append("|---|---|---|---|")
        for e in ct["entries"][:20]:
            delta = e.get("change_delta") or {}
            if delta:
                delta_str = f"{delta.get('from')}→{delta.get('to')} ({delta.get('delta'):+d})"
            else:
                delta_str = "—"
            approval = "✓" if e.get("approval_signal") else "—"
            text = (e.get("text") or "").replace("|", "\\|").replace("\n", " ")[:160]
            lines.append(f"| {e.get('kind', '—')} | {delta_str} | {approval} | {text} |")
        lines.append("")

    sr = envelope.get("site_readiness") or {}
    if sr.get("sites"):
        lines.append("---")
        lines.append("")
        lines.append(
            f"## Site Readiness ({sr.get('site_count', 0)} sites, avg {sr.get('avg_readiness', 0):.2f})"
        )
        lines.append("")
        lines.append("| Site | Readiness | Devices | Stakeholders | Constraints | Contradictions |")
        lines.append("|---|---|---|---|---|---|")
        for s in sr["sites"]:
            lines.append(
                f"| `{s.get('site', '—')}` | {s.get('readiness', 0):.2f} | "
                f"{s.get('device_count', 0)} | {s.get('stakeholder_count', 0)} | "
                f"{s.get('constraint_count', 0)} | {s.get('contradiction_count', 0)} |"
            )
        lines.append("")

    sl = envelope.get("stakeholder_load") or {}
    if sl.get("stakeholders"):
        lines.append("---")
        lines.append("")
        lines.append(f"## Stakeholder Load ({sl.get('stakeholder_count', 0)} stakeholders)")
        lines.append("")
        if sl.get("bottlenecks"):
            lines.append(f"⚠ **Bottlenecks:** {', '.join(sl['bottlenecks'])}")
            lines.append("")
        lines.append("| Stakeholder | Risks | Critical | High | Actions | Severity Load |")
        lines.append("|---|---|---|---|---|---|")
        for s in sl["stakeholders"]:
            lines.append(
                f"| {s.get('slug', '—')} | {s.get('risk_count', 0)} | "
                f"{s.get('critical_risk_count', 0)} | {s.get('high_risk_count', 0)} | "
                f"{s.get('action_item_count', 0)} | {s.get('risk_severity_load', 0)} |"
            )
        lines.append("")

    return lines


def _resolve_artifact_path(project_dir: Path, filename: str) -> Path:
    """Resolve a stored artifact file inside the project dir.

    Manifest filenames are project-relative (forward-slash normalized).
    """
    rel = filename.replace("\\", "/")
    return (project_dir / rel).resolve()


def _structured_projection_for(
    *,
    artifact_path: Path,
    artifact_type: ArtifactType,
    artifact_atoms: list[EvidenceAtom],
    filename: str,
) -> dict[str, Any]:
    """Build the per-artifact ``structured`` payload for the envelope.

    Every Parser OS parser that opts in writes a structured doc to
    ``<stem>.derived/structured.json`` — load it directly so the
    envelope's markdown projection has the same fidelity for PDFs,
    XLSX/CSV workbooks, DOCX documents, email threads, transcripts,
    and vendor quotes.  When a parser hasn't produced one yet (legacy
    or unsupported artifacts), synthesize a flat atom projection so
    the artifact still shows up in the envelope.
    """
    derived = artifact_path.with_name(f"{artifact_path.stem}{DERIVED_DIR_SUFFIX}") / STRUCTURED_FILENAME
    if derived.is_file():
        try:
            return json.loads(derived.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return _project_atoms_to_structured(
        atoms=artifact_atoms,
        filename=filename,
        artifact_type=artifact_type,
    )


def _project_atoms_to_structured(
    *,
    atoms: list[EvidenceAtom],
    filename: str,
    artifact_type: ArtifactType,
) -> dict[str, Any]:
    """Synthesize a structured doc from a flat list of atoms.

    Group atoms by their ``SourceRef.locator['section_path']`` (or
    ``location`` / ``sheet`` / ``speaker`` for legacy locators) so the
    markdown projection still has section structure.
    """
    sections_by_path: dict[tuple[str, ...], dict[str, Any]] = {}
    for atom in atoms:
        path = _atom_section_path(atom)
        key = tuple(path)
        section = sections_by_path.get(key)
        if section is None:
            section = {
                "id": f"sec_{abs(hash(key)) % (10**12):012d}",
                "level": max(len(path), 1),
                "heading": path[-1] if path else "",
                "blocks": [],
                "subsections": [],
            }
            sections_by_path[key] = section
        section["blocks"].append(
            {
                "id": atom.id.replace("atm_", "blk_"),
                "kind": _atom_to_block_kind(atom),
                "text": atom.raw_text,
            }
        )

    sections = [sections_by_path[k] for k in sections_by_path]
    return {
        "schema_version": "orbitbrief.atom_projection.v1",
        "source": {"filename": filename, "artifact_type": artifact_type.value},
        "document": {"title": filename, "metadata": []},
        "pages": [
            {
                "page": 0,
                "title": filename,
                "metadata": [],
                "outline": [
                    {
                        "level": s["level"],
                        "heading": s["heading"],
                        "block_count": len(s["blocks"]),
                    }
                    for s in sections
                ],
                "sections": sections,
            }
        ],
    }


def _atom_section_path(atom: EvidenceAtom) -> list[str]:
    if atom.source_refs:
        ref: SourceRef = atom.source_refs[0]
        locator = ref.locator or {}
        section_path = locator.get("section_path")
        if isinstance(section_path, list) and section_path:
            return [str(x) for x in section_path]
        # Fall back to whatever locator field gives us section-ish context.
        fallback_keys = ("section", "sheet", "speaker", "channel", "location")
        for key in fallback_keys:
            value = locator.get(key)
            if value:
                return [str(value)]
    # Email Include/Exclude bullets carry polarity in value when locator
    # predates section_path stamping — keep envelope grouping stable.
    # Prefer lead_in + header so connective tissue survives compact projection.
    val = atom.value or {}
    list_section = str(val.get("list_section") or "").strip().lower()
    if list_section in {
        "include",
        "exclude",
        "equipment",
        "action_items",
        "key_decisions",
        "executive_summary",
        "open_questions",
        "decisions",
        "next_steps",
        "attendees",
        "participants",
        "agenda",
        "discussion",
        "notes",
        "follow_ups",
    }:
        path: list[str] = []
        lead = val.get("lead_in") or val.get("intro")
        if isinstance(lead, list):
            for item in lead:
                s = str(item or "").strip().rstrip(":")
                if s and s not in path:
                    path.append(s)
        elif isinstance(lead, str) and lead.strip():
            path.append(lead.strip().rstrip(":"))
        if list_section == "equipment":
            header = val.get("section_header") or "Equipment list"
        elif list_section in {"include", "exclude"}:
            header = val.get("section_header") or (
                "Include" if list_section == "include" else "Exclude"
            )
        else:
            header = val.get("section_header") or list_section.replace("_", " ").title()
        header_s = str(header).strip()
        if header_s and header_s not in path:
            path.append(header_s)
        return path
    return []


def _atom_to_block_kind(atom: EvidenceAtom) -> str:
    if atom.source_refs:
        locator_kind = atom.source_refs[0].locator.get("block_kind")
        if isinstance(locator_kind, str) and locator_kind:
            return locator_kind
    return "text"


def _build_drawings_section(
    *,
    atoms: list[EvidenceAtom],
    packets: list[Any],
    edges: list[Any],
    atoms_by_artifact: dict[str, list[EvidenceAtom]],
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the schematic ``drawings`` envelope section.

    Groups every schematic_* atom by (artifact, page), surfaces the
    parsed legend, the per-page target counts, the resolved schedule
    rows, the keyed notes, the rooms, the warnings, and any
    quantity_conflict packets that came out of schematic atoms.

    Empty by design when the project has no schematic atoms — the
    envelope's caller drops the section in that case so non-schematic
    projects produce byte-identical output.
    """
    schematic_atom_types = {
        "schematic_sheet_metadata",
        "schematic_legend",
        "schematic_room",
        "schematic_keyed_note",
        "schematic_note_callout",
        "schematic_schedule_row",
        "schematic_detection_target_set",
        "schematic_symbol_detection",
        "schematic_warning",
    }
    schematic_atoms = [a for a in atoms if a.atom_type.value in schematic_atom_types]
    if not schematic_atoms:
        return {"artifacts": [], "indexes": {}}

    artifact_filenames = {d["artifact_id"]: d.get("filename") for d in documents}

    by_art: dict[str, list[EvidenceAtom]] = defaultdict(list)
    for a in schematic_atoms:
        by_art[a.artifact_id].append(a)

    artifacts_out: list[dict[str, Any]] = []
    drawings_by_sheet: dict[str, list[str]] = defaultdict(list)
    detections_by_target: dict[str, int] = defaultdict(int)
    warnings_by_type: dict[str, int] = defaultdict(int)

    for artifact_id in sorted(by_art):
        art_atoms = by_art[artifact_id]
        per_page: dict[int, dict[str, Any]] = defaultdict(
            lambda: {
                "sheet_number": None,
                "sheet_metadata": None,
                "legend_id": None,
                "target_counts": defaultdict(int),
                "warnings": [],
                "rooms": [],
                "keyed_notes": [],
                "schedule_rows": [],
                "atom_ids": [],
            }
        )
        legends_out: list[dict[str, Any]] = []
        for atom in art_atoms:
            value = atom.value if isinstance(atom.value, dict) else {}
            page = value.get("page")
            if isinstance(page, int):
                per_page[page]["atom_ids"].append(atom.id)
            atom_kind = atom.atom_type.value
            if atom_kind == "schematic_sheet_metadata":
                if isinstance(page, int):
                    per_page[page]["sheet_metadata"] = {
                        k: v for k, v in value.items() if k != "page"
                    }
                    per_page[page]["sheet_number"] = value.get("sheet_number")
                    sn = value.get("sheet_number")
                    if isinstance(sn, str) and sn:
                        drawings_by_sheet[sn].append(atom.id)
            elif atom_kind == "schematic_legend":
                legends_out.append(
                    {
                        "legend_id": value.get("legend_id"),
                        "page": value.get("page"),
                        "sheet_number": value.get("sheet_number"),
                        "scope": value.get("scope"),
                        "entry_count": value.get("entry_count"),
                    }
                )
            elif atom_kind == "schematic_detection_target_set":
                if isinstance(page, int):
                    per_page[page]["legend_id"] = value.get("legend_id")
            elif atom_kind == "schematic_symbol_detection":
                tk = value.get("target_key")
                if isinstance(page, int) and isinstance(tk, str):
                    per_page[page]["target_counts"][tk] += 1
                    detections_by_target[tk] += 1
            elif atom_kind == "schematic_warning":
                wt = value.get("warning_type")
                if isinstance(page, int):
                    per_page[page]["warnings"].append(
                        {
                            "warning_type": wt,
                            "detail": value.get("detail"),
                            "target_key": value.get("target_key"),
                        }
                    )
                if isinstance(wt, str):
                    warnings_by_type[wt] += 1
            elif atom_kind == "schematic_room":
                if isinstance(page, int):
                    per_page[page]["rooms"].append(
                        {
                            "room_id": value.get("room_id"),
                            "label": value.get("label"),
                            "number": value.get("number"),
                        }
                    )
            elif atom_kind == "schematic_keyed_note":
                if isinstance(page, int):
                    per_page[page]["keyed_notes"].append(
                        {
                            "number": value.get("number"),
                            "text": value.get("text"),
                            "callout_count": value.get("callout_count", 0),
                        }
                    )
            elif atom_kind == "schematic_schedule_row":
                if isinstance(page, int):
                    per_page[page]["schedule_rows"].append(
                        {
                            "row_id": value.get("row_id"),
                            "schedule_kind": value.get("schedule_kind"),
                            "tag": value.get("tag"),
                            "fields": value.get("fields", {}),
                        }
                    )

        # Schematic quantity conflicts on this artifact.
        artifact_packet_ids: list[str] = []
        artifact_atom_ids = {a.id for a in art_atoms}
        for p in packets:
            if p.family.value != "quantity_conflict":
                continue
            packet_atom_ids = set(
                (p.contradicting_atom_ids or []) + (p.governing_atom_ids or [])
            )
            if packet_atom_ids & artifact_atom_ids:
                artifact_packet_ids.append(p.id)

        # Stabilize per_page payloads (dict -> dict).
        pages_out = []
        for page_index in sorted(per_page):
            entry = per_page[page_index]
            pages_out.append(
                {
                    "page": page_index,
                    "sheet_number": entry["sheet_number"],
                    "legend_id": entry["legend_id"],
                    "sheet_metadata": entry["sheet_metadata"],
                    "target_counts": dict(sorted(entry["target_counts"].items())),
                    "warnings": entry["warnings"],
                    "rooms": entry["rooms"],
                    "keyed_notes": entry["keyed_notes"],
                    "schedule_rows": entry["schedule_rows"],
                    "atom_ids": sorted(entry["atom_ids"]),
                }
            )

        artifacts_out.append(
            {
                "artifact_id": artifact_id,
                "filename": artifact_filenames.get(artifact_id),
                "pages": pages_out,
                "legends": sorted(legends_out, key=lambda l: (l.get("page") or 0, l.get("legend_id") or "")),
                "quantity_conflict_packet_ids": sorted(artifact_packet_ids),
            }
        )

    return {
        "artifacts": artifacts_out,
        "indexes": {
            "drawings_by_sheet_number": {
                k: sorted(v) for k, v in sorted(drawings_by_sheet.items())
            },
            "detections_by_target_key": dict(sorted(detections_by_target.items())),
            "warnings_by_type": dict(sorted(warnings_by_type.items())),
        },
    }


def _build_summary(
    *,
    atoms: list[EvidenceAtom],
    packets: list[EvidencePacket],
    documents: list[dict[str, Any]],
    entities: list[EntityRecord] | None = None,
    edges: list[EvidenceEdge] | None = None,
) -> dict[str, Any]:
    entities = entities or []
    edges = edges or []
    by_atom_type: Counter[str] = Counter(a.atom_type.value for a in atoms)
    by_authority: Counter[str] = Counter(a.authority_class.value for a in atoms)
    by_artifact_type: Counter[str] = Counter(d.get("artifact_type", "") for d in documents)
    by_edge_type: Counter[str] = Counter(e.edge_type.value for e in edges)
    by_entity_type: Counter[str] = Counter(e.entity_type for e in entities)
    cross_artifact_edges = sum(1 for e in edges if e.metadata.get("cross_artifact"))
    page_count = 0
    for doc in documents:
        structured = doc.get("structured") or {}
        if isinstance(structured, dict):
            page_count += len(structured.get("pages") or [])
    # A6 graceful degradation: roll up per-file parse_outcome into a
    # summary counter + an explicit degraded-files list. PM_HANDOFF
    # uses this to render a "Files requiring manual review" callout.
    parse_outcomes_counter: Counter[str] = Counter()
    degraded_files: list[dict[str, str]] = []
    for doc in documents:
        outcome = doc.get("parse_outcome") or {}
        status = outcome.get("status") or "unknown"
        parse_outcomes_counter[status] += 1
        if status in {"failed_parse", "skipped_no_parser", "ok_empty"}:
            degraded_files.append({
                "filename": str(doc.get("filename", "")),
                "status": status,
                "reason": str(outcome.get("reason", ""))[:300],
            })
    return {
        "artifact_count": len(documents),
        "page_count": page_count,
        "atom_count": len(atoms),
        "packet_count": len(packets),
        "entity_count": len(entities),
        "edge_count": len(edges),
        "cross_artifact_edge_count": cross_artifact_edges,
        "by_artifact_type": dict(by_artifact_type),
        "by_atom_type": dict(by_atom_type),
        "by_authority_class": dict(by_authority),
        "by_edge_type": dict(by_edge_type),
        "by_entity_type": dict(by_entity_type),
        "parse_outcomes": dict(parse_outcomes_counter),
        "degraded_files": degraded_files,
    }


def _build_indexes(
    *,
    atoms: list[EvidenceAtom],
    entities: list[EntityRecord] | None = None,
    edges: list[EvidenceEdge] | None = None,
) -> dict[str, dict[str, list[str]]]:
    entities = entities or []
    edges = edges or []
    by_section: dict[str, list[str]] = defaultdict(list)
    by_type: dict[str, list[str]] = defaultdict(list)
    by_authority: dict[str, list[str]] = defaultdict(list)
    by_artifact: dict[str, list[str]] = defaultdict(list)
    by_entity_key: dict[str, list[str]] = defaultdict(list)
    by_stakeholder: dict[str, list[str]] = defaultdict(list)
    by_device: dict[str, list[str]] = defaultdict(list)
    by_site: dict[str, list[str]] = defaultdict(list)
    tasks_by_site: dict[str, list[str]] = defaultdict(list)
    physical_site_slugs: set[str] = set()
    for atom in atoms:
        section_key = " > ".join(_atom_section_path(atom)) or "(root)"
        by_section[section_key].append(atom.id)
        by_type[atom.atom_type.value].append(atom.id)
        by_authority[atom.authority_class.value].append(atom.id)
        by_artifact[atom.artifact_id].append(atom.id)
        if atom.atom_type.value == "physical_site":
            for key in atom.entity_keys:
                if key.startswith("site:"):
                    physical_site_slugs.add(key[len("site:"):])
        site_slugs_for_atom = [
            key[len("site:"):]
            for key in atom.entity_keys
            if isinstance(key, str) and key.startswith("site:")
        ]
        if atom.atom_type.value == "task":
            # The tier gate decides ONLY whether the task lands in
            # tasks_by_site. It must not decide whether the atom is indexed
            # at all -- a `continue` here silently dropped every
            # non-quote-line task from atoms_by_entity_key and the
            # per-prefix indexes, and did so only when the classifier
            # import succeeded.
            in_task_index = True
            try:
                from app.core.task_tier_classifier import is_quote_line_task_atom

                in_task_index = is_quote_line_task_atom(atom)
            except Exception:
                pass
            if in_task_index:
                for slug in site_slugs_for_atom:
                    tasks_by_site[slug].append(atom.id)
        for key in atom.entity_keys:
            by_entity_key[key].append(atom.id)
            # Per-entity-prefix specialized indexes: O(1) lookup of
            # "every fact about this stakeholder / device / site"
            # without re-scanning atoms_by_entity_key. Downstream
            # consumers (SOWSmith.scope_clause, PM cockpit) hit these
            # constantly.
            if key.startswith("stakeholder:"):
                by_stakeholder[key[len("stakeholder:"):]].append(atom.id)
            elif key.startswith("device:"):
                by_device[key[len("device:"):]].append(atom.id)
            elif key.startswith("site:"):
                by_site[key[len("site:"):]].append(atom.id)
    edges_by_atom: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        edges_by_atom[edge.from_atom_id].append(edge.id)
        edges_by_atom[edge.to_atom_id].append(edge.id)
    entities_by_key: dict[str, str] = {entity.canonical_key: entity.id for entity in entities}
    return {
        "atoms_by_section_path": {k: sorted(v) for k, v in sorted(by_section.items())},
        "atoms_by_atom_type": {k: sorted(v) for k, v in sorted(by_type.items())},
        "atoms_by_authority": {k: sorted(v) for k, v in sorted(by_authority.items())},
        "atoms_by_artifact": {k: sorted(v) for k, v in sorted(by_artifact.items())},
        "atoms_by_entity_key": {k: sorted(v) for k, v in sorted(by_entity_key.items())},
        "atoms_by_stakeholder_slug": {k: sorted(v) for k, v in sorted(by_stakeholder.items())},
        "atoms_by_device_slug": {k: sorted(v) for k, v in sorted(by_device.items())},
        "atoms_by_site_slug": {k: sorted(v) for k, v in sorted(by_site.items())},
        "tasks_by_site_slug": {k: sorted(v) for k, v in sorted(tasks_by_site.items())},
        "physical_site_slugs": sorted(physical_site_slugs),
        "edges_by_atom": {k: sorted(v) for k, v in sorted(edges_by_atom.items())},
        "entity_id_by_canonical_key": dict(sorted(entities_by_key.items())),
    }


def _compact_structured(value: Any) -> dict[str, Any]:
    """The atom's structured value as it ships, with the source row bounded.

    A tabular atom carries the row it was minted from under ``raw_cells`` so
    a provenance claim about it — "does this display name appear in its own
    source?" — is decidable from the envelope alone rather than only by
    re-reading the artifact. That row is capped at the minter, but a value
    can also arrive here after ``semantic_dedup`` merged two atoms, which
    concatenates list fields; this re-applies the SAME cap at serialization
    so no merge path can grow the envelope without bound. Everything else on
    the value is passed through untouched.
    """
    if not value:
        return {}
    out = dict(value)
    raw_cells = out.get("raw_cells")
    if isinstance(raw_cells, (list, tuple)) and raw_cells:
        out["raw_cells"] = capped_source_row(raw_cells)
    return out


def _compact_atom(atom: EvidenceAtom) -> dict[str, Any]:
    primary_ref = atom.source_refs[0] if atom.source_refs else None
    projected: dict[str, Any] = {
        "id": atom.id,
        "artifact_id": atom.artifact_id,
        "atom_type": atom.atom_type.value,
        "authority_class": atom.authority_class.value,
        "confidence": atom.confidence,
        "text": atom.raw_text,
        "section_path": _atom_section_path(atom),
        "locator": dict(primary_ref.locator) if primary_ref is not None else {},
        "verified": _atom_verification_state(atom),
        # A5 cross-doc reconciliation needs entity_keys + structured
        # values on every atom so consumers can group atoms touching
        # the same logical entity (e.g. total_contract_value) and
        # flag value contradictions across documents. Previously the
        # compact projection dropped both, forcing PM_HANDOFF to
        # regex over raw_text. Same data unlocks B2 (risk register),
        # B6 (per-site pricing rollup), etc.
        "entity_keys": list(atom.entity_keys),
        "structured": _compact_structured(atom.value),
        # Per-atom trust signal: the calibrated probability + the accept/
        # needs_review verdict. Previously dropped on projection, so every
        # consumer (PM-chip "unsure" gate, truth_gate, auto-accept) read null
        # and fell back to the raw heuristic. confidence_raw lets consumers tell
        # the calibrated value apart from the pre-calibration heuristic.
        "calibrated_confidence": atom.calibrated_confidence,
        "review_status": atom.review_status.value if hasattr(atom.review_status, "value") else atom.review_status,
        # Why an atom sits in the queue must be auditable from the envelope
        # alone; the flags were never serialised (live 010300: 117 queued
        # atoms, "flags []", cause invisible).
        "review_flags": list(getattr(atom, "review_flags", None) or []),
        "confidence_raw": getattr(atom, "confidence_raw", None),
    }
    # HOW a decision was made, when something recorded it. Emitted only when
    # present, so envelopes for atoms nothing stamped are byte-identical.
    #
    # This carries real weight for site links: after site_provenance_join, an
    # atom joined to a school because it lived in that school's SOW looks
    # exactly like one that named the school in its own text. Without this
    # field an auditor cannot tell a derived link from an asserted one, which
    # is the same silent-equivalence problem as a zero that might be an
    # absence.
    prov = getattr(atom, "decision_provenance", None)
    if prov:
        projected["decision_provenance"] = dict(prov)
    return projected


def _atom_verification_state(atom: EvidenceAtom) -> str:
    if not atom.receipts:
        return "unverified"
    statuses = {r.replay_status for r in atom.receipts}
    if "failed" in statuses:
        return "failed"
    if statuses == {"verified"}:
        return "verified"
    if "verified" in statuses:
        return "partial"
    return "unsupported"


def _filter_site_entities_against_physical_atoms(
    entities: list[EntityRecord],
    atoms: list[EvidenceAtom],
) -> list[EntityRecord]:
    """v57.3 — drop ``site:*`` entity records that don't trace to a
    physical_site atom; v57.3.1 — also REWRITE atom.entity_keys so
    atoms previously tagged with a ghost site key get reassigned to
    the canonical site they actually describe.

    Why: ``orbitbrief-core/world_model/site_reality/cluster.py`` walks
    every ``site:*`` entity in the envelope and builds one cluster per
    entity. The dossier renders one row per cluster. Without this
    filter, every LLM-extracted ghost (``atlanta_west_office``,
    ``optbot_atlanta_office``, ``atl_hq_2026``, ``site:site``,
    ``atlanta_headquarters_innovation_tower``, ...) becomes a dossier
    site even though the canonical roster only has the 5 ATL-XX-XX rows.

    The rule: a ``site:*`` entity is real iff its canonical_key,
    canonical_name, or any alias matches the slugified site_id, name,
    or facility_name of a physical_site atom. Everything else is a
    ghost LLM cluster.

    v57.3.1 follow-up: when we DROP a ghost site:* entity, atoms that
    previously had ``entity_keys=[..., site:ghost_name]`` get orphaned
    (their cluster disappears) — that's why the OPTBOT dossier showed
    ATL-WEST-02 missing. Fix: for each dropped ghost, find the best
    canonical match by token overlap against the physical_site facility
    names, then walk all atoms and rewrite their entity_keys so
    ``site:atlanta_west_office`` becomes ``site:atl_west_02``. The
    canonical cluster then absorbs those atoms and passes
    orbitbrief-core's >2-evidence gate.

    Non-``site:*`` entities (vendor, device, money, etc.) pass through
    untouched.
    """
    import re as _re

    def _slug(s: str) -> str:
        return _re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")

    def _tokens(slug: str) -> set[str]:
        return {t for t in slug.split("_") if t and len(t) >= 2}

    # ── 1. Build canonical site catalog from physical_site atoms.
    # ``canonical_site_id_slugs`` = the *strict* truth (only the
    # ``slug(site_id)`` of each canonical roster atom). An entity is
    # accepted as canonical ONLY when its key matches one of these
    # exactly. Everything else — even legitimate aliases like
    # ``site:atl_047`` (truncation of ``atl_047_04``) — is treated as
    # ghost and routed through token-overlap redirect so atoms collapse
    # into the canonical key. This guarantees the dossier sees one
    # cluster per real roster row.
    canonical_site_id_slugs: set[str] = set()
    canonical_to_token_sets: dict[str, list[set[str]]] = {}
    # site_id slug → primary facility_name for entity injection.
    primary_to_facility_name: dict[str, str] = {}

    for a in atoms:
        atype = getattr(a, "atom_type", None)
        atype_s = atype.value if hasattr(atype, "value") else str(atype or "")
        if atype_s != "physical_site":
            continue
        val = getattr(a, "value", None) or {}
        if not isinstance(val, dict):
            continue
        sid = val.get("site_id") or val.get("id") or ""
        sid_slug = _slug(str(sid))
        if not sid_slug:
            continue
        primary = sid_slug
        canonical_site_id_slugs.add(primary)
        canonical_to_token_sets.setdefault(primary, [])
        facility = val.get("facility_name") or val.get("name") or sid
        if isinstance(facility, str) and facility.strip():
            primary_to_facility_name.setdefault(primary, facility.strip())
        identity_strings: list[str] = []
        for field in ("site_id", "id", "name", "facility_name", "address", "street_address"):
            v = val.get(field)
            if isinstance(v, str) and v.strip():
                identity_strings.append(v)
        names_field = val.get("names") or val.get("aliases") or ()
        if isinstance(names_field, (list, tuple)):
            for n in names_field:
                if isinstance(n, str) and n.strip():
                    identity_strings.append(n)
        for s in identity_strings:
            slug = _slug(s)
            if slug:
                canonical_to_token_sets[primary].append(_tokens(slug))

    if not canonical_site_id_slugs:
        return entities

    # ── 2. Pass 1 — classify each entity as canonical-or-ghost. For
    # ghosts, also pick the best canonical to redirect them to.
    kept: list[EntityRecord] = []
    ghost_to_canonical: dict[str, str] = {}  # ghost site_key → canonical site_key

    def _best_canonical_for(candidate_slugs: set[str]) -> str | None:
        """Return the primary canonical slug whose token sets best
        overlap with any candidate slug, or None if no confident match.

        Scoring is two-tier: ``(total_overlap, site_id_overlap)``.
        Total overlap counts any identity-string token shared; site_id
        overlap counts only tokens shared with the canonical's site_id
        slug itself (the distinguishing piece).

        Acceptance rule: REQUIRE at least one shared site_id token. This
        catches ``atlanta_west_office`` -> ``atl_west_02`` (shares ``west``
        with the site_id ``atl_west_02``) but rejects ``headquarters`` /
        ``site:site`` (no site_id contains ``headquarters`` or ``site``).
        Pure facility-name matches without a site_id token are too risky
        — ``office`` alone is in every canonical and would over-collapse.
        """
        if not candidate_slugs:
            return None
        cand_tokens: set[str] = set()
        for cs in candidate_slugs:
            cand_tokens |= _tokens(cs)
        if not cand_tokens:
            return None
        best_primary: str | None = None
        best_score: tuple[int, int] = (0, 0)
        for primary, token_sets in canonical_to_token_sets.items():
            site_id_tokens = _tokens(primary)
            # Exclude tokens that are pure digits or 1-char — they're
            # too generic to be discriminative (``01``, ``02``, ``a``).
            site_id_tokens = {t for t in site_id_tokens if len(t) >= 2 and not t.isdigit()}
            site_id_overlap = len(cand_tokens & site_id_tokens)
            if site_id_overlap < 1:
                continue
            for ts in token_sets:
                total_overlap = len(cand_tokens & ts)
                score = (total_overlap, site_id_overlap)
                if score > best_score:
                    best_score = score
                    best_primary = primary
        return best_primary

    canonical_seen_in_entities: set[str] = set()
    for ent in entities:
        ck = getattr(ent, "canonical_key", "") or ""
        if not ck.startswith("site:"):
            kept.append(ent)
            continue
        ck_slug = ck[len("site:"):]
        # STRICT canonical: entity is real iff its slug == one of the
        # canonical site_id slugs exactly. Everything else (truncated
        # aliases, LLM names, year-suffix hallucinations) is a ghost and
        # gets routed through the token-overlap redirect below.
        if ck_slug in canonical_site_id_slugs:
            kept.append(ent)
            canonical_seen_in_entities.add(ck_slug)
            continue
        # Ghost — try to redirect to a canonical via token overlap.
        candidate_slugs: set[str] = {ck_slug}
        cname = getattr(ent, "canonical_name", "") or ""
        if cname:
            candidate_slugs.add(_slug(cname))
        for alias in (getattr(ent, "aliases", None) or ()):
            if isinstance(alias, str) and alias:
                candidate_slugs.add(_slug(alias))
        candidate_slugs.discard("")
        best_primary = _best_canonical_for(candidate_slugs)
        if best_primary:
            ghost_to_canonical[ck] = f"site:{best_primary}"

    # ── 3. Pass 2 — rewrite atom.entity_keys: replace each ghost site
    # key with its canonical-mapped site key. This redirects orphaned
    # atoms into the canonical cluster so the >2-evidence gate in
    # orbitbrief-core promotes them.
    if ghost_to_canonical:
        for atom in atoms:
            keys = getattr(atom, "entity_keys", None)
            if not keys:
                continue
            try:
                new_keys: list[str] = []
                changed = False
                seen: set[str] = set()
                for k in keys:
                    if isinstance(k, str) and k.startswith("site:") and k in ghost_to_canonical:
                        canon = ghost_to_canonical[k]
                        changed = True
                        if canon not in seen:
                            new_keys.append(canon)
                            seen.add(canon)
                    else:
                        if isinstance(k, str) and k not in seen:
                            new_keys.append(k)
                            seen.add(k)
                if changed:
                    try:
                        atom.entity_keys = new_keys
                    except (AttributeError, TypeError):
                        pass
            except TypeError:
                continue

    # ── 4. Inject canonical site entities for any physical_site atom
    # whose canonical key isn't already represented. Without this, the
    # ``site:atl_047_04`` key has atoms tagged to it (from the
    # physical_site atom + redirected ghosts) but no entity record →
    # orbitbrief-core's cluster builder never seeds a cluster for it
    # and the dossier shows the alias name instead of the facility name.
    # We inject a minimal EntityRecord so the cluster gets built with
    # the canonical key + facility-name display.
    missing = canonical_site_id_slugs - canonical_seen_in_entities
    if missing:
        try:
            from app.core.schemas import EntityRecord, ReviewStatus  # local import to avoid cycles
            import uuid as _uuid
            for slug in sorted(missing):
                # Skip if we couldn't extract a facility name (shouldn't
                # happen for real physical_site atoms but defensive).
                facility = primary_to_facility_name.get(slug, "")
                if not facility:
                    continue
                # Find one source atom_id to anchor provenance.
                anchor_atom_id = ""
                for a in atoms:
                    atype = getattr(a, "atom_type", None)
                    atype_s = atype.value if hasattr(atype, "value") else str(atype or "")
                    if atype_s != "physical_site":
                        continue
                    val = getattr(a, "value", None) or {}
                    if not isinstance(val, dict):
                        continue
                    sid = val.get("site_id") or val.get("id") or ""
                    if _slug(str(sid)) == slug:
                        anchor_atom_id = getattr(a, "id", "") or ""
                        break
                # Find the original project_id from any atom (all share one).
                proj_id = ""
                for a in atoms:
                    pid = getattr(a, "project_id", None)
                    if pid:
                        proj_id = str(pid)
                        break
                injected = EntityRecord(
                    id=f"ent_canon_{slug}_{_uuid.uuid4().hex[:8]}",
                    project_id=proj_id,
                    entity_type="site",
                    canonical_key=f"site:{slug}",
                    canonical_name=facility,
                    aliases=[],
                    source_atom_ids=[anchor_atom_id] if anchor_atom_id else [],
                    confidence=0.99,
                    review_status=ReviewStatus.auto_accepted,
                )
                kept.append(injected)
        except Exception:
            # If injection fails for any reason, fall through silently
            # — the dossier might miss a cluster but at least won't crash.
            pass

    return kept


def _compact_entity(
    entity: EntityRecord,
    atoms_by_artifact: dict[str, list[EvidenceAtom]],
    all_atoms: list[EvidenceAtom],
) -> dict[str, Any]:
    """Add ``artifact_ids`` provenance so consumers can see which files
    mention this entity."""
    artifact_ids: set[str] = set()
    atom_ids = set(entity.source_atom_ids)
    for atom in all_atoms:
        if atom.id in atom_ids:
            artifact_ids.add(atom.artifact_id)
    return {
        "id": entity.id,
        "entity_type": entity.entity_type,
        "canonical_key": entity.canonical_key,
        "canonical_name": entity.canonical_name,
        "aliases": list(entity.aliases),
        "artifact_ids": sorted(artifact_ids),
        "source_atom_ids": list(entity.source_atom_ids),
        "review_status": entity.review_status.value,
        "confidence": entity.confidence,
    }


def _compact_edge(edge: EvidenceEdge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "edge_type": edge.edge_type.value,
        "from_atom_id": edge.from_atom_id,
        "to_atom_id": edge.to_atom_id,
        "reason": edge.reason,
        "confidence": edge.confidence,
        "cross_artifact": bool(edge.metadata.get("cross_artifact")),
        "metadata": dict(edge.metadata or {}),
    }


def _compact_packet(packet: EvidencePacket) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": packet.id,
        "family": packet.family.value,
        "anchor_type": packet.anchor_type,
        "anchor_key": packet.anchor_key,
        "status": packet.status.value,
        "confidence": packet.confidence,
        "governing_atom_ids": list(packet.governing_atom_ids),
        "supporting_atom_ids": list(packet.supporting_atom_ids),
        "contradicting_atom_ids": list(packet.contradicting_atom_ids),
        "reason": packet.reason,
    }
    # Preserve the PacketCertificate so downstream consumers
    # (SOWSmith.scope_clause, OrbitBrief.scope_truth, RunbookGen.site_steps,
    # AtlasDispatch.site_readiness, VisionQC.photo_requirements) see the
    # cert's blast_radius declaration through the envelope.
    cert = getattr(packet, "certificate", None)
    if cert is not None:
        try:
            out["certificate"] = cert.model_dump()
        except Exception:  # pragma: no cover
            try:
                out["certificate"] = dict(cert)
            except Exception:
                out["certificate"] = None
        if out.get("certificate") and isinstance(out["certificate"], dict):
            br = out["certificate"].get("blast_radius") or []
            if br:
                out["blast_radius"] = list(br)
    return out


def _render_generic_structured_md(structured: dict[str, Any]) -> str:
    """Render a non-PDF projection (the lighter ``atom_projection.v1``)."""
    lines: list[str] = []
    document = structured.get("document") or {}
    title = document.get("title")
    if title:
        lines.append(f"### {title}")
        lines.append("")
    for page in structured.get("pages", []) or []:
        for section in page.get("sections", []) or []:
            heading = section.get("heading") or "(uncategorized)"
            sec_id = section.get("id")
            anchor = f'  <a id="{sec_id}"></a>' if sec_id else ""
            lines.append(f"#### {heading}{anchor}")
            lines.append("")
            for block in section.get("blocks", []) or []:
                block_id = block.get("id")
                if block_id:
                    lines.append(f'<a id="{block_id}"></a>')
                text = (block.get("text") or "").strip()
                if text:
                    lines.append(f"- {text}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "ENVELOPE_FILENAME",
    "ENVELOPE_MARKDOWN_FILENAME",
    "build_orbitbrief_envelope",
    "write_orbitbrief_envelope",
    "envelope_to_markdown",
]
