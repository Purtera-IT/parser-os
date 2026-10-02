from __future__ import annotations

import functools
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from app.core.cache import (
    build_cached_artifact_result,
    compute_artifact_sha256,
    load_cached_artifact_result,
    save_cached_artifact_result,
)
from app.core.candidate_adjudicator import adjudicate_candidates
from app.core.candidates import summarize_candidate_outcomes
from app.core.entity_extraction import enrich_atoms as enrich_entity_keys
from app.core.entity_resolution import (
    collect_site_alias_groups,
    collect_stakeholder_alias_groups,
    extract_entity_records,
    fuse_alias_groups,
    resolve_aliases,
)
from app.core.quality_metrics import compute_quality
from app.core.graph_builder import build_edges
from app.core.ids import stable_id
from app.core.manifest import (
    build_artifact_fingerprint,
    compute_output_signature,
    create_manifest,
    finalize_manifest,
)
from app.core.packet_certificates import build_packet_certificate
from app.core.packetizer import build_packets
from app.core.risk import packet_pm_sort_key, score_packet_risk
from app.core.schemas import (
    COMPILER_VERSION,
    SCHEMA_VERSION,
    CandidateAtom,
    CompileResult,
    EvidenceAtom,
    ParserDerivedFile,
    ParserOutput,
)
from app.core.source_replay import attach_receipts_to_atoms, replay_atom_receipts, summarize_receipts
from app.core.suppression_ledger import (
    SUPPRESSION_FLAG_PREFIX,
    capture_suppressed,
    merge_suppressed,
)
from app.core.telemetry import CompileTelemetry
from app.core.validators import validate_compile_result
from app.domain import load_domain_pack, set_active_domain_pack
from app.domain.pack_router import auto_route_pack
from app.domain.schemas import DomainPack
from app.learning.calibration import apply_calibration
from app.parsers.parser_router import choose_parser

# Atoms below this confidence are forced to needs_review with a stable flag.
# Anything below the floor is too uncertain to govern a packet without a human
# look — keep them in the result for transparency, but never let them ride into
# active packets unchallenged.
LOW_CONFIDENCE_FLOOR = 0.50


def _a_shade_less_than(parent: Any, field: str, *, drop: float = 0.05,
                       floor: float = 0.5, absent: float = 0.8) -> float:
    """A split child is a shade less certain than the paragraph it came from.

    ``getattr(parent, field, 0.8)`` looks like it handles a missing value and
    does not: the attribute EXISTS on every atom and is routinely ``None``
    before calibration runs. ``None - 0.05`` raised, the whole prose_list_split
    stage was caught by its own `except`, and the compile carried on with one
    warning and no splitting at all -- so a paragraph holding six stakeholders
    stayed one atom. Live 010180 (compile cmp_135c10268e527d45):

        WARNING: prose_list_split failed: TypeError: unsupported operand
                 type(s) for -: 'NoneType' and 'float'
    """
    value = getattr(parent, field, None)
    if value is None:
        value = absent
    return max(floor, float(value) - drop)


def _dropped_atom_notes(stage: str, dropped: list[Any], *, cap: int = 60) -> list[str]:
    """One trace line per atom a stage removed, so a live compile explains
    what it threw away (live 010300: two scanned clauses vanished with no
    stage owning the loss; the suppression ledger is never persisted)."""
    notes: list[str] = []
    for a in list(dropped or [])[:cap]:
        t = getattr(a, "atom_type", None)
        t = str(getattr(t, "value", t) or "")
        text = " ".join(str(getattr(a, "raw_text", None) or getattr(a, "normalized_text", None) or "").split())
        notes.append(f"INFO: dropped[{stage}] {t}: {text[:140]}")
    if len(dropped or []) > cap:
        notes.append(f"INFO: dropped[{stage}] ... {len(dropped) - cap} more")
    return notes


def _is_provenance_record(atom: Any) -> bool:
    """A record the parser wrote about the document's own plumbing — a quoted
    message header ("From: X | Sent: Y"), an image / attachment / binary-region
    marker. It carries no claim for a PM to doubt, so confidence gates skip it."""
    value = getattr(atom, "value", None)
    kind = str((value or {}).get("kind") or "") if isinstance(value, dict) else ""
    return kind.endswith(("_header", "_marker"))

_DERIVED_DIR_SUFFIXES = (".derived",)

# Directory names that should never be walked for input artifacts.
# These are project metadata / outputs from previous compiles, not
# scope content.  See PRODUCTION_GAPS.md P1.6.
_NON_ARTIFACT_DIRS = frozenset(
    {
        "labels",          # gold standards / human-curated review labels
        ".orbitbrief",     # OrbitBrief envelope outputs from prior compiles
        ".cache",          # generic cache dir
        ".git",            # vcs metadata
        ".github",
        ".vscode",
        ".idea",
        "node_modules",
        "__pycache__",
    }
)

# File names (case-insensitive) that should never be parsed as artifacts.
# These are project metadata or known output sentinels.
_NON_ARTIFACT_FILES = frozenset(
    {
        "source_notes.md",       # case-level provenance notes
        "readme.md",             # project README
        "license",
        "license.md",
        "license.txt",
        ".gitignore",
        ".gitattributes",
        "thumbs.db",
        ".ds_store",
        "project.yaml",          # parser-os project config (read separately)
        "project.yml",
        ".parserignore",         # ignore-pattern list
        # Purpulse's own manifest, which the envelope builder reads separately
        # (orbitbrief_envelope._load_manifest_crm / _load_manifest_provenance).
        # Left in the scan it was parsed as a DOCUMENT: on deal 010215 it
        # produced 789 of the envelope's 1,933 atoms -- 40% -- consisting of
        # mime types, ingest timestamps, attachment UUIDs and metadata email
        # addresses. It also leaked another deal's filename into this deal's
        # evidence. It is bookkeeping about the artifacts, never an artifact.
        ".parser_manifest.json",
    }
)

# File-name patterns (case-insensitive substring) that mark gold/review
# files which must never be parsed as scope content.
_NON_ARTIFACT_PATTERNS = (
    "gold_standard",
    ".gold.",
    "_gold.",
    "_review.",
    ".review.",
)


#: Spreadsheets are what the table rollup exists for (a 9 MB rate card is
#: tens of thousands of rows); everything else -- a PDF or a Word table --
#: is a document a person reads line by line.
_SPREADSHEET_TYPES = frozenset({"xlsx", "xls", "xlsm", "csv", "tsv"})


def _is_spreadsheet_atom(atom: Any) -> bool:
    for ref in list(getattr(atom, "source_refs", None) or [])[:1]:
        at = getattr(ref, "artifact_type", None)
        at = str(getattr(at, "value", at) or "").lower()
        fn = str(getattr(ref, "filename", "") or "").lower()
        if at in _SPREADSHEET_TYPES or fn.rsplit(".", 1)[-1] in _SPREADSHEET_TYPES:
            return True
    return False


def _keep_unsurvived_document_rows(before: list[Any], after: list[Any], warnings: list[str]) -> list[Any]:
    """table_rollup must not hide a document's lines behind a count.

    A rolled-up summary reads "N table rows (rolled up)"; no kept atom carries
    the rows' words. On a PDF/Word document every such row goes back, and a
    summary none of whose rows stay folded is removed (nothing was folded).
    Spreadsheets keep the rollup: that is what it is for.
    """
    from app.core.suppression_ledger import keep_unsurvived_lines

    after, restored = keep_unsurvived_lines(
        before, after, stage="table_rollup", eligible=lambda a: not _is_spreadsheet_atom(a),
    )
    if not restored:
        return after
    after_ids = {id(a) for a in after}
    still_folded = {
        str(getattr(a, "artifact_id", "") or "")
        for a in before
        if id(a) not in after_ids
    }
    before_ids = {id(a) for a in before}
    out = []
    for a in after:
        v = getattr(a, "value", None)
        if (
            id(a) not in before_ids
            and isinstance(v, dict)
            and v.get("_source") == "table_rollup_backstop"
            and str(getattr(a, "artifact_id", "") or "") not in still_folded
        ):
            continue
        out.append(a)
    warnings.append(f"INFO: table_rollup kept {len(restored)} document rows no survivor contains")
    return out


def _deal_state_atom(template: Any, line: Any) -> Any:
    """One line of where-the-deal-stands, as an atom.

    Copied off a real atom so it inherits the project and an artifact and
    stays traceable. It carries `why` and `assumption` for the same reason
    every derived claim does: nobody wrote it, so it has to be arguable.
    """
    import copy as _copy

    from app.core.schemas import AtomType, ReviewStatus

    atom = _copy.deepcopy(template)
    text = f"{line.key.replace('_', ' ')}: {line.value} — {line.why}"
    atom.raw_text = text
    if hasattr(atom, "normalized_text"):
        atom.normalized_text = text.lower()
    atom.atom_type = AtomType.deal_state
    atom.review_status = ReviewStatus.needs_review
    atom.review_flags = ["derived_deal_state"]
    atom.entity_keys = []
    atom.value = {
        "kind": "deal_state",
        "key": line.key,
        "state": line.value,
        "why": line.why,
        "assumption": line.assumption,
        "evidence": list(line.evidence or []),
    }
    atom.confidence = 0.8
    if hasattr(atom, "id"):
        atom.id = stable_id("atm", str(getattr(atom, "project_id", "")),
                            "deal_state", line.key)
    # Nobody wrote this line, so it has no page, row or cell: the copied
    # locator pointed a reader at a cell that says something else (010003:
    # a Deal Kit SELL RATES row). Keep the artifact; name the evidence.
    _ev_ids = [str(getattr(a, "id", "")) for a in (getattr(line, "evidence_atoms", None) or []) if getattr(a, "id", None)]
    atom.value["evidence_atom_ids"] = _ev_ids
    for ref in getattr(atom, "source_refs", None) or []:
        try:
            ref.locator = {"derived": True, "derived_from": _ev_ids}
        except Exception:
            pass
    return atom


def _ask_the_pm_atom(template: Any, text: str) -> Any:
    """One open question in place of an export's withheld OCR rows.

    Built by copying an atom that WAS there, so it inherits the artifact and
    source ref and stays traceable to the file it stands for. Without it a
    sheet whose rows were all withheld would simply be absent, and absent is
    indistinguishable from never-looked-at.
    """
    import copy as _copy

    from app.core.schemas import AtomType, ReviewStatus

    atom = _copy.deepcopy(template)
    atom.raw_text = text
    if hasattr(atom, "normalized_text"):
        atom.normalized_text = text.lower()
    atom.atom_type = AtomType.open_question
    atom.review_status = ReviewStatus.needs_review
    atom.review_flags = ["export_ocr_withheld", "needs_pm_description"]
    if isinstance(getattr(atom, "value", None), dict):
        atom.value = {"kind": "ask_the_pm", "reason": "export_ocr_withheld"}
    atom.confidence = 0.5
    if hasattr(atom, "id"):
        atom.id = stable_id("atm", str(getattr(atom, "artifact_id", "")), "ask_the_pm",
                            text[:80])
    return atom


#: How many artifacts to parse at once.
#:
#: Measured on 24 live artifacts, pinned to 4 CPUs to match the worker
#: container. With OCR stubbed at zero the widths are indistinguishable --
#: 0.98x to 1.00x, because the reading itself is zip inflate, XML and regex,
#: and the GIL means more threads cannot make that faster. The entire benefit
#: is overlapping the Document Intelligence round trip, so with OCR stubbed at
#: a realistic 1.5s:
#:
#:     workers   median      speedup
#:           1   171.2s        1.00x
#:           2    87.0s        1.97x
#:           4    47.0s        3.64x
#:           8    45.8s        3.74x
#:
#: Four takes 97% of the available speedup. The fourth thread is worth 1.4x;
#: the next four are worth 1.03x between them, because past that the GIL-bound
#: reading is the floor. Eight costs double the concurrent Document
#: Intelligence calls -- and 429s there are what latch `_llm_unreachable` and
#: degrade the rest of a compile -- and double the peak memory, with every
#: worker holding a decompressed workbook.
#:
#: Raising this is safe for CORRECTNESS but not free: a parser that keeps
#: per-document state on `self` is shared between these threads, because the
#: registry hands out one instance. See `PerThreadState` in `parsers/base.py`.
#:
#: 0 or 1 disables the prefetch entirely and the compile behaves exactly as it
#: did before: the loop below computes each parse inline.
PARSE_WORKERS = int(os.environ.get("SOWSMITH_PARSE_WORKERS", "4"))


def _prefetch_parses(
    plan: "list[tuple[str, Any, Any]]",
    *,
    project_id: str,
    domain_pack: Any,
    workers: int,
) -> "dict[str, Any]":
    """Parse artifacts concurrently, keyed by artifact_id.

    The compile spent 881 of its 1500 seconds in `parse_artifacts` on a
    68-document deal -- 13 seconds apiece, almost none of it computing. The
    loop was strictly sequential, so 68 network round trips happened one after
    another and the deal died before the stages that use the atoms ever ran.

    This does NOT restructure that loop, deliberately. The loop keeps its
    order, its accumulators, its per-artifact error handling and its cache
    writes; it simply finds the expensive call already done. Anything this
    misses -- a parser that raised here, a prefetch that was skipped -- the
    loop computes inline exactly as before, so the worst case is the old
    behaviour rather than a gap.

    ORDER IS NOT AFFECTED, which is the point that matters most. Results are
    returned in a dict and consumed by the caller in the artifacts' own order,
    never in completion order. Atom ordering feeds `label_key`, and a
    completion-ordered parse would re-key every atom on every compile -- which
    would silently detach every label ever written.
    """
    out: dict[str, Any] = {}
    if workers <= 1 or len(plan) < 2:
        return out
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(artifact_id: str, parser: Any, path: Any):
        return parser.parse_artifact_full(
            project_id=project_id,
            artifact_id=artifact_id,
            path=path,
            domain_pack=domain_pack,
        )

    started = time.time()
    with ThreadPoolExecutor(max_workers=min(workers, len(plan)),
                            thread_name_prefix="parse") as pool:
        futures = {pool.submit(_one, aid, p, path): aid for aid, p, path in plan}
        for fut in as_completed(futures):
            aid = futures[fut]
            try:
                out[aid] = fut.result()
            except Exception:  # noqa: BLE001 - the loop will retry inline and report
                logging.getLogger(__name__).warning(
                    "prefetch parse failed for %s; the compile will parse it inline",
                    aid, exc_info=True,
                )
    logging.getLogger(__name__).info(
        "Prefetched %d/%d parse(s) on %d worker(s) in %.1fs",
        len(out), len(plan), workers, time.time() - started,
    )
    return out


def _materialize_derived_files(
    artifact: Path,
    derived_files: list[ParserDerivedFile],
) -> None:
    """Write parser-emitted derived files next to ``artifact``.

    ``relative_path`` is interpreted relative to the artifact's parent
    directory so a parser can write into ``<stem>.derived/...`` or any
    other sibling location.  Path traversal is rejected up-front to
    keep the cache safe (a malicious cache entry can't escape the
    project directory).
    """
    base = artifact.parent.resolve()
    for entry in derived_files:
        rel = (entry.relative_path or "").strip()
        if not rel:
            continue
        target = (base / rel).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            # Path tried to escape the artifact directory — skip.
            continue
        # long_write_text: Windows >260-char paths (deep _rerun/<uuid>/.../<sha>/
        # <long name>.derived/...) otherwise throw WinError 206 here AFTER the
        # parser did all the work, silently dropping the whole file's output.
        from app.core.longpath import long_write_text
        if entry.content_kind == "json":
            long_write_text(
                target,
                json.dumps(entry.content_json, indent=2, ensure_ascii=False),
            )
        elif entry.content_kind in {"markdown", "text"}:
            long_write_text(target, entry.content_text or "")


def _is_derived_path(path: Path, project_dir: Path) -> bool:
    """Return True when ``path`` is inside a parser-managed derived dir.

    Parsers (e.g. orbitbrief_pdf) materialize structured artifacts into
    sibling ``<stem>.derived/`` directories next to their source.  These
    are downstream consumer outputs (OrbitBrief input format), not
    inputs to compile — skip them so they don't get re-routed as
    unknown ``.json`` artifacts on the next compile pass.
    """
    try:
        rel = path.relative_to(project_dir)
    except ValueError:
        return False
    return any(
        part.endswith(_DERIVED_DIR_SUFFIXES) for part in rel.parts[:-1]
    )


def _is_excluded_artifact(path: Path, project_dir: Path) -> bool:
    """Return True when ``path`` should never be parsed as an artifact.

    Excludes project metadata (`labels/`, `SOURCE_NOTES.md`,
    `project.yaml`), VCS / IDE dirs, and gold-standard files that
    accompany the corpus but aren't scope content.  See PRODUCTION_GAPS
    P1.6.
    """
    try:
        rel = path.relative_to(project_dir)
    except ValueError:
        return False
    parts = rel.parts
    # Any ancestor directory in the no-walk list?
    for part in parts[:-1]:
        if part.lower() in _NON_ARTIFACT_DIRS:
            return True
    name_lower = path.name.lower()
    if name_lower in _NON_ARTIFACT_FILES:
        return True
    for pattern in _NON_ARTIFACT_PATTERNS:
        if pattern in name_lower:
            return True
    return False


def _read_parserignore(project_dir: Path) -> list[str]:
    """Read ``<project>/.parserignore`` glob patterns if present.

    Returns lowercased glob patterns; ``#`` comments and blank lines
    are skipped.  Honors ``project.yaml``'s ``parserignore_extra`` too
    so operators can keep ignore rules in one config file.
    """
    patterns: list[str] = []
    ignore_path = project_dir / ".parserignore"
    if ignore_path.is_file():
        try:
            for line in ignore_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                patterns.append(stripped.lower())
        except Exception:  # pragma: no cover — never fail compile on ignore read
            pass
    # project.yaml extras — silently merged so a missing /.parserignore
    # doesn't matter.
    try:
        from app.domain.project_config import load_project_config

        cfg = load_project_config(project_dir)
        if cfg is not None and cfg.parserignore_extra:
            patterns.extend(p.strip().lower() for p in cfg.parserignore_extra if p and p.strip())
    except Exception:  # pragma: no cover — config load errors shouldn't kill compile
        pass
    return patterns


def _matches_ignore_pattern(rel_path: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    from fnmatch import fnmatch

    rel_lower = rel_path.replace("\\", "/").lower()
    for pattern in patterns:
        if fnmatch(rel_lower, pattern) or fnmatch("/" + rel_lower, pattern):
            return True
    return False


def _iter_artifacts(project_dir: Path) -> list[Path]:
    ignore_patterns = _read_parserignore(project_dir)
    results: list[Path] = []
    # Prefer a dedicated `artifacts/` subdir if one exists — this is the
    # canonical "this is real scope content" convention used across the
    # STRESS_* corpus.  When present, walk only that subtree.
    artifacts_dir = project_dir / "artifacts"
    if artifacts_dir.is_dir():
        walk_root = artifacts_dir
    else:
        walk_root = project_dir
    for path in walk_root.rglob("*"):
        if not path.is_file():
            continue
        if _is_derived_path(path, project_dir):
            continue
        if _is_excluded_artifact(path, project_dir):
            continue
        try:
            rel = str(path.relative_to(project_dir))
        except ValueError:
            rel = path.name
        if _matches_ignore_pattern(rel, ignore_patterns):
            continue
        results.append(path)
    # A HubSpot note that only says "Note" and carried files is a pointer, not
    # a document: its author and date travel on the files it carried instead
    # (see app/core/note_attachments.py). Dropped here so the compile, the
    # census and the envelope all agree it is not an artifact.
    try:
        from app.core.note_attachments import note_attachment_links

        folded = {
            f.lstrip("/\\").replace("\\", "/") for f in note_attachment_links(project_dir).folded
        }
    except Exception:  # pragma: no cover - never fail discovery over links
        folded = set()
    if folded:
        def _rel(p: Path) -> str:
            try:
                return str(p.relative_to(project_dir)).replace("\\", "/")
            except ValueError:
                return p.name

        results = [p for p in results if _rel(p) not in folded]
    return sorted(results, key=lambda p: str(p).lower())


def project_census(project_dir: Path | str, atoms):
    """Build the deal-level content census and reconcile it against ``atoms``.

    The census is an inventory of every region of every modality in the deal's
    source files, read *independently* of the production parser. Reconciling it
    against the emitted atoms yields the never-detected loss set
    (``census.uncovered()``) — exactly the denominator
    :func:`app.core.complaint_router.route` needs for its NEEDS_EXTRACTOR
    bucket.

    It deliberately reuses :func:`_iter_artifacts`, so the census denominator is
    drawn from the *same* file set the compile consumed — no drift between what
    was compiled and what we check coverage against. Pure read; no LLM, no
    network; safe to call after any compile from ``(project_dir, result.atoms)``.
    Never raises — returns an empty census on any failure.
    """
    from app.parsers.census import reconciled_census

    try:
        root = Path(project_dir).resolve()
        return reconciled_census(
            _iter_artifacts(root), atoms, artifact=root.name
        )
    except Exception:  # pragma: no cover - census must never break a caller
        from app.core.content_census import ContentCensus

        return ContentCensus(artifact=str(project_dir))


# Opt-in feedback store. The store stays OFF by default (decide() is a
# transparent pass-through to the LLM, byte-identical to Phase 2) so the whole
# test suite is unaffected. Operations turn it on by pointing
# SOWSMITH_FEEDBACK_STORE_DB at a SQLite path; we wire it once per process,
# seed the built-in global corrections (PurTera self-address), and register it
# via decide.set_store. Any failure is swallowed — a store that won't load
# must never break a compile.
_FEEDBACK_STORE_WIRED = False


def _open_feedback_store(db_path: str):
    """Open + seed a FeedbackStore at ``db_path``. Raises on failure."""
    from app.core.feedback_store import FeedbackStore, seed_default_corrections

    store = FeedbackStore(db_path)
    seed_default_corrections(store)
    return store


def _maybe_wire_feedback_store() -> None:
    global _FEEDBACK_STORE_WIRED
    if _FEEDBACK_STORE_WIRED:
        return
    import os as _os
    import tempfile as _tempfile

    db_path = _os.environ.get("SOWSMITH_FEEDBACK_STORE_DB", "").strip()
    if not db_path:
        return
    _FEEDBACK_STORE_WIRED = True  # one attempt per process, success or not
    _log = logging.getLogger(__name__)
    try:
        from app.core.decide import get_store, set_store

        if get_store() is not None:  # already wired (e.g. by a test/host)
            return
    except Exception:
        _log.warning("feedback store: decide module unavailable", exc_info=True)
        return

    try:
        store = _open_feedback_store(db_path)
    except Exception:
        # A store that won't load must never break a compile -- but it must not
        # fail INVISIBLY either. With no store, decide() returns its fallback
        # without ever reaching resolve, so every taught correction behaves as
        # though it was never taught and the abstention log points inside
        # resolve never fire. That is the shape this hid: site fusion asking one
        # pair and getting "fallback:None" while the matching correction sat in
        # the store, with nothing anywhere saying the store had failed to load.
        #
        # The observed cause was SQLite on an Azure Files (SMB) mount: the file
        # predated the `candidates` column, so every open attempted the ALTER
        # TABLE migration -- an exclusive write -- and SMB would not grant the
        # lock. It failed on 100% of compiles for weeks.
        #
        # Blob is the durable master for corrections (feedback_blob.sync_into_store
        # runs right after this and rehydrates from it), so a PROCESS-LOCAL db is
        # a fully functional store, not a degraded one. Falling back to local disk
        # turns a total learning outage into a working compile.
        _log.warning(
            "feedback store NOT wired from %s — falling back to local disk",
            db_path, exc_info=True,
        )
        fallback = _os.path.join(_tempfile.gettempdir(), "_feedback.db")
        if _os.path.abspath(fallback) == _os.path.abspath(db_path):
            _log.error(
                "feedback store unavailable: %s is already local — every "
                "correction will be ignored for this compile", db_path,
            )
            return
        try:
            store = _open_feedback_store(fallback)
        except Exception:
            _log.error(
                "feedback store unavailable: local fallback %s failed too — "
                "every correction will be ignored for this compile",
                fallback, exc_info=True,
            )
            return
        db_path = fallback

    set_store(store)
    _log.info(
        "feedback store wired from %s (%d correction(s))",
        db_path, len(store.all_corrections(active_only=True)),
    )


#: Most atoms one artifact may contribute before it is treated as a data export
#: rather than deal evidence. 0 disables the cap.
#:
#: A real deal document lands in the hundreds; the largest legitimate scope
#: workbook measured across the corpus is far under this. 94,047 from a single
#: customer report is what this exists to catch.
from app.core.admission_chatter import ADMISSION_REGEX_FLAG as _ADMISSION_REJECT_FLAG  # noqa: E402
_MAX_ATOMS_PER_ARTIFACT = int(os.environ.get("SOWSMITH_MAX_ATOMS_PER_ARTIFACT", "12000"))


@functools.lru_cache(maxsize=1)
def _parser_code_fingerprint() -> str:
    """SHA of the parser package source, folded into the artifact cache key so
    that ANY parser code change auto-invalidates the cache.

    Root cause it fixes: ``parser_version`` is a hand-maintained constant (e.g.
    ``"docx_parser_v1"``) that nobody bumps when the parser is improved. Since
    the cache key includes only that string + the file sha256, a stale cached
    parse survives parser fixes forever — e.g. the prose-gate fix that keeps
    "Estimated quantity: 110 units" never reaches an already-parsed deal, on the
    worker too. Keying on the actual code (not a manual constant) makes every
    parser fix take effect on the next compile. Computed once per process."""
    import hashlib

    root = Path(__file__).resolve().parent.parent / "parsers"
    h = hashlib.sha256()
    try:
        for p in sorted(root.rglob("*.py")):
            try:
                h.update(p.name.encode("utf-8"))
                h.update(p.read_bytes())
            except Exception:
                continue
    except Exception:
        return "nofp"
    return h.hexdigest()[:12]



def _attach_rule_decisions(atoms: "list[Any]") -> int:
    """Put each atom's SemanticRule decisions into ``value["rule_decisions"]``.

    Matched on the exact text the rule was asked about. A rule judges a LINE,
    and an atom is usually that line, so an exact match is the honest join --
    a fuzzy one would attribute a decision to text it was never made about.
    Atoms with no matching decision simply get nothing.

    Never raises: a labelling aid must not be able to fail a compile.
    """
    try:
        from app.core.semantic_rules import decisions_enabled, decisions_for

        if not decisions_enabled():
            return 0
        attached = 0
        for atom in atoms:
            text = (getattr(atom, "raw_text", "") or getattr(atom, "text", "") or "").strip()
            if not text:
                continue
            found = decisions_for(text)
            if not found:
                continue
            value = getattr(atom, "value", None)
            if not isinstance(value, dict):
                continue
            value["rule_decisions"] = found
            attached += 1
        return attached
    except Exception:  # pragma: no cover - never break a compile over this
        return 0


def compile_project(
    project_dir: Path,
    project_id: str | None = None,
    allow_errors: bool = False,
    allow_unverified_receipts: bool = False,
    persistence_hook: Callable[[CompileResult], None] | None = None,
    domain_pack: DomainPack | str | Path | None = None,
    calibrator_path: Path | None = None,
    abstain_threshold: float = 0.70,
    use_cache: bool = True,
    stage_callback: Callable[..., None] | None = None,
    stage_start_callback: Callable[..., None] | None = None,
) -> CompileResult:
    project_dir = project_dir.resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"Project path does not exist: {project_dir}")

    # Decide ONCE whether the embedder serves this compile, before any parser
    # asks. Probed per rule evaluation it can answer yes for one document and
    # no for the next, so a deal reads differently depending on which files
    # happened to be in flight -- see `semantic_rules.semantic_backend_available`.
    try:
        from app.core.semantic_rules import reset_decisions, reset_semantic_backend

        reset_semantic_backend()
        # And forget the previous compile's rule decisions, so what the
        # labelling UI shows belongs to THIS run.
        reset_decisions()
    except Exception:  # pragma: no cover - never break a compile over this
        pass

    # Opt-in: activate the feedback store iff SOWSMITH_FEEDBACK_STORE_DB is set.
    # No-op otherwise, so default compiles (and the test suite) are unchanged.
    _maybe_wire_feedback_store()
    # Cross-container instant learning: pull PM corrections the SERVICE mirrored
    # to blob into this worker's live store so decide() honors them on THIS
    # compile. Gated + best-effort; no-op unless SOWSMITH_FEEDBACK_BLOB is on.
    # Restore what earlier containers already read. `fetch_ml.py` downloads
    # /tmp/ml at boot but nothing ever uploaded the OCR cache back, so every
    # cold start re-read every picture through a billed service -- and got a
    # fresh chance to read it differently, which is the determinism work
    # stopping at the container boundary. Gated + best-effort.
    try:
        from app.core import cache_blob as _cb

        _cb.sync_ocr_into_cache()
        _cb.sync_embed_into_cache()
    except Exception:  # pragma: no cover - a cache restore must never break a compile
        pass

    try:
        from app.core import feedback_blob as _fb
        from app.core.decide import get_store as _get_store

        _st = _get_store()
        if _st is not None:
            _fb.sync_into_store(_st)
    except Exception:  # pragma: no cover - sync must never break a compile
        pass

    resolved_project_id = project_id or project_dir.name

    # v44: expose project_dir name to per-pack domain extractors via env.
    # Used by app.core.exemplars.detect_domain_extras() to add domain-
    # specific exemplars (POS / ITAD / cabling / wireless / etc.).
    import os as _os
    _os.environ["SOWSMITH_PROJECT_DIR_NAME"] = project_dir.name
    if isinstance(domain_pack, DomainPack):
        # Pre-loaded pack from caller wins outright (e.g. tests)
        resolved_domain_pack = domain_pack
        pack_routing_decision = None
    else:
        # Pack auto-routing: explicit `--domain-pack` overrides; otherwise
        # we look at project.yaml → SOURCE_NOTES.md → content scoring.
        # See PRODUCTION_GAPS.md P0.1.
        resolved_domain_pack, pack_routing_decision = auto_route_pack(
            project_dir, explicit=domain_pack
        )
    set_active_domain_pack(resolved_domain_pack)
    telemetry = CompileTelemetry(
        project_id=resolved_project_id,
        on_stage_end=stage_callback,
        on_stage_start=stage_start_callback,
    )
    warnings: list[str] = []
    if pack_routing_decision is not None:
        warnings.append(
            f"INFO: domain pack '{resolved_domain_pack.pack_id}' selected via "
            f"{pack_routing_decision.source} ({pack_routing_decision.rationale})"
        )
    if resolved_domain_pack.reference_ontology_path:
        warnings.append(
            "WARNING: Domain pack uses reference-schema subset adapter (TODO: strict DomainPack mapper); "
            f"bundled ontology: {resolved_domain_pack.reference_ontology_path}"
        )
    atoms = []
    candidates: list[CandidateAtom] = []
    rejected_candidates: list[CandidateAtom] = []
    # Retained-suppression ledger: atoms removed by a drop-stage are captured
    # here (flagged + reason-stamped) instead of vanishing, so omission
    # complaints stay localizable and every drop stays auditable. Pure sidecar
    # — never re-enters the accepted atom set.
    suppressed_atoms: list[EvidenceAtom] = []
    try:
        from app.core.suppression_ledger import DROP_NOTES as _DROP_NOTES_RESET
        _DROP_NOTES_RESET.clear()
    except Exception:
        pass
    fingerprints = []
    parser_atom_counts: Counter[str] = Counter()
    parser_routing: list[dict] = []
    artifact_paths: dict[str, Path] = {}
    cache_hits = 0
    cache_misses = 0
    reused_artifact_ids: list[str] = []

    with telemetry.stage("discover_artifacts", input_count=1) as stage:
        artifacts = _iter_artifacts(project_dir)
        telemetry.end_stage(stage, output_count=len(artifacts))

    parse_warnings: list[str] = []
    parse_errors: list[str] = []
    # Warm the expensive per-artifact call concurrently. The loop below is
    # unchanged -- same order, same accumulators, same error handling -- it
    # just finds the parse already done. Anything missed here it computes
    # inline, so the worst case is the behaviour this replaced.
    _prefetched: dict[str, Any] = {}
    if PARSE_WORKERS > 1 and len(artifacts) > 1:
        _plan: list[tuple[str, Any, Any]] = []
        for _art in artifacts:
            try:
                _rel = str(_art.relative_to(project_dir)).replace("\\", "/")
                _aid = stable_id("art", resolved_project_id, _rel)
                _p, _m, _ = choose_parser(_art, domain_pack=resolved_domain_pack)
                if _p is None:
                    continue
                # Skip what the cache already holds. Without this a warm
                # re-compile would parse every artifact concurrently and then
                # throw all of it away, which is slower than the sequential
                # path it replaced -- the prefetch has to respect the same
                # cache the loop consults, or it is not an optimisation.
                if use_cache:
                    _pv = f"{_p.capability.parser_version}+code{_parser_code_fingerprint()}"
                    if load_cached_artifact_result(
                        artifact_id=_aid,
                        sha256=compute_artifact_sha256(_art),
                        parser_name=_m.parser_name,
                        parser_version=_pv,
                        domain_pack_id=resolved_domain_pack.pack_id,
                        domain_pack_version=resolved_domain_pack.version,
                    ) is not None:
                        continue
                _plan.append((_aid, _p, _art))
            except Exception:  # noqa: BLE001 - planning must not break the compile
                continue
        if _plan:
            try:
                _prefetched = _prefetch_parses(
                    _plan, project_id=resolved_project_id,
                    domain_pack=resolved_domain_pack, workers=PARSE_WORKERS,
                )
            except Exception:  # noqa: BLE001 - fall back to the sequential path
                logging.getLogger(__name__).warning(
                    "parse prefetch unavailable; parsing inline", exc_info=True)
                _prefetched = {}

    with telemetry.stage("parse_artifacts", input_count=len(artifacts)) as stage:
        # The longest stage in 23% of compiles. It already NAMES the file it is
        # on; counting them as well is what lets a reader turn "9 of 35, in 40
        # seconds" into a remaining time from this run's own pace.
        telemetry.set_stage_progress(0, len(artifacts))
        for _artifact_index, artifact in enumerate(artifacts):
            relative_name = str(artifact.relative_to(project_dir)).replace("\\", "/")
            # Name the file in the heartbeat. Without it a wedge on ONE artifact
            # is indistinguishable from a slow stage — 35 files, six hours, and
            # the log could not say which one had stopped.
            telemetry.set_stage_item(relative_name)
            # Files ALREADY finished, so the count never claims the one in hand
            # is done. A bar that counts the current item is a bar that reaches
            # the end before the work does.
            telemetry.set_stage_progress(_artifact_index, len(artifacts))
            artifact_id = stable_id("art", resolved_project_id, relative_name)
            artifact_paths[artifact_id] = artifact
            parsed_atoms = []
            parsed_candidates: list[CandidateAtom] = []
            per_artifact_warnings: list[str] = []
            parser_key = "none"
            parser_name = "none"
            parser_version = "unknown"
            cache_hit = False
            artifact_sha256 = compute_artifact_sha256(artifact)
            try:
                parser, match, all_matches = choose_parser(
                    path=artifact,
                    domain_pack=resolved_domain_pack,
                )
                parser_name = match.parser_name
                parser_version = parser.capability.parser_version if parser is not None else "unknown"
                # Fold the parser code fingerprint into the cache key so any
                # parser change busts the cache (a never-bumped manual
                # parser_version otherwise serves stale parses forever).
                if parser is not None:
                    parser_version = f"{parser_version}+code{_parser_code_fingerprint()}"
                parser_routing.append(
                    {
                        "artifact_id": artifact_id,
                        "filename": relative_name,
                        "chosen_parser": parser_name,
                        "parser_version": parser_version,
                        "confidence": match.confidence,
                        "reasons": match.reasons,
                        "cache_hit": False,
                        "matches": [row.model_dump(mode="json") for row in all_matches],
                        # A6 graceful degradation: per-file outcome
                        # status. Defaults to pending; overwritten below
                        # when the parse succeeds, is skipped, or fails.
                        # PM_HANDOFF readers (and the systems engineer
                        # diffing successful vs failed files) depend on
                        # this being present on every routing entry.
                        "outcome": {
                            "status": "pending",
                            "atom_count": 0,
                            "warning_count": 0,
                        },
                    }
                )
                parser_key = parser_name
                if parser is None:
                    warning = f"WARNING: No parser matched artifact {relative_name}; skipping file"
                    parse_warnings.append(warning)
                    parser_routing[-1]["outcome"] = {
                        "status": "skipped_no_parser",
                        "reason": (
                            f"no parser matched (best candidate: {parser_name} "
                            f"@ confidence={match.confidence:.2f})"
                        ),
                        "atom_count": 0,
                        "warning_count": 0,
                    }
                else:
                    cached = None
                    if use_cache:
                        cached = load_cached_artifact_result(
                            artifact_id=artifact_id,
                            sha256=artifact_sha256,
                            parser_name=parser_name,
                            parser_version=parser_version,
                            domain_pack_id=resolved_domain_pack.pack_id,
                            domain_pack_version=resolved_domain_pack.version,
                        )
                    parsed_derived_files: list[ParserDerivedFile] = []
                    if cached is not None:
                        parsed_atoms = list(cached.atoms)
                        parsed_candidates = list(cached.candidates)
                        per_artifact_warnings.extend(cached.warnings)
                        parsed_derived_files = list(cached.derived_files)
                        cache_hits += 1
                        reused_artifact_ids.append(artifact_id)
                        cache_hit = True
                    else:
                        # Warmed in parallel above when possible; identical
                        # object either way, and computed here if the prefetch
                        # skipped it or raised. See `_prefetch_parses`.
                        parser_result = _prefetched.pop(artifact_id, None)
                        if parser_result is None:
                            parser_result = parser.parse_artifact_full(
                                project_id=resolved_project_id,
                                artifact_id=artifact_id,
                                path=artifact,
                                domain_pack=resolved_domain_pack,
                            )
                        parsed_atoms = list(parser_result.atoms)
                        parsed_candidates = list(parser_result.candidates)
                        per_artifact_warnings.extend(parser_result.warnings)
                        parsed_derived_files = list(parser_result.derived_files)
                        if use_cache:
                            save_cached_artifact_result(
                                build_cached_artifact_result(
                                    artifact_id=artifact_id,
                                    sha256=artifact_sha256,
                                    parser_name=parser_name,
                                    parser_version=parser_version,
                                    domain_pack_id=resolved_domain_pack.pack_id,
                                    domain_pack_version=resolved_domain_pack.version,
                                    candidates=parsed_candidates,
                                    atoms=parsed_atoms,
                                    warnings=per_artifact_warnings,
                                    derived_files=parsed_derived_files,
                                )
                            )
                        cache_misses += 1
                    # A sheet row joined into one atom ("AP-1 | Access point
                    # | 4") is in no single cell: name the cells it came
                    # from, whichever parser read the workbook (the quote
                    # parser reads many). Provenance only -- never fails.
                    try:
                        from app.parsers.cell_fragments import stamp_cell_fragments
                        parsed_atoms = stamp_cell_fragments(parsed_atoms, artifact)
                    except Exception:  # noqa: BLE001
                        pass
                    # Materialize parser-emitted derived files next to
                    # the source artifact on every pass — cache or no
                    # cache.  This is what keeps OrbitBrief PDF
                    # ``structured.json`` projections in lock-step with
                    # the cached atom set.
                    if parsed_derived_files:
                        _materialize_derived_files(artifact, parsed_derived_files)
                    if parser_routing:
                        parser_routing[-1]["cache_hit"] = cache_hit

                    # One artifact must not be able to take the whole compile.
                    #
                    # Deal 5bd32822 carried an SSRS customer report —
                    # `SSRS-SL-CUS001-CustomerOut`, 15,253 rows x 43 columns —
                    # that parsed to 94,047 atoms, 48,321 of them `scope_item`,
                    # and serialised to a 319 MB cache payload. The compile ran
                    # 6.8 hours and blocked every deploy behind it, because the
                    # drain will not roll over a running compile.
                    #
                    # Dropped, not truncated: keeping the first N atoms of a
                    # 94,000-atom file is a silent, arbitrary sample of somebody
                    # else's data presented as this deal's evidence. Skipping it
                    # loudly is honest, and the routing row says exactly what
                    # happened so a real oversized scope file is visible rather
                    # than mysterious.
                    # Lines the admission regexes refused (a greeting, a
                    # sign-off) ride along as chatter atoms and are held aside
                    # below. They were never counted before they were emitted,
                    # so they are not counted now: no routing number moves.
                    kept_parsed = sum(
                        1 for _a in parsed_atoms
                        if _ADMISSION_REJECT_FLAG not in (getattr(_a, "review_flags", None) or [])
                    )
                    if _MAX_ATOMS_PER_ARTIFACT and len(parsed_atoms) > _MAX_ATOMS_PER_ARTIFACT:
                        warning = (
                            f"WARNING: {relative_name} produced {len(parsed_atoms):,} atoms "
                            f"(cap {_MAX_ATOMS_PER_ARTIFACT:,}); skipping the file. A document "
                            f"this dense is a data export, not deal evidence — raise "
                            f"SOWSMITH_MAX_ATOMS_PER_ARTIFACT if it really is scope."
                        )
                        parse_warnings.append(warning)
                        if parser_routing:
                            parser_routing[-1]["outcome"] = {
                                "status": "skipped_oversized",
                                "reason": (
                                    f"{len(parsed_atoms)} atoms exceeds the per-artifact cap "
                                    f"of {_MAX_ATOMS_PER_ARTIFACT}"
                                ),
                                "atom_count": 0,
                                "atoms_discarded": len(parsed_atoms),
                                "warning_count": len(per_artifact_warnings) + 1,
                            }
                        continue

                    candidates.extend(parsed_candidates)
                    parse_warnings.extend(per_artifact_warnings)
                    atoms.extend(parsed_atoms)
                    parser_atom_counts[parser_key] += kept_parsed
                    if parser_routing:
                        # Successful parse — record concrete outcome.
                        # Use ``ok`` when the parser produced ≥1 atom;
                        # ``ok_empty`` when it ran but produced none
                        # (e.g. an image-only PDF the parser skipped
                        # without erroring). PM_HANDOFF distinguishes
                        # so reviewers know whether a 0-atom file means
                        # "parser is healthy, just no content" vs "parser
                        # silently failed."
                        status = "ok" if kept_parsed > 0 else "ok_empty"
                        parser_routing[-1]["outcome"] = {
                            "status": status,
                            "atom_count": kept_parsed,
                            "warning_count": len(per_artifact_warnings),
                            "cache_hit": cache_hit,
                        }
                        # Parse-coverage gate: a parser that ran without
                        # error but extracted ZERO atoms is the most
                        # dangerous failure mode — silent data loss. A
                        # scanned / image-only PDF (e.g. Notes.pdf that
                        # yields 0 text), an empty sheet, or a parser that
                        # quietly bailed all land here. Surface a hard
                        # warning so the reviewer knows an input contributed
                        # nothing, instead of the file vanishing without a
                        # trace. Universal: keys off atom_count, not file type.
                        if kept_parsed == 0:
                            parse_warnings.append(
                                f"WARNING: artifact '{relative_name}' parsed cleanly "
                                f"with {parser_name} but yielded 0 atoms — no content "
                                f"extracted (possible scanned/image-only PDF, empty "
                                f"file, or unextractable layout). This input "
                                f"contributed nothing to the compile."
                            )
            except Exception as exc:  # pragma: no cover
                message = f"Failed parsing {artifact.name} ({parser_key}): {exc}"
                parse_warnings.append(message)
                parse_errors.append(message)
                if parser_routing:
                    parser_routing[-1]["outcome"] = {
                        "status": "failed_parse",
                        "reason": f"{type(exc).__name__}: {str(exc)[:280]}",
                        "atom_count": 0,
                        "warning_count": len(per_artifact_warnings),
                    }
                cache_misses += 1
            fingerprints.append(
                build_artifact_fingerprint(
                    artifact,
                    artifact_id,
                    parsed_atoms,
                    filename=relative_name,
                    parser_name=parser_name,
                    parser_version=parser_version,
                )
            )
        warnings.extend(parse_warnings)
        # Hang each rule's decision on the atom whose text it judged, so the
        # labelling UI can show WHY an atom came out the way it did and a
        # person can say whether the rule was right. Half of these decisions
        # land within 0.08 of a hand-tuned threshold, and nothing else carries
        # them off the box that ran the compile.
        #
        # Opt-in (`SOWSMITH_RULE_DECISIONS=1`): a normal compile does no work
        # here and no envelope grows.
        _attach_rule_decisions(atoms)
        # The loop is done; later stages are not per-file, so stop naming one.
        telemetry.set_stage_item("")
        telemetry.end_stage(
            stage,
            output_count=len(atoms),
            warnings=parse_warnings,
            errors=parse_errors,
        )

    # Divert parse-time pre-suppressed atoms into the suppressed sidecar. A
    # parser can mark an atom suppressed at parse time (e.g. the xlsx sheet-role
    # router emits a ``dropped_sheet`` marker for a whole sheet it routes DROP,
    # stamped ``suppressed:sheet_router``). Such atoms must never reach scope,
    # but routing them through the ledger keeps them auditable and lets an
    # omission complaint ("you missed the Lookup tab") localize to the stage
    # that removed them — instead of the sheet vanishing without a trace.
    # Universal: keys off the suppression-flag prefix, not any file type.
    # Chrome a parser still emitted as an atom -- a signature logo / badge /
    # banner read off an inline image, an e-signature page stamp -- leaves
    # the atom set here, into the ledger at stage ``chrome`` with its reason,
    # so the envelope lists it under ``suppressed_chrome`` and nothing else
    # ever carries it. See app/core/email_chrome.py.
    try:
        from app.core.email_chrome import divert_chrome

        _n_before = len(atoms)
        atoms = divert_chrome(atoms, suppressed_atoms)
        if len(atoms) != _n_before:
            warnings.append(
                f"INFO: diverted {_n_before - len(atoms)} chrome atom(s) (signature images, "
                f"e-sign stamps) to the ledger"
            )
    except Exception as exc:  # never fail a compile over chrome
        warnings.append(f"WARNING: chrome diversion failed: {type(exc).__name__}: {exc}")

    pre_suppressed = [
        a
        for a in atoms
        if any(
            str(f).startswith(SUPPRESSION_FLAG_PREFIX)
            for f in (getattr(a, "review_flags", None) or [])
        )
    ]
    if pre_suppressed:
        merge_suppressed(suppressed_atoms, pre_suppressed)
        _sup_ids = {str(getattr(a, "id", "")) for a in pre_suppressed}
        atoms = [a for a in atoms if str(getattr(a, "id", "")) not in _sup_ids]
        warnings.append(
            f"INFO: diverted {len(pre_suppressed)} parse-time suppressed "
            f"atom(s) (e.g. whole-sheet drops) to the suppressed sidecar"
        )

    # Lines a parser's admission regex refused (a greeting, a sign-off, the
    # name under it) are KEPT atoms flagged chatter, so the labeling page can
    # show them and the admission head gets its negatives. They are held out
    # of every stage from here to packetizing -- threading, dedup, the
    # substance gate, typing, entity resolution, signals, packets -- and put
    # back only for text coverage and the result. So no head ever reads one,
    # and every other atom, entity, edge and packet is exactly what it would be
    # without them.
    held_chatter = [a for a in atoms if _ADMISSION_REJECT_FLAG in (getattr(a, "review_flags", None) or [])]
    if held_chatter:
        _held_ids = {id(a) for a in held_chatter}
        atoms = [a for a in atoms if id(a) not in _held_ids]

    # Email threading: each .eml is a separate artifact, so a short reply
    # ("yes, go ahead with 36") parses as an atom with no idea what it answers.
    # Reconstruct the conversation across files (RFC In-Reply-To/References,
    # subject-norm fallback) and stamp every atom with its thread position +
    # the gist of the message it replies to. Purely additive: no atom is
    # removed, retyped, or re-id'd here — runs before dedup so the surviving
    # copy of a quoted/duplicated line keeps its thread context.
    with telemetry.stage("email_threading", input_count=len(atoms)) as stage:
        thread_summary: dict = {}
        try:
            from app.core.email_threading import thread_emails

            atoms, thread_summary = thread_emails(atoms, project_id=resolved_project_id)
            tc = int(thread_summary.get("thread_count", 0))
            mm = int(thread_summary.get("multi_message_threads", 0))
            mc = int(thread_summary.get("threaded_message_count", 0))
            if mc:
                warnings.append(
                    f"INFO: email_threading linked {mc} email(s) into {tc} "
                    f"thread(s) ({mm} multi-message)"
                )
        except Exception as exc:
            warnings.append(
                f"WARNING: email_threading failed: {type(exc).__name__}: {exc}"
            )
        telemetry.end_stage(stage, output_count=len(atoms))

    # The held chatter atoms read where their message reads: each copies the
    # thread stamp of a kept atom from the same file and message, so the
    # envelope's reading order puts "Hi Trent," above the body it opens.
    # Copying is one-way -- nothing here touches a kept atom.
    if held_chatter:
        try:
            _stamp_by_msg: dict[tuple[str, Any], dict] = {}
            for _a in atoms:
                _v = _a.value if isinstance(getattr(_a, "value", None), dict) else {}
                _et = _v.get("email_thread")
                if isinstance(_et, dict):
                    _stamp_by_msg.setdefault((str(_a.artifact_id), _v.get("message_index")), _et)
            for _a in held_chatter:
                _v = _a.value if isinstance(getattr(_a, "value", None), dict) else None
                if _v is None or "email_thread" in _v:
                    continue
                _et = _stamp_by_msg.get((str(_a.artifact_id), _v.get("message_index")))
                if _et is not None:
                    _v["email_thread"] = dict(_et)
        except Exception:  # pragma: no cover - ordering sugar, never fatal
            pass
        # Quoted copies of a held line ("Hi Megan," once per reply that quoted
        # the message) collapse to the one its message authored. They skip
        # quoted_history_dedup with every other head, so they get their own
        # pass, keyed by the message as well as the words.
        try:
            from app.core.email_threading import dedup_quoted_chatter

            _before_chatter = list(held_chatter)
            held_chatter, _dropped_chatter = dedup_quoted_chatter(held_chatter, context=atoms)
            if _dropped_chatter:
                merge_suppressed(
                    suppressed_atoms,
                    capture_suppressed(
                        _before_chatter, held_chatter,
                        stage="quoted_chatter_dedup",
                        reason="quoted copy of a greeting/sign-off its own message already holds",
                    ),
                )
                warnings.append(
                    f"INFO: quoted_chatter_dedup diverted {len(_dropped_chatter)} quoted "
                    f"copy(ies) of held chatter to the ledger"
                )
        except Exception as exc:
            warnings.append(f"WARNING: quoted_chatter_dedup failed: {type(exc).__name__}: {exc}")
        try:
            from app.core.email_threading import mark_repeated_signature_copies

            _sig_copies = mark_repeated_signature_copies(held_chatter)
            if _sig_copies:
                warnings.append(f"INFO: {_sig_copies} repeated signature line(s) marked as copies of their first email")
        except Exception as exc:
            warnings.append(f"WARNING: repeated_signature_copies failed: {type(exc).__name__}: {exc}")

    # A HubSpot note that is a pasted email is the same message, not a second
    # source -- and the fold has to happen HERE, before the first pass that
    # removes an atom.
    #
    # It ran after the general dedup once, which let a similarity pass pick the
    # winner and keep the NOTE copy: three of Alec's sentences ended up alone
    # in a file of their own, attributed to the man who pasted them. Moving it
    # ahead of semantic_dedup was not enough -- quoted_history_dedup and the
    # collapse stages run first, and they remove the very email atoms the note
    # needs to match against. Measured on 010288: 0.90 of the note is found in
    # the mail as parsed, and only 0.61 by the time the old position was
    # reached, which put it in the "ambiguous, do not guess" band and folded
    # nothing.
    #
    # email_threading is directly above and is purely additive (no atom is
    # removed, retyped or re-id'd), so the atoms here are exactly what the
    # parsers produced, with thread membership stamped on.
    #
    # Which document is earliest decides who OWNS a line two documents share,
    # in this stage and every dedup stage after it; a fold across documents
    # leaves the later document its own copy, held out of every stage (like
    # the chatter above) and put back for the result. See cross_doc_copies.
    held_copies: list = []
    _doc_order: dict = {}
    try:
        from app.core.cross_doc_copies import document_order as _document_order
        from app.core.orbitbrief_envelope import _load_manifest_provenance as _load_prov

        _doc_order = _document_order(list(atoms) + list(held_chatter), provenance=_load_prov(project_dir))
    except Exception as exc:
        warnings.append(f"WARNING: document_order failed: {type(exc).__name__}: {exc}")

    def _hold_copies(before: list, after: list, stage_name: str) -> list:
        """Mark this stage's cross-document folds as copies and hold them."""
        try:
            from app.core.cross_doc_copies import split_copies

            got = split_copies(before, after, stage=stage_name)
        except Exception as exc:  # never fail a compile over a copy
            warnings.append(f"WARNING: cross_doc_copies failed after {stage_name}: {type(exc).__name__}: {exc}")
            return []
        held_copies.extend(got)
        return got

    with telemetry.stage("pasted_note_dedup", input_count=len(atoms)) as stage:
        try:
            from app.core.pasted_note_dedup import collapse_pasted_note_duplicates

            before_paste = list(atoms)
            atoms, _pasted = collapse_pasted_note_duplicates(atoms, doc_order=_doc_order)
            _paste_copies = _hold_copies(before_paste, atoms, "pasted_note_dedup")
            # Gate on the LIST, not on the helper's report. The two disagreed
            # on live 010237: three atoms left and `_pasted` was empty, so
            # nothing reached the ledger. What was removed is the only thing
            # that decides whether a receipt is owed.
            if len(atoms) < len(before_paste):
                merge_suppressed(
                    suppressed_atoms,
                    # `reason` is required and this call never passed it, so
                    # every invocation raised TypeError into the `except
                    # Exception` below and was logged as the stage failing.
                    # The ledger call has never once run.
                    capture_suppressed(
                        before_paste, atoms + _paste_copies, stage="pasted_note_dedup",
                        reason="copy of a note/email text folded onto its original (quoted copy, later copy, or note pasted from mail)",
                    ),
                )
                warnings.append(
                    f"INFO: pasted_note_dedup folded {len(_pasted)} copies onto their originals"
                )
        except Exception as exc:
            warnings.append(f"WARNING: pasted_note_dedup failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=len(atoms))

    # Quoted-history dedup: in a long thread every reply re-quotes the whole
    # history, so the same sentence is emitted once per reply (the #010045
    # 9,452-atom flood). Drop a QUOTED echo when the same content already exists
    # in the thread as authored text or an earlier quoted copy — the authored
    # original is always kept, only redundant echoes are diverted to the ledger.
    # Runs right after threading so thread membership/order is available and
    # before the generic dedup stages so they operate on the slim set.
    with telemetry.stage("quoted_history_dedup", input_count=len(atoms)) as stage:
        try:
            from app.core.email_threading import dedup_quoted_history

            before_qh_atoms = list(atoms)
            atoms, dropped_qh = dedup_quoted_history(
                atoms, project_id=resolved_project_id
            )
            if dropped_qh:
                merge_suppressed(
                    suppressed_atoms,
                    capture_suppressed(
                        before_qh_atoms, atoms,
                        stage="quoted_history_dedup",
                        reason="quoted email history already present in the thread (authored original kept)",
                    ),
                )
                warnings.append(
                    f"INFO: quoted_history_dedup diverted {len(dropped_qh)} "
                    f"redundant quoted-history atom(s) to the ledger"
                )
            # Who said it, to whom, for which company. Runs right after
            # threading, so every downstream stage (and the labeler) can tell
            # our own account exec from the reseller's rep.
            try:
                from app.core.deal_parties import stamp_parties

                _stamped = stamp_parties(atoms)
                if _stamped:
                    warnings.append(f"INFO: deal_parties stamped who-said-it on {_stamped} atom(s)")
            except Exception as exc:
                warnings.append(f"WARNING: deal_parties failed: {type(exc).__name__}: {exc}")
            # A question and its answer are one fact. Runs after threading so
            # "Where is this site located?" can find the reply that answered
            # it, and after the quoted-history dedup so it pairs with the
            # authored original rather than a quoted echo.
            try:
                from app.core.qa_pairing import (
                    pair_across_thread,
                    pair_questions_with_answers,
                    pair_within_one_line,
                )

                # First: a line the parser split into several atoms. The other
                # two rules cannot see inside one line, so this runs before
                # them and marks those questions answered.
                _inline = pair_within_one_line(atoms)
                if _inline:
                    # The answer is now part of its question's text. Keeping the
                    # fragment as well would show the same words twice.
                    _absorbed = [a for a in atoms
                                 if (getattr(a, "value", None) or {}).get("absorbed_into")]
                    if _absorbed:
                        _before_abs = list(atoms)
                        atoms = [a for a in atoms
                                 if not (getattr(a, "value", None) or {}).get("absorbed_into")]
                        merge_suppressed(
                            suppressed_atoms,
                            capture_suppressed(
                                _before_abs, atoms,
                                stage="qa_pairing_merge",
                                reason="answer merged into its question on the same line",
                            ),
                        )
                _same = pair_questions_with_answers(atoms)
                _cross = pair_across_thread(atoms)
                if _inline or _same or _cross:
                    warnings.append(
                        f"INFO: qa_pairing joined {_inline} question(s) to an answer on the same"
                        f" split line, {_same} in the same message"
                        f" and proposed {_cross} answered by a reply"
                    )
            except Exception as exc:
                warnings.append(f"WARNING: qa_pairing failed: {type(exc).__name__}: {exc}")
        except Exception as exc:
            warnings.append(
                f"WARNING: quoted_history_dedup failed: {type(exc).__name__}: {exc}"
            )
        telemetry.end_stage(stage, output_count=len(atoms))

    # Register {artifact_id: Path} with the vision module so its leaf
    # fitz.open() calls — invoked from enrich_entities via atom
    # source_refs, which only carry basenames — can resolve to the
    # absolute on-disk path. Without this, find_table_pages and PDF
    # render fall through to "no such file" warnings and we lose
    # table-derived atoms.
    try:
        from app.core.vision_extraction import register_artifact_paths
        register_artifact_paths(artifact_paths)
    except Exception as _vp_exc:
        warnings.append(
            f"WARNING: vision artifact-path registration failed: "
            f"{type(_vp_exc).__name__}: {_vp_exc}"
        )

    # NOTHING FEEDS THIS LANE, and that is worth saying out loud.
    #
    # `CandidateAtom` is constructed in exactly two places: a sandbox
    # experiment, and `candidates.candidate_from_evidence_atom`, whose only
    # callers are tests. No parser sets `ParserOutput.candidates`. So this
    # adjudicates an empty set on every compile -- 0 atoms in, 0.03ms, on
    # every deal measured.
    #
    # The cost is not the 0.03ms, it is downstream:
    # `active_learning.build_review_queue` reads `rejected_candidates` and
    # `candidates` alongside `packets`, so TWO of its three inputs are
    # structurally empty. The queue works off packets and is degraded, not
    # broken -- but nothing said so, which is how a switched-off feature gets
    # mistaken for a working one.
    #
    # Left in place deliberately rather than deleted. The design is sound (a
    # parser proposes, an adjudicator accepts or rejects) and the `suppression`
    # head now surfaces the same class of decision with more context -- what
    # was dropped, by which stage, and what it was folded INTO. Feed this lane
    # only if that turns out not to be enough; do not feed it by reflex.
    with telemetry.stage("candidate_adjudication", input_count=len(candidates)) as stage:
        adjudication = adjudicate_candidates(candidates, artifact_paths)
        atoms.extend(adjudication.accepted_atoms)
        rejected_candidates = adjudication.rejected_candidates
        warnings.extend(adjudication.warnings)
        telemetry.end_stage(
            stage,
            output_count=len(adjudication.accepted_atoms),
            warnings=adjudication.warnings,
            errors=[],
        )

    manifest = create_manifest(resolved_project_id, fingerprints, domain_pack=resolved_domain_pack)
    manifest.parser_routing = sorted(parser_routing, key=lambda row: row["artifact_id"])
    manifest.cache_hits = cache_hits if use_cache else 0
    manifest.cache_misses = cache_misses if use_cache else len(artifacts)
    manifest.reused_artifact_ids = sorted(set(reused_artifact_ids)) if use_cache else []
    telemetry.set_compile_id(manifest.compile_id)

    replay_warnings: list[str] = []
    with telemetry.stage("source_replay", input_count=len(atoms)) as stage:
        atoms = attach_receipts_to_atoms(atoms, artifact_paths)
        receipt_summary = summarize_receipts(atoms)
        if receipt_summary["unsupported"] > 0:
            replay_warnings.append(
                f"WARNING: {receipt_summary['unsupported']} source receipts are unsupported and require manual audit"
            )
            warnings.extend(replay_warnings)
        telemetry.end_stage(stage, output_count=len(atoms), warnings=replay_warnings)

    # Pictures LINKED from a document rather than embedded in one: a drawing
    # whose URL is written in an email body or a note. ``pdf_image_vision``
    # below cannot see these -- it wants a crop saved out of a PDF page -- so
    # until now the deal held the link and never the contents. Stamp the link
    # first (idempotent; the substance gate stamps again later for anything
    # that arrives after this point), then read what it points at.
    #
    # It sits HERE, beside the PDF reader, so the atoms it makes go through
    # every stage the rest of the deal does: entity extraction, typing, dedup.
    # Emitted later they would be facts nothing else in the compile had seen.
    try:
        from app.core import linked_picture_vision
        if linked_picture_vision.enabled():
            from app.core.linked_pictures import stamp_linked_pictures

            with telemetry.stage("linked_picture_vision", input_count=len(atoms)) as stage:
                stamp_linked_pictures(atoms)
                picture_atoms = linked_picture_vision.atoms_from_linked_pictures(atoms)
                if picture_atoms:
                    atoms.extend(picture_atoms)
                telemetry.end_stage(stage, output_count=len(picture_atoms))
    except Exception as _lpv_exc:
        warnings.append(
            f"WARNING: linked_picture_vision pass failed (non-fatal): "
            f"{type(_lpv_exc).__name__}: {_lpv_exc}"
        )

    # Who supplies what, where two documents disagree. This needs BOTH supply
    # tables to exist, so it runs straight after the drawing has been read.
    #
    # On 010288 the email states outright that the vendor drawing is wrong --
    # "the 'Installer Supplied Components' are not accurate, as we provide
    # several of those pieces" -- and never says which several. Answering that
    # meant reading eighteen labels off a picture and laying them against a
    # ten-line list by hand. Both are atoms with a heading now.
    try:
        from app.core import supply_conflicts

        with telemetry.stage("supply_conflicts", input_count=len(atoms)) as stage:
            # No document table needed: every atom carries its own sender and,
            # for a drawing, the sheet it came off, which is what decides
            # whether a heading names the document's own side.
            asked = supply_conflicts.find_supply_conflicts(atoms)
            if asked:
                atoms.extend(asked)
                warnings.append(
                    f"INFO: supply_conflicts raised {len(asked)} question(s) where two "
                    f"documents hand the same part to different companies"
                )
            telemetry.end_stage(stage, output_count=len(asked))
    except Exception as _sc_exc:
        warnings.append(
            f"WARNING: supply_conflicts pass failed (non-fatal): "
            f"{type(_sc_exc).__name__}: {_sc_exc}"
        )

    # PDF embedded-image understanding (SEPARATE from schematics). Describes /
    # transcribes raster images the parser saved as image_marker atoms. Purely
    # additive + abstain-first; the whole stage is a no-op unless
    # SOWSMITH_PDF_IMAGE_VISION is set, so the default path is unchanged.
    try:
        from app.core import pdf_image_vision
        if pdf_image_vision.enabled():
            with telemetry.stage("pdf_image_vision", input_count=len(atoms)) as stage:
                image_atoms = pdf_image_vision.process_image_markers(atoms)
                if image_atoms:
                    # A picture of text the page's text layer already reads
                    # is a copy (often with its step number swapped): kept
                    # as a suppression naming the text atom, not emitted.
                    _vis_kept, _vis_copies = pdf_image_vision.split_text_layer_copies(image_atoms, atoms)
                    if _vis_copies:
                        merge_suppressed(
                            suppressed_atoms,
                            capture_suppressed(
                                image_atoms, _vis_kept,
                                stage="vision_copy_of_text_layer",
                                reason="the page's text layer already reads this line; "
                                       "the vision transcription is a copy",
                            ),
                        )
                    image_atoms = _vis_kept
                    atoms.extend(image_atoms)
                telemetry.end_stage(stage, output_count=len(image_atoms))
    except Exception as _piv_exc:
        warnings.append(
            f"WARNING: pdf_image_vision pass failed (non-fatal): "
            f"{type(_piv_exc).__name__}: {_piv_exc}"
        )

    # Hardening: enforce a confidence floor so atoms whose extractor was very
    # uncertain can't quietly govern packets.  We intentionally don't drop the
    # atoms — OrbitBrief still benefits from seeing them — we just refuse to
    # trust them without a human in the loop.
    floor_warnings: list[str] = []
    with telemetry.stage("confidence_floor", input_count=len(atoms)) as stage:
        floored = 0
        from app.core.schemas import ReviewStatus  # local import keeps top of file tidy
        for atom in atoms:
            if atom.confidence < LOW_CONFIDENCE_FLOOR:
                # The floor is doubt about a CLAIM. A provenance record the
                # parser wrote itself (a quoted-message header "From: X |
                # Sent: Y", an image/attachment marker) asserts nothing a PM
                # can review; live 010300 queued six such headers per thread.
                if _is_provenance_record(atom):
                    continue
                if atom.review_status != ReviewStatus.needs_review:
                    atom.review_status = ReviewStatus.needs_review
                if "low_confidence_floor" not in atom.review_flags:
                    atom.review_flags = sorted(set(atom.review_flags + ["low_confidence_floor"]))
                floored += 1
        if floored:
            floor_warnings.append(
                f"WARNING: {floored} atoms below confidence floor {LOW_CONFIDENCE_FLOOR:.2f} forced to needs_review"
            )
            warnings.extend(floor_warnings)
        telemetry.end_stage(stage, output_count=floored, warnings=floor_warnings)

    # v50 PROSE-LIST SPLITTER — atomize multi-fact paragraphs.
    # A single scope_item containing 6 stakeholders / 6 phases / 4
    # payment tiers becomes N child atoms. Each child inherits the
    # parent's source_ref + a sub_idx locator. Universal patterns
    # (pipe records, numbered prefixes, semicolon-parallel, bulleted
    # lines, label-prefix runs) — no customer-specific tuning.
    with telemetry.stage("prose_list_split", input_count=len(atoms)) as stage:
        split_count = 0
        try:
            from app.core.prose_list_splitter import split_prose_paragraph
            from app.core.schemas import (
                ArtifactType as _AT, AtomType as _AtomT, AuthorityClass as _Auth,
                EvidenceAtom as _EvAtom, ReviewStatus as _Rev, SourceRef as _SrcRef,
            )
            from app.core.ids import stable_id as _stable_id

            _splittable_types = {"scope_item", "entity", "raw_table_row"}
            _child_atoms: list = []
            for parent in atoms:
                _ptype = getattr(parent, "atom_type", None)
                _ptype_v = _ptype.value if hasattr(_ptype, "value") else str(_ptype or "")
                if _ptype_v not in _splittable_types:
                    continue
                _ptext = getattr(parent, "raw_text", "") or ""
                items = split_prose_paragraph(_ptext)
                if not items:
                    continue
                # Inherit source_ref from parent
                _parent_refs = list(getattr(parent, "source_refs", None) or [])
                _parent_aid = getattr(parent, "artifact_id", "") or ""
                _parent_pid = getattr(parent, "project_id", "") or ""
                # v52: detect section_path signal — if the parent atom
                # sits under a "Deliverables" / "Stakeholders" / etc.
                # heading, type the child atoms accordingly instead of
                # generic scope_item.
                _section_path = []
                if _parent_refs:
                    _loc0 = getattr(_parent_refs[0], "locator", None) or {}
                    if isinstance(_loc0, dict):
                        _section_path = _loc0.get("section_path") or []
                _section_blob = " ".join(str(s).lower() for s in _section_path)
                _child_type = _AtomT.scope_item
                _SECTION_TYPE_HINTS = {
                    "deliverable": _AtomT.deliverable,
                    "deliverables": _AtomT.deliverable,
                    "assumption": _AtomT.assumption,
                    "assumptions": _AtomT.assumption,
                    "exclusion": _AtomT.exclusion,
                    "out of scope": _AtomT.exclusion,
                    "exclusions": _AtomT.exclusion,
                    "signature": _AtomT.signatory,
                    "signatures": _AtomT.signatory,
                    "signatories": _AtomT.signatory,
                    "stakeholders": _AtomT.stakeholder,
                    "approver": _AtomT.approval_authority,
                    "approvers": _AtomT.approval_authority,
                    "approval matrix": _AtomT.approval_authority,
                    "payment schedule": _AtomT.payment_term,
                    "payment terms": _AtomT.payment_term,
                    "milestone": _AtomT.milestone_phase,
                    "milestones": _AtomT.milestone_phase,
                    "phase plan": _AtomT.milestone_phase,
                    "phase": _AtomT.milestone_phase,
                    "acceptance criteria": _AtomT.acceptance_criterion,
                    "acceptance": _AtomT.acceptance_criterion,
                    "lead time": _AtomT.lead_time_constraint,
                    "lead times": _AtomT.lead_time_constraint,
                    "cutover": _AtomT.cutover_step,
                    "cutover checklist": _AtomT.cutover_step,
                    "compliance": _AtomT.compliance_rule,
                    "data flow": _AtomT.data_flow_step,
                    "field mapping": _AtomT.system_mapping,
                    "system mapping": _AtomT.system_mapping,
                    "blackout": _AtomT.blackout_date_range,
                    "blackouts": _AtomT.blackout_date_range,
                }
                for hint, atype in _SECTION_TYPE_HINTS.items():
                    if hint in _section_blob:
                        _child_type = atype
                        break
                for sub_idx, item_text in enumerate(items):
                    _aid = _stable_id("atm", _parent_aid, "prose_split", parent.id, sub_idx)
                    _srcs: list = []
                    for r in _parent_refs[:1]:
                        # Build a fresh SourceRef carrying the parent's locator
                        # plus the sub_idx so provenance traces back to the
                        # original paragraph.
                        _loc = dict(getattr(r, "locator", None) or {})
                        _loc["prose_split_sub_idx"] = sub_idx
                        _loc["parent_atom_id"] = parent.id
                        _srcs.append(_SrcRef(
                            id=_stable_id("src", _aid),
                            artifact_id=_parent_aid,
                            artifact_type=getattr(r, "artifact_type", _AT.docx),
                            filename=getattr(r, "filename", ""),
                            locator=_loc,
                            extraction_method="prose_list_split_v50",
                            parser_version=getattr(r, "parser_version", "prose_split_v50"),
                        ))
                    _child_atoms.append(_EvAtom(
                        id=_aid,
                        project_id=_parent_pid,
                        artifact_id=_parent_aid,
                        atom_type=_child_type,
                        raw_text=item_text[:4000],
                        normalized_text=item_text.lower()[:4000],
                        value={"_prose_split": True, "_parent_atom_id": parent.id, "_sub_idx": sub_idx},
                        entity_keys=[],
                        source_refs=_srcs,
                        receipts=[],
                        authority_class=getattr(parent, "authority_class", _Auth.contractual_scope),
                        confidence=_a_shade_less_than(parent, "confidence"),
                        confidence_raw=_a_shade_less_than(parent, "confidence_raw"),
                        calibrated_confidence=_a_shade_less_than(parent, "calibrated_confidence"),
                        review_status=_Rev.auto_accepted,
                        review_flags=[],
                        parser_version="prose_split_v50",
                    ))
                split_count += 1
            if _child_atoms:
                atoms.extend(_child_atoms)
                warnings.append(f"INFO: prose-list splitter created {len(_child_atoms)} child atoms from {split_count} multi-fact paragraphs")
        except Exception as _split_exc:
            warnings.append(f"WARNING: prose_list_split failed: {type(_split_exc).__name__}: {_split_exc}")
        telemetry.end_stage(stage, output_count=split_count)

    # v57 PIPELINE REORDER (#10): prune BEFORE the expensive per-atom LLM
    # stages (enrich_entities, typed_atom_classification). Both pruners
    # below are deterministic and depend on neither entity_keys nor typed
    # classification, so running them up front shrinks the atom set the
    # LLM has to chew through WITHOUT changing the final atom set — the
    # same atoms are dropped, just before they burn an LLM call instead
    # of after.
    with telemetry.stage("duplicate_atom_collapse", input_count=len(atoms)) as stage:
        before_atoms = list(atoms)
        before = len(atoms)
        try:
            from app.core.entity_resolution import collapse_duplicate_atoms
            atoms = collapse_duplicate_atoms(atoms)
            # A "near duplicate" whose words no survivor carries is a
            # different line (numbered steps 6-15 on a two-column PDF page).
            from app.core.suppression_ledger import keep_unsurvived_lines
            atoms, _kept_back = keep_unsurvived_lines(before_atoms, atoms, stage="duplicate_atom_collapse")
            if _kept_back:
                warnings.append(
                    f"INFO: duplicate_atom_collapse kept {len(_kept_back)} lines no survivor contains"
                )
        except Exception as exc:
            warnings.append(f"WARNING: duplicate_atom_collapse failed: {type(exc).__name__}: {exc}")
        dropped = before - len(atoms)
        if dropped > 0:
            merge_suppressed(
                suppressed_atoms,
                capture_suppressed(
                    before_atoms, atoms,
                    stage="duplicate_atom_collapse",
                    reason="collapsed as an intra-document duplicate of another atom",
                ),
            )
            warnings.append(f"INFO: collapsed {dropped} duplicate atoms (intra-doc)")
        telemetry.end_stage(stage, output_count=len(atoms))

    # Execution-block hygiene: SOW signature pages ("Signature:",
    # "Name: ___ Date: ___", "Services By: PurTera | Agreed By:") get
    # swept into scope_item / raw_table_row atoms and masquerade as
    # scope. Drop them up front so they never reach scope_truth or burn
    # an LLM enrich / classification call.
    with telemetry.stage("execution_boilerplate_drop", input_count=len(atoms)) as stage:
        before_bp_atoms = list(atoms)
        before_bp = len(atoms)
        try:
            from app.core.entity_hygiene import drop_execution_boilerplate
            atoms = drop_execution_boilerplate(atoms)
        except Exception as exc:
            warnings.append(f"WARNING: execution_boilerplate_drop failed: {type(exc).__name__}: {exc}")
        dropped_bp = before_bp - len(atoms)
        if dropped_bp > 0:
            merge_suppressed(
                suppressed_atoms,
                capture_suppressed(
                    before_bp_atoms, atoms,
                    stage="execution_boilerplate_drop",
                    reason="signature / execution-block boilerplate, not deal scope",
                ),
            )
            warnings.append(f"INFO: dropped {dropped_bp} signature/execution-block boilerplate atoms")
        telemetry.end_stage(stage, output_count=len(atoms))

    # Pre-enrich table rollup — universal backstop for money-bearing
    # spreadsheet tables the parser's sheet-classifier missed. A 9 MB rate
    # card otherwise lands here as tens of thousands of per-row raw_table_row
    # atoms, each dragged through the per-atom LLM enrich pass (hours) and
    # flooding the training store with low-diversity catalog rows. Folding
    # each high-cardinality group losslessly into one pricing_assumption
    # summary atom (value.rows preserved) collapses cost and training noise
    # while leaving the deliverable unchanged (build_bill_of_materials and the
    # commercial_summary packet already read value.rows). Runs BEFORE enrich
    # so the expensive stages never see the exploded cardinality.
    with telemetry.stage("table_rollup", input_count=len(atoms)) as stage:
        before_tr_atoms = list(atoms)
        before_tr = len(atoms)
        try:
            from app.core.table_rollup import roll_up_table_rows
            atoms, tr_stats = roll_up_table_rows(atoms)
            atoms = _keep_unsurvived_document_rows(before_tr_atoms, atoms, warnings)
        except Exception as exc:
            tr_stats = {}
            warnings.append(f"WARNING: table_rollup failed: {type(exc).__name__}: {exc}")
        folded_tr = before_tr - len(atoms)
        _tr_kept_ids = {id(a) for a in atoms}
        if any(id(a) not in _tr_kept_ids for a in before_tr_atoms):
            merge_suppressed(
                suppressed_atoms,
                capture_suppressed(
                    before_tr_atoms, atoms,
                    stage="table_rollup",
                    reason="high-cardinality money-bearing table folded into pricing rollup (value.rows preserved)",
                ),
            )
            warnings.append(
                f"INFO: table_rollup folded {tr_stats.get('rows_folded', folded_tr)} rows "
                f"across {tr_stats.get('groups_folded', 0)} table(s) into "
                f"{tr_stats.get('summary_atoms', 0)} pricing rollup atom(s)"
            )
        telemetry.end_stage(stage, output_count=len(atoms))

    enrich_warnings: list[str] = []
    with telemetry.stage("enrich_entities", input_count=len(atoms)) as stage:
        # Universal entity extraction — populates atom.entity_keys for any
        # atom whose parser hardcoded an empty list.  Without this, the
        # downstream graph_builder anchors land on `device:unknown` and
        # quantity_conflict edges never form.  See PRODUCTION_GAPS.md P0.2.
        atoms_enriched, keys_added = enrich_entity_keys(atoms, resolved_domain_pack)
        if atoms_enriched:
            enrich_warnings.append(
                f"INFO: enriched {atoms_enriched} atoms with {keys_added} entity keys "
                f"(parser-supplied entity_keys preserved)"
            )
        telemetry.end_stage(stage, output_count=atoms_enriched, warnings=enrich_warnings)
    warnings.extend(enrich_warnings)

    # Pre-classification shadow collapse: a single table row is emitted as
    # multiple atoms (raw_table_row intermediate + docx_table_row_v1 scope_item
    # + any schema-enriched type). Without this, the classifier sees each fact
    # 2-3x and may assign the copies conflicting types before the post-dedup
    # stage collapses them. Collapsing the same-text cross-type duplicates HERE
    # means the classifier sees ONE clean atom per fact (the most-specific type).
    # Pure/deterministic (no embedder); the later semantic_dedup stage still runs.
    with telemetry.stage("pre_classify_dedup", input_count=len(atoms)) as stage:
        _before_pcd = list(atoms)
        try:
            from app.core.semantic_dedup import cross_type_dedup_atoms
            atoms = cross_type_dedup_atoms(atoms, doc_order=_doc_order)
            _pcd_copies = _hold_copies(_before_pcd, atoms, "pre_classify_dedup")
            _dropped_pcd = len(_before_pcd) - len(atoms)
            if _dropped_pcd > 0:
                merge_suppressed(
                    suppressed_atoms,
                    capture_suppressed(
                        _before_pcd, atoms + _pcd_copies,
                        stage="pre_classify_dedup",
                        reason="same-text cross-type shadow collapsed before classification",
                    ),
                )
                warnings.append(
                    f"INFO: pre_classify_dedup collapsed {_dropped_pcd} duplicate-row shadow atoms"
                )
        except Exception as exc:
            warnings.append(f"WARNING: pre_classify_dedup failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=len(atoms))

    # Is each document about THIS deal's job? A programme customer's deal
    # carries other jobs' mail and packing lists (010162: a kiosk close-down
    # beside the SD-WAN scope). Judged per document through decide() -- a PM's
    # correction first, then the model -- and only a confident "other job" sets
    # a document aside. Lossless: its atoms go to the suppression ledger.
    with telemetry.stage("document_job_scope", input_count=len(atoms)) as stage:
        _djs_dropped = 0
        _djs_notes: list[str] = []
        try:
            from app.core import document_job_scope as _djs

            if _djs.enabled():
                _deal_name = _djs.deal_name_from_manifest(project_dir)
                _before_djs = list(atoms)
                atoms, _dropped_djs, _djs_verdicts = _djs.judge_documents(
                    atoms, deal_name=_deal_name, project_id=resolved_project_id,
                    project_dir=project_dir,
                )
                # Every conversation's verdict goes to the trace, kept or not,
                # so a brief built from the wrong job can be read back to the
                # judgement that let it in.
                _djs_notes.extend(_djs.verdict_note(_v) for _v in _djs_verdicts)
                if _dropped_djs:
                    # Each set-aside conversation carries ITS reason (who said
                    # other_job, how sure, about which title) onto every atom it
                    # drops -- a drop with no recorded reason cannot be audited.
                    _djs_kept_ids = {id(_a) for _a in atoms}
                    _djs_reason_by_id: dict[str, str] = {}
                    for _v in _djs_verdicts:
                        if _v.get("verdict") == "other_job":
                            for _aid in _v.get("atom_ids") or []:
                                _djs_reason_by_id.setdefault(_aid, _v.get("reason") or "")
                    _djs_default = "document describes another job for this customer, not the work this deal is named for"
                    _djs_groups: dict[str, list] = {}
                    for _a in _before_djs:
                        if id(_a) in _djs_kept_ids:
                            continue
                        _aid = str(getattr(_a, "id", "") or "")
                        _djs_groups.setdefault(_djs_reason_by_id.get(_aid) or _djs_default, []).append(_a)
                    for _reason, _group in _djs_groups.items():
                        merge_suppressed(
                            suppressed_atoms,
                            capture_suppressed(_group, [], stage="document_job_scope", reason=_reason),
                        )
                    _djs_dropped = len(_dropped_djs)
                warnings.extend(_djs_notes)
        except Exception as exc:
            warnings.append(f"WARNING: document_job_scope failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=_djs_dropped, warnings=_djs_notes)

    # v47 typed-atom classification — promotes scope_item / entity
    # into the rich taxonomy (milestone_phase, stakeholder, bom_line,
    # commercial_total, payment_term, requirement, acceptance_criterion,
    # electrical_acceptance_test, compliance_*, ...). LLM-driven so
    # it generalises across customer terminology variations without
    # hardcoded regex column headers.
    with telemetry.stage("typed_atom_classification", input_count=len(atoms)) as stage:
        promoted = 0
        try:
            from app.core.typed_atom_classifier import classify_atoms
            promoted = classify_atoms(atoms)
        except Exception as exc:
            warnings.append(f"WARNING: typed_atom_classifier failed: {type(exc).__name__}: {exc}")
        if promoted:
            warnings.append(f"INFO: typed-atom classifier promoted {promoted} atoms from scope_item/entity")
        # A callout read off a drawing ("Solar Panel", "Cell Modem") is a
        # label, never a site_infrastructure fact (010246).
        try:
            from app.core.diagram_labels import retype_diagram_labels

            _dl = retype_diagram_labels(atoms)
            if _dl:
                warnings.append(f"INFO: {_dl} diagram callout(s) kept as diagram_label rejects")
        except Exception as exc:
            warnings.append(f"WARNING: diagram_labels failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=promoted)

    # Work-order reassembly: the per-atom classifier answers "is this span a
    # task?" one sentence at a time, and a job is not stated in one sentence.
    # This stage judges each DOCUMENT for relevance through decide() (so the
    # judgement is correctable), reads what survives together, and mints the
    # resulting work lines as task atoms whose provenance is inherited from the
    # real atoms they were summarised from. Opt-in: SOWSMITH_WORK_ORDER=1.
    with telemetry.stage("work_order", input_count=len(atoms)) as stage:
        _wo_minted = 0
        try:
            from app.core import work_order as _wo

            if _wo.enabled():
                try:
                    from app.core import document_job_scope as _djs_name

                    _wo_deal = _djs_name.deal_name_from_manifest(project_dir)
                except Exception:
                    _wo_deal = ""
                atoms, _wo_minted, _wo_report = _wo.apply_work_order(
                    atoms, project_id=resolved_project_id, deal_name=_wo_deal,
                )
                if _wo_minted:
                    warnings.append(
                        f"INFO: work_order reassembled {len(_wo_report['kept_docs'])} "
                        f"relevant document(s) into {_wo_minted} work line(s)"
                    )
                for _dropped in _wo_report["dropped_docs"]:
                    warnings.append(f"INFO: work_order set aside {_dropped}")
                _wo_summary = _wo_report.get("summary") or {}
                if _wo_summary.get("one_line_summary"):
                    warnings.append(
                        f"INFO: work_order reads the job as: "
                        f"{_wo_summary['one_line_summary']}"
                    )
                if _wo_summary.get("site_count"):
                    warnings.append(
                        f"INFO: work_order counts {_wo_summary['site_count']} site(s)"
                    )
        except Exception as exc:
            warnings.append(f"WARNING: work_order failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=_wo_minted)

    # Geographic fallback: a deal whose only locational anchor is a bare
    # "City, ST ZIP" in a notes file produces zero physical_site atoms,
    # an empty site_readiness, and a RED "no confirmed site" brief. When
    # no real site exists, mine a City/State/ZIP anchor and emit one
    # low-confidence (needs_review) physical_site so the deal anchors
    # somewhere instead of going blank-RED. No-op when a real site exists.
    # Type-sanity guardrail: classification labels atoms in isolation, so
    # commercial meta ("28.57% margin", "99 pricing lines") leaks into the
    # quantity bucket and the deal's headline count ("110 displays") stays
    # buried in prose. This deterministic pass demotes non-deliverable
    # quantities to pricing_assumption and surfaces strong prose counts as
    # quantity atoms. No LLM, no customer tuning.
    with telemetry.stage("atom_type_sanity", input_count=len(atoms)) as stage:
        sanity_changed = 0
        # `apply_type_sanity` RETURNS an atom list, and it is shorter than the
        # one it was given: 37 atoms on live 010238, 293 on 010237. Nothing
        # recorded that, not even a warning, so those atoms left the compile
        # with no receipt and every audit that reads the suppression ledger --
        # which is all of them, including `_tools/_phase3_audit.py` -- was
        # blind to the stage. One of the 37 was the only atom stating the
        # deal's account number and its contract effective and expiry dates.
        _before_ats = list(atoms)
        try:
            from app.core.atom_type_sanity import apply_type_sanity
            # ``artifact_paths`` is the compile's authoritative artifact set
            # ({artifact_id: Path}); the promotion gate uses it to cap any
            # atom whose source does not resolve to a real file.
            atoms, _demoted, _surfaced = apply_type_sanity(
                atoms,
                project_id=resolved_project_id,
                artifact_ids=set(artifact_paths),
                documents=artifact_paths,
            )
            sanity_changed = _demoted + _surfaced
            if _demoted:
                warnings.append(f"INFO: atom_type_sanity demoted {_demoted} non-deliverable quantity atom(s) to pricing_assumption")
            if _surfaced:
                warnings.append(f"INFO: atom_type_sanity surfaced {_surfaced} headline quantity atom(s) from prose")
            _ats_gone = capture_suppressed(
                _before_ats, atoms, stage="atom_type_sanity",
                reason="atom whose type could not be reconciled with its text",
            )
            if _ats_gone:
                merge_suppressed(suppressed_atoms, _ats_gone)
                warnings.append(
                    f"INFO: atom_type_sanity dropped {len(_ats_gone)} atom(s)"
                )
        except Exception as exc:
            warnings.append(f"WARNING: atom_type_sanity failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=sanity_changed)

    # Span admission (#span_admission seam): for atoms still sitting in a
    # generic/retained type (scope_item/entity/deal_metadata/site_note), ask the
    # decide() STORE whether they should be re-typed into a recovered specific
    # type (milestone_phase, requirement, acceptance_criterion, quantity,
    # commercial category, ...). STORE-ONLY (no LLM), guess-free (abstain →
    # untouched). This makes RECALL text-ruleable: a PM teaches the system to
    # catch a missed class by adding a correction, no code change. Flag-gated so
    # it is a no-op in production until enabled AND a feedback store is wired.
    # Who supplies each hardware line -- us, or the customer? A kit's
    # materials are what we buy; a hardware list in the documents is often
    # the customer's own (010095: four SHI-supplied lines became BOM rows).
    # Judged per line through decide(); only a store hit or a confident model
    # verdict stamps value.supplied_by. Nothing is dropped.
    with telemetry.stage("bom_owner", input_count=len(atoms)) as stage:
        _bo_stamped = 0
        _bo_notes: list[str] = []
        try:
            from app.core import bom_owner as _bo

            if _bo.enabled():
                _bo_stamped, _bo_verdicts = _bo.stamp_bom_owners(atoms, project_id=resolved_project_id)
                for _v in _bo_verdicts:
                    if _v["verdict"]:
                        _bo_notes.append(f"INFO: bom_owner {_v['verdict']} ({_v['source']} {_v['confidence']:.2f}): {_v['text'][:80]}")
                warnings.extend(_bo_notes)
        except Exception as exc:
            warnings.append(f"WARNING: bom_owner failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=_bo_stamped, warnings=_bo_notes)

    with telemetry.stage("span_admission", input_count=len(atoms)) as stage:
        readmitted = 0
        import os as _os_sa
        if _os_sa.environ.get("SOWSMITH_SPAN_ADMISSION", "") in ("1", "true", "yes", "on"):
            try:
                from app.core.span_admission import readmit_atom_types
                readmitted = readmit_atom_types(atoms)
                if readmitted:
                    warnings.append(
                        f"INFO: span_admission re-typed {readmitted} retained atom(s) "
                        f"into recovered types via the store"
                    )
            except Exception as exc:
                warnings.append(f"WARNING: span_admission failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=readmitted)

    # Open-question resolution: an open_question whose answer already
    # exists in the corpus (shares an answer-bearing entity key with a
    # fact atom) is flagged answered so it stops surfacing as a PM
    # blocker. Genuine gaps are added later from the SRL schema, not from
    # literal "?" detection. Deterministic, no LLM.
    with telemetry.stage("open_question_resolution", input_count=len(atoms)) as stage:
        resolved_q = 0
        dropped_noise_q = []
        try:
            from app.core.open_question_resolution import (
                filter_unhelpful_open_questions,
                resolve_open_questions,
            )
            resolved_q = resolve_open_questions(atoms)
            if resolved_q:
                warnings.append(f"INFO: open_question_resolution flagged {resolved_q} already-answered question(s)")
            # ...and the ones a teacher closed. Key overlap cannot see an answer
            # that names nothing in common with its question -- "the notes I
            # sent over, that has everything" answers "another way of sharing
            # the recording?" and shares not one entity with it. Live 010180:
            # that question is still needs_review on every compile, two months
            # after the customer withdrew it, and a labeler had already linked
            # the pair.
            try:
                from app.core.taught_answers import resolve_taught_answers

                _taught_q = resolve_taught_answers(atoms, project_id=resolved_project_id)
                if _taught_q:
                    warnings.append(
                        f"INFO: open_question_resolution closed {_taught_q} question(s) a teacher "
                        f"had already answered"
                    )
            except Exception as exc:
                warnings.append(f"WARNING: taught_answers failed: {type(exc).__name__}: {exc}")
            atoms, dropped_noise_q = filter_unhelpful_open_questions(atoms)
            if dropped_noise_q:
                # Held, not deleted: out of every head from here on, exactly as
                # before, but put back into the result beside the held chatter
                # (flagged answered_in_corpus / not_pm_actionable_question), so
                # a question somebody asked is still an atom a labeler sees.
                # It used to go to the suppression sidecar, which the labeling
                # page does not show -- 010087 lost "How many devices per
                # school?" that way.
                held_chatter.extend(dropped_noise_q)
                warnings.append(
                    f"INFO: open_question_quality_filter held {len(dropped_noise_q)} "
                    f"non-actionable or answered question atom(s) out of the heads"
                )
        except Exception as exc:
            warnings.append(f"WARNING: open_question_resolution failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=resolved_q + len(dropped_noise_q))

    with telemetry.stage("site_geo_fallback", input_count=len(atoms)) as stage:
        added_geo = 0
        import os as _os_geo
        if _os_geo.environ.get("SOWSMITH_DISABLE_GEO_FALLBACK"):
            # Opt-out for bulk / training runs. On site-rich deals the geo
            # inference short-circuits to 0 (real sites already exist), but the
            # LLM-backed vendor-suppression sub-step still runs one call per
            # site serially — pure wall-time for ~zero net atoms. Skip the
            # whole stage when this flag is set. Default-off: production keeps
            # the vendor gate (which is what stops PurTera's own address from
            # being mistaken for a job site).
            warnings.append("INFO: site_geo_fallback skipped (SOWSMITH_DISABLE_GEO_FALLBACK set)")
        else:
            try:
                from app.core.site_geo_fallback import (
                    enrich_site_geo,
                    geo_fallback_sites,
                    suppress_vendor_sites,
                )
                # 0) Fill city/state/ZIP on sites we DID detect but whose
                #    address came through as one lumped string. Runs before
                #    the fallback because it changes nothing about whether a
                #    site exists — it only makes an existing one locatable.
                enriched_geo = enrich_site_geo(atoms)
                if enriched_geo:
                    warnings.append(
                        f"INFO: site_geo_fallback enriched {enriched_geo} physical_site "
                        f"atom(s) with city/state/ZIP recovered from the document"
                    )
                # 1) Infer fallback physical_site atoms from bare City/State/ZIP
                #    anchors. This must run FIRST: a vendor letterhead address
                #    ("PurTera LLC … Alpharetta, GA 30009") only becomes a
                #    physical_site atom HERE, so the suppression gate below has
                #    nothing to act on until these exist.
                geo_atoms = geo_fallback_sites(atoms, project_id=resolved_project_id)
                if geo_atoms:
                    atoms.extend(geo_atoms)
                    added_geo = len(geo_atoms)
                    warnings.append(
                        f"INFO: site_geo_fallback inferred {added_geo} physical_site atom(s) "
                        f"from City/State/ZIP (no confirmed site found)"
                    )
                # 1b) A place the documents name that no site carries yet -- a
                #    "City, ST" on a list, a facility in prose, a Location column
                #    -- judged with the lines around it (relation
                #    geo_mention_role: store first, model when it abstains).
                try:
                    from app.core.site_geo_fallback import geo_mention_sites as _geo_mentions

                    mention_atoms = _geo_mentions(atoms, project_id=resolved_project_id)
                except Exception as _exc:  # pragma: no cover - never break the stage
                    mention_atoms = []
                    warnings.append(f"WARNING: site_geo_mention failed: {type(_exc).__name__}: {_exc}")
                if mention_atoms:
                    atoms.extend(mention_atoms)
                    added_geo += len(mention_atoms)
                    for _m in mention_atoms:
                        _mv = getattr(_m, "value", {}) or {}
                        warnings.append(
                            f"INFO: site_geo_mention {_mv.get('name')} ({_mv.get('geo_mention_source')} "
                            f"{float(_mv.get('geo_mention_confidence') or 0):.2f}): {str(_mv.get('mention') or '')[:60]}"
                        )
                # 2) Demote any physical_site that is actually the vendor's own
                #    letterhead / billing address (semantic role gate, LLM-backed;
                #    a no-op when the LLM is unreachable, only one site exists, or
                #    suppression would remove every site).
                before_vendor_atoms = list(atoms)
                atoms, dropped_vendor = suppress_vendor_sites(
                    atoms, project_id=resolved_project_id
                )
                if dropped_vendor:
                    merge_suppressed(
                        suppressed_atoms,
                        capture_suppressed(
                            before_vendor_atoms, atoms,
                            stage="site_geo_fallback",
                            reason="vendor / selling-party letterhead address, not a job site",
                        ),
                    )
                    warnings.append(
                        f"INFO: site_geo_fallback suppressed {dropped_vendor} "
                        f"vendor/letterhead address(es) misread as job sites"
                    )
            except Exception as exc:
                warnings.append(f"WARNING: site_geo_fallback failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=added_geo)

    # Receipt backfill: source_replay (stage 4) runs before the late
    # atom-creating stages (typed_atom_classification, site_geo_fallback),
    # so any atom *born* after replay carries source_refs but empty
    # receipts — which the quality gate rejects ("no receipts while source
    # files are available"). Re-replay only those stragglers so every atom
    # reaching the gate is provenance-complete. Idempotent: atoms that
    # already have receipts are untouched, and this generalises to any
    # future stage that mints atoms post-replay.
    with telemetry.stage("receipt_backfill", input_count=len(atoms)) as stage:
        backfilled = 0
        try:
            for atom in atoms:
                if getattr(atom, "source_refs", None) and not getattr(atom, "receipts", None):
                    atom.receipts = replay_atom_receipts(atom, artifact_paths)
                    backfilled += 1
        except Exception as exc:
            warnings.append(f"WARNING: receipt_backfill failed: {type(exc).__name__}: {exc}")
        if backfilled:
            warnings.append(f"INFO: receipt_backfill attached receipts to {backfilled} late-created atom(s)")
        telemetry.end_stage(stage, output_count=backfilled)

    # v52: semantic dedup by entity key. Catches the cases the text-based
    # v48 collapse misses — same fact extracted via 3 paths (schema /
    # prose / LLM bridge) with different text shapes but same phase_id /
    # req_id / sku / email. Drops milestone_phase from 23→6, requirement
    # from 19→5, etc., losslessly (loser fields merged into winner).
    with telemetry.stage("semantic_dedup", input_count=len(atoms)) as stage:
        # A conversation repeats itself; two tellings of one commitment are one
        # commitment. Speech only — two similar lines in a document are two facts.
        from app.core.semantic_dedup import collapse_repeated_speech

        # Snapshot BEFORE the speech collapse: its drops used to happen ahead
        # of the ledger snapshot below, so a collapsed utterance left no atom
        # AND no suppression entry -- nothing a labeller could find.
        before_sem_atoms = list(atoms)
        _before_speech = len(atoms)
        # Only CLAIMS collapse. An untyped utterance (raw_utterance) asserts
        # nothing, so a repeat of it cannot double-count anything; folding it
        # into a longer line by word overlap took real turns off the page
        # ("the only region that won't have a stack coordinator" vanished
        # into an earlier, longer turn that shared its words).
        from app.core.utterance_typing import is_untyped_speech as _is_untyped_speech

        _untyped = {id(a) for a in atoms if _is_untyped_speech(a)}
        _speech_kept = {
            id(a) for a in collapse_repeated_speech([a for a in atoms if id(a) not in _untyped])
        }
        atoms = [a for a in atoms if id(a) in _untyped or id(a) in _speech_kept]
        if len(atoms) != _before_speech:
            warnings.append(
                f"INFO: collapsed {_before_speech - len(atoms)} repeated spoken claim(s)"
            )
        before_sem = len(atoms)
        try:
            from app.core.semantic_dedup import (
                cross_type_dedup_atoms,
                semantic_dedup_atoms,
            )
            # A party's signature-page address must stop being a site BEFORE
            # the location-bucket merge, or it is folded into a real site and
            # takes that site's city, state and ZIP with it. Live 010300: HQ
            # "2970 Brandywine Rd, STE 200, Atlanta, GA 30641" came out as
            # "2970 Brandywine Rd", no city, alias "vernon hills".
            try:
                from app.core.party_address_veto import veto_party_page_sites
                _party_early = veto_party_page_sites(atoms)
                if _party_early:
                    warnings.append(f"INFO: {_party_early} signature-page address(es) kept as party_address before dedup")
            except Exception as exc:
                warnings.append(f"WARNING: party_address_veto failed: {type(exc).__name__}: {exc}")
            atoms = semantic_dedup_atoms(atoms, doc_order=_doc_order)
            # Cross-type pass: the same sentence emitted as raw_table_row +
            # scope_item + service_line + task collapses to the single most-
            # specific type. semantic_dedup keys with atom_type so it can't
            # catch these; without this, one table row inflates scope_truth
            # and the scorecards four-fold.
            before_xt = len(atoms)
            atoms = cross_type_dedup_atoms(atoms, doc_order=_doc_order)
            dropped_xt = before_xt - len(atoms)
            if dropped_xt > 0:
                warnings.append(
                    f"INFO: cross_type_dedup collapsed {dropped_xt} same-text cross-type atoms"
                )
        except Exception as exc:
            warnings.append(f"WARNING: semantic_dedup failed: {type(exc).__name__}: {exc}")
        dropped_sem = before_sem - len(atoms)
        _sem_notes: list[str] = []
        _sem_copies = _hold_copies(before_sem_atoms, atoms, "semantic_dedup")
        # Measured from the snapshot, so a turn only the speech collapse
        # dropped still gets its suppression entry.
        if len(before_sem_atoms) > len(atoms):
            merge_suppressed(
                suppressed_atoms,
                capture_suppressed(
                    before_sem_atoms, atoms + _sem_copies,
                    stage="semantic_dedup",
                    reason="semantic/cross-type duplicate collapsed into a canonical atom",
                ),
            )
            warnings.append(f"INFO: semantic_dedup collapsed {dropped_sem} duplicate-by-key atoms")
            _kept_ids = {id(a) for a in atoms}
            _sem_notes = _dropped_atom_notes("semantic_dedup", [a for a in before_sem_atoms if id(a) not in _kept_ids])
        telemetry.end_stage(stage, output_count=len(atoms), warnings=_sem_notes)

    # Two documents, one clause template, different figures ($500 vs $300
    # cancellation fee, two weeks vs five business days, ZIP 30641 vs 30341 —
    # live 010300's two NewBold PSOWs). Dedup rightly keeps both atoms; the
    # disagreement itself becomes an open_question naming both sources. Runs
    # after dedup so agreeing copies are already one atom.
    with telemetry.stage("cross_document_conflicts", input_count=len(atoms)) as stage:
        _xdoc_n = 0
        try:
            from app.core.cross_document_conflicts import find_cross_document_conflicts

            _xdoc = find_cross_document_conflicts(atoms, project_id=resolved_project_id)
            if _xdoc:
                atoms = atoms + _xdoc
                _xdoc_n = len(_xdoc)
                warnings.append(f"INFO: cross_document_conflicts raised {_xdoc_n} open question(s) on clauses that differ between documents")
        except Exception as exc:
            warnings.append(f"WARNING: cross_document_conflicts failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=_xdoc_n)

    # A per-site document speaks for its own rows. The name-mention linker only
    # joins an atom to a site when the site's NAME is in the atom's own text,
    # which a per-site SOW says exactly once -- in the row that becomes the site
    # atom. Marion County resolved ten schools and linked nothing to them: 10 of
    # 486 atoms carried a site key, and those 10 WERE the sites. This runs after
    # dedup, so the sites are already canonical.
    with telemetry.stage("site_provenance_join", input_count=len(atoms)) as stage:
        site_join_n = 0
        try:
            from app.core.party_address_veto import veto_party_page_sites
            from app.core.site_provenance_join import join_atoms_to_document_site

            # An address on a signature page is a party's, not a site's. Done
            # here, before the join, so a document never "resolves to exactly
            # one site" that is its own signatory's mailing address.
            _party_n = veto_party_page_sites(atoms)
            if _party_n:
                warnings.append(f"INFO: {_party_n} address(es) on signature pages kept as party_address, not sites")
            site_join_n = join_atoms_to_document_site(atoms)
            if site_join_n:
                warnings.append(
                    f"INFO: site_provenance_join attached a site to {site_join_n} atom(s) "
                    f"from documents that resolve to exactly one site"
                )
        except Exception as exc:
            warnings.append(
                f"WARNING: site_provenance_join failed: {type(exc).__name__}: {exc}"
            )
        telemetry.end_stage(stage, output_count=site_join_n)

    # Stakeholder atoms were deliberately left uncollapsed by semantic_dedup so
    # each instance could still be tagged with its OWN document's site above --
    # a person common to every one of ten per-site SOWs (a district-wide backup
    # contact) must accumulate all ten site: keys, not the one its dedup
    # survivor happened to inherit. Collapse them now, after the join, so
    # _merge_atom_metadata's existing entity_keys union does that accumulation.
    with telemetry.stage("stakeholder_dedup", input_count=len(atoms)) as stage:
        try:
            from app.core.semantic_dedup import dedupe_stakeholder_atoms

            before_sh = len(atoms)
            _before_sh_atoms = list(atoms)
            atoms = dedupe_stakeholder_atoms(atoms, doc_order=_doc_order)
            _sh_copies = _hold_copies(_before_sh_atoms, atoms, "stakeholder_dedup")
            dropped_sh = before_sh - len(atoms)
            if dropped_sh > 0:
                warnings.append(
                    f"INFO: stakeholder_dedup collapsed {dropped_sh} duplicate "
                    f"stakeholder identity atom(s)"
                )
                _kept_sh = {id(a) for a in atoms}
                _sh_notes = _dropped_atom_notes("stakeholder_dedup", [a for a in _before_sh_atoms if id(a) not in _kept_sh])
                # The notes above are telemetry: they are read by a person
                # looking at one compile, not by the audits, which read the
                # suppression ledger. Without this the stage dropped 17 atoms
                # on live 010237 and the ledger showed none of them.
                merge_suppressed(
                    suppressed_atoms,
                    capture_suppressed(
                        _before_sh_atoms, atoms + _sh_copies, stage="stakeholder_dedup",
                        reason="duplicate stakeholder identity folded into another record",
                    ),
                )
        except Exception as exc:
            warnings.append(f"WARNING: stakeholder_dedup failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=len(atoms), warnings=locals().get("_sh_notes") or [])

    with telemetry.stage("note_provenance_backfill", input_count=len(atoms)) as stage:
        note_prov_n = 0
        try:
            from app.core.note_provenance_backfill import ensure_hubspot_note_provenance

            atoms, note_prov_n = ensure_hubspot_note_provenance(
                atoms,
                project_id=resolved_project_id,
                artifact_paths=artifact_paths,
            )
            if note_prov_n:
                warnings.append(
                    f"INFO: note_provenance_backfill minted {note_prov_n} provenance atom(s) "
                    f"for HubSpot notes that lost atoms in dedup"
                )
        except Exception as exc:
            warnings.append(f"WARNING: note_provenance_backfill failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=note_prov_n)

    # A note with real text that carried files says so on its own header atom,
    # so a reader of the note sees what was attached to it. (A "Note"-only
    # note never got this far -- _iter_artifacts dropped it.)
    try:
        from app.core.note_attachments import note_attachment_links

        _carried = note_attachment_links(project_dir).notes
        if _carried:
            _rel_by_id: dict[str, str] = {}
            for _aid, _p in artifact_paths.items():
                try:
                    _rel_by_id[_aid] = str(Path(_p).relative_to(project_dir)).replace("\\", "/")
                except ValueError:
                    _rel_by_id[_aid] = Path(_p).name
            for _atom in atoms:
                _v = _atom.value if isinstance(getattr(_atom, "value", None), dict) else None
                if not _v or _v.get("kind") != "hubspot_note_meta":
                    continue
                _link = _carried.get(_rel_by_id.get(str(_atom.artifact_id or ""), ""))
                if _link:
                    _v["attachments"] = [a["filename"] for a in _link["attachments"]]
                    _v["attachment_ids"] = list(_link["attachment_ids"])
    except Exception as exc:
        warnings.append(f"WARNING: note_attachment_links failed: {type(exc).__name__}: {exc}")

    # HubSpot notes / short email bullets often carry quote-level work units
    # before a SOW exists, but the type classifier may leave them as scope_item
    # or open_question because they are terse or phrased as a request.
    with telemetry.stage("task_atom_backfill", input_count=len(atoms)) as stage:
        task_backfill_n = 0
        try:
            from app.core.task_atom_backfill import backfill_quote_task_atoms

            atoms, task_backfill_n = backfill_quote_task_atoms(
                atoms, project_id=resolved_project_id
            )
            if task_backfill_n:
                warnings.append(
                    f"INFO: task_atom_backfill minted {task_backfill_n} quote-level task atom(s) "
                    f"from notes/email scope"
                )
        except Exception as exc:
            warnings.append(f"WARNING: task_atom_backfill failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=task_backfill_n)

    # Single-facility deals: tasks often parse with device:* keys only.
    # After dedup the roster is final — link orphans to the one confirmed site.
    with telemetry.stage("site_task_anchor", input_count=len(atoms)) as stage:
        linked_site = 0
        try:
            from app.core.site_task_anchor import anchor_orphan_atoms_to_confirmed_site

            atoms, linked_site = anchor_orphan_atoms_to_confirmed_site(atoms)
            if linked_site:
                warnings.append(
                    f"INFO: site_task_anchor linked {linked_site} task/note atom(s) "
                    f"to the sole confirmed job site"
                )
        except Exception as exc:
            warnings.append(f"WARNING: site_task_anchor failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=linked_site)

    # Parent vs child task tiers — quote-level work units vs runbook steps.
    with telemetry.stage("task_tier_classification", input_count=len(atoms)) as stage:
        tier_stamped = 0
        try:
            from app.core.task_tier_classifier import classify_task_tiers

            atoms, tier_stamped = classify_task_tiers(atoms)
            if tier_stamped:
                warnings.append(
                    f"INFO: task_tier_classification stamped {tier_stamped} task atom(s) "
                    f"with parent/child quote-line tiers"
                )
            from app.core.task_tier_classifier import fold_task_mentions

            folded = fold_task_mentions(atoms)
            if folded:
                warnings.append(
                    f"INFO: task_tier_classification folded {folded} repeated mention(s) of "
                    f"a unit of work into its fullest statement"
                )
        except Exception as exc:
            warnings.append(f"WARNING: task_tier_classification failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=tier_stamped)

    # Task admission (learning-loop gate): a task a PM removed as "not this
    # job" taught admission → drop on its own sentence; the next compile of the
    # same documents must not propose it again. Store-only, guess-free,
    # lossless (dropped atoms go to the suppression ledger).
    with telemetry.stage("task_admission", input_count=len(atoms)) as stage:
        try:
            from app.core.task_admission import drop_taught_out_tasks

            before_admission = list(atoms)
            atoms, dropped_tasks = drop_taught_out_tasks(atoms, project_id=resolved_project_id)
            if dropped_tasks:
                merge_suppressed(
                    suppressed_atoms,
                    capture_suppressed(
                        before_admission, atoms,
                        stage="task_admission",
                        reason="a PM taught this line is not this job (admission → drop)",
                    ),
                )
                warnings.append(f"INFO: task_admission dropped {len(dropped_tasks)} task(s) a PM taught out")
        except Exception as exc:
            warnings.append(f"WARNING: task_admission failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=len(atoms))

    # Quote-context head seam — classifies the commercial delivery model
    # (configuration-only vs install/buildout vs survey/design) with a promoted
    # neural head when available. Cold start logs a trainable row and uses a
    # conservative source-grounded fallback, so Deal Kit does not need to encode
    # this as frontend-only heuristics.
    with telemetry.stage("quote_context_head", input_count=len(atoms)) as stage:
        quote_context_n = 0
        try:
            from app.core.quote_context_head import annotate_quote_context

            atoms, quote_context_n = annotate_quote_context(
                atoms, project_id=resolved_project_id
            )
            if quote_context_n:
                warnings.append(
                    f"INFO: quote_context_head annotated {quote_context_n} quote-level task atom(s)"
                )
        except Exception as exc:
            warnings.append(f"WARNING: quote_context_head failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=quote_context_n)

    # A schedule row named "Milestone ..." is a milestone and a row that is
    # only a PO number is the deal's PO reference, whatever the row typers
    # said -- before the quote-line head, which would drop them as tasks.
    try:
        from app.core.atom_type_sanity import retype_schedule_reference_rows

        _sched_n = retype_schedule_reference_rows(atoms)
        if _sched_n:
            warnings.append(f"INFO: retyped {_sched_n} milestone / PO-reference table row(s)")
    except Exception as exc:
        warnings.append(f"WARNING: schedule row typing failed: {type(exc).__name__}: {exc}")

    with telemetry.stage("quote_line_head", input_count=len(atoms)) as stage:
        quote_line_n = 0
        try:
            from app.core.quote_line_head import consolidate_quote_line_tasks

            _before_quote_line = list(atoms)
            atoms, quote_line_n = consolidate_quote_line_tasks(
                atoms, project_id=resolved_project_id
            )
            # A PMO/admin step ("Complete billing tasks", "Develop schedule
            # for installation activities") is not a quote line, and the
            # head drops it -- which removed the line from the deal without a
            # record (010003). Every removal goes to the ledger.
            merge_suppressed(
                suppressed_atoms,
                capture_suppressed(
                    _before_quote_line, atoms, stage="quote_line_head",
                    reason="PMO/admin task or a line folded into a quote-line umbrella",
                ),
            )
            if quote_line_n:
                warnings.append(
                    f"INFO: quote_line_head consolidated {quote_line_n} quote-level task atom(s)"
                )
        except Exception as exc:
            warnings.append(f"WARNING: quote_line_head failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=quote_line_n)

    # Learned hours per unit of work, taught from finished Deal Kits
    # (relation task_hours). Store-only, guess-free: a task nobody taught
    # anything like keeps no estimate.
    with telemetry.stage("task_hours", input_count=len(atoms)) as stage:
        task_hours_n = 0
        try:
            from app.core.task_hours import estimate_task_hours

            task_hours_n = estimate_task_hours(atoms)
            if task_hours_n:
                warnings.append(f"INFO: task_hours stamped learned hours on {task_hours_n} task atom(s)")
        except Exception as exc:
            warnings.append(f"WARNING: task_hours failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=task_hours_n)

    # The commercial shape finished kits gave this kind of request (billing
    # type, PM/PC hours, travel days; relation commercial_terms). Store-only.
    with telemetry.stage("commercial_terms", input_count=len(atoms)) as stage:
        commercial_n = 0
        try:
            from app.core.commercial_terms import stamp_commercial_terms

            commercial_n = stamp_commercial_terms(atoms)
            if commercial_n:
                warnings.append(f"INFO: commercial_terms stamped a learned kit shape on {commercial_n} task atom(s)")
        except Exception as exc:
            warnings.append(f"WARNING: commercial_terms failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=commercial_n)

    with telemetry.stage("hardware_evidence_backfill", input_count=len(atoms)) as stage:
        hardware_bom_n = 0
        try:
            from app.core.hardware_evidence_backfill import backfill_hardware_bom_lines

            atoms, hardware_bom_n = backfill_hardware_bom_lines(
                atoms, project_id=resolved_project_id
            )
            if hardware_bom_n:
                warnings.append(
                    f"INFO: hardware_evidence_backfill minted {hardware_bom_n} bom_line atom(s)"
                )
        except Exception as exc:
            warnings.append(f"WARNING: hardware_evidence_backfill failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=hardware_bom_n)

    with telemetry.stage("site_facility_head", input_count=len(atoms)) as stage:
        facility_n = 0
        try:
            from app.core.site_facility_head import annotate_site_facility_labels

            atoms, facility_n = annotate_site_facility_labels(
                atoms, project_id=resolved_project_id
            )
            if facility_n:
                warnings.append(
                    f"INFO: site_facility_head labeled {facility_n} physical_site atom(s)"
                )
        except Exception as exc:
            warnings.append(f"WARNING: site_facility_head failed: {type(exc).__name__}: {exc}")
        # Any site this head minted from a signature page is a party address.
        try:
            from app.core.party_address_veto import veto_party_page_sites
            _party_late = veto_party_page_sites(atoms)
            if _party_late:
                warnings.append(f"INFO: {_party_late} signature-page site(s) re-vetoed after site_facility_head")
        except Exception as exc:
            warnings.append(f"WARNING: late party_address_veto failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=facility_n)

    # Noise suppression (learning-loop gate): divert reference/template atoms
    # (master rate-card / materials-catalog rows, rate-label-as-person) out of
    # scope using the store-only ``atom_noise_admission`` gate. Guess-free and
    # opt-in (SOWSMITH_NOISE_SUPPRESSION); a no-op byte-for-byte when disabled,
    # no store is wired, or the store has nothing confident to say. Runs after
    # dedup so the gate sees canonical atoms, and before packetization /
    # scorecards so suppressed noise never inflates totals or stakeholder counts.
    with telemetry.stage("noise_suppression", input_count=len(atoms)) as stage:
        try:
            from app.core.noise_suppression import suppress_noise_atoms

            before_noise_atoms = list(atoms)
            atoms, dropped_noise = suppress_noise_atoms(
                atoms, project_id=resolved_project_id
            )
            if dropped_noise:
                merge_suppressed(
                    suppressed_atoms,
                    capture_suppressed(
                        before_noise_atoms, atoms,
                        stage="noise_suppression",
                        reason="reference/template content (rate-card / catalog row or rate-label-as-person), not deal scope",
                    ),
                )
                warnings.append(
                    f"INFO: noise_suppression diverted {len(dropped_noise)} "
                    f"reference/template atom(s) out of scope"
                )
        except Exception as exc:
            warnings.append(f"WARNING: noise_suppression failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=len(atoms))

    # Substance gate: drop context-free atom fragments (bare-name stakeholders,
    # backchannel filler) AFTER every backfill/dedup head has finished — the
    # typed classifier, semantic dedup, and task/quote heads can still mint or
    # re-type atoms late in the pipeline; running here ensures sign-off name
    # fragments ("Tom Amble.") and salutations re-classified as stakeholder are
    # caught. Lossless: dropped atoms go to the suppression ledger.
    # A sheet that arrived as both DWG and PDF is ONE sheet read twice. The
    # export's OCR rows ("JAN", "ADA RR") are a worse reading of a drawing we
    # parsed properly, so they are demoted to evidence rather than deleted --
    # and left alone entirely when the drawing never converted, because then
    # the export is the only account of the sheet there is.
    # A PDF export of a sheet whose DRAWING we parsed is the same sheet read
    # badly, by the worse of two available methods. Demoting its rows was the
    # first attempt and it was not enough: a demoted row still takes a line in
    # the labeling pane, and live 010180 had twenty-six of them -- "JAN",
    # "ADA RR", "P: 203.246.1900", "NOTHING BEATS 72 YEARS OF STABILITY".
    # None of that reaches a deal kit or a SOW.
    #
    # So they are withheld: out of the accepted set, into the suppression
    # ledger with the reason, which keeps them auditable and keeps them as
    # training data without asking a PM to read them. One atom takes their
    # place and asks the PM for anything the drawing did not give.
    with telemetry.stage("drawing_pairs", input_count=len(atoms)) as _dp:
        _withheld = 0
        try:
            from app.core.drawing_pairs import (
                export_rows_to_withhold,
                withhold_export_rows,
            )

            by_sheet = export_rows_to_withhold(atoms)
            if by_sheet:
                before_pairs = list(atoms)
                drop = {id(a) for rows in by_sheet.values() for a in rows}
                atoms = [a for a in atoms if id(a) not in drop]
                _withheld = len(drop)
                for sheet, rows in by_sheet.items():
                    ask = withhold_export_rows(before_pairs, sheet, rows)
                    template = rows[0]
                    atoms.append(_ask_the_pm_atom(template, ask))
                merge_suppressed(
                    suppressed_atoms,
                    capture_suppressed(
                        before_pairs, atoms,
                        stage="drawing_pairs",
                        reason=("OCR of a PDF export of a sheet whose drawing parsed — "
                                "the same sheet read by the worse of two methods"),
                    ),
                )
                warnings.append(
                    f"INFO: drawing_pairs withheld {_withheld} OCR row(s) from PDF "
                    f"export(s) of {len(by_sheet)} sheet(s) parsed from the drawing"
                )
        except Exception as exc:
            warnings.append(f"WARNING: drawing_pairs failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(_dp, output_count=_withheld)

    # Where the deal STANDS, as a handful of lines instead of forty atoms.
    #
    # A mailbox carries two kinds of sentence and only one belongs in a deal
    # kit. "Two Cat6A drops per workstation" is content. "Can you provide
    # availability for the walkthrough" is the deal moving -- and on live
    # 010180 that traffic was 43 of 252 non-rejected atoms, competing for a
    # PM's attention with the two numbers that decide the job.
    #
    # The consolidation also carries a gate: a price its own authors call
    # budgetary, with no completed survey on the record, must not become a
    # firm SOW. 010180's $110,108 came from a solutions architect listening to
    # a call recording; nobody from PurTera had been on site.
    with telemetry.stage("deal_state", input_count=len(atoms)) as _ds:
        _state_lines = 0
        try:
            from app.core.deal_state import read_deal_state

            state = read_deal_state(atoms)
            if state.lines:
                fallback = atoms[0] if atoms else None
                for line in state.lines:
                    # Pin the line to the artifact its evidence came from; the
                    # first atom of the deal is only a last resort (000132's
                    # survey line landed on an unrelated note).
                    # A sentence outranks a sheet row as the anchor: a row's
                    # locator names a cell the line was never written in
                    # (010003: survey lines pinned to the Deal Kit SELL RATES).
                    from app.core.deal_state import _from_a_sheet as _ds_sheet

                    _ev = list(getattr(line, "evidence_atoms", None) or [])
                    template = next((a for a in _ev if not _ds_sheet(a)), None) or next(iter(_ev), None) or fallback
                    if template is None:
                        break
                    atoms.append(_deal_state_atom(template, line))
                _state_lines = len(state.lines)
                blocks = state.get("blocks")
                if blocks is not None:
                    warnings.append(
                        "INFO: deal_state — price basis is "
                        f"{state.get('price_basis').value}; {blocks.value}"
                    )
        except Exception as exc:
            warnings.append(f"WARNING: deal_state failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(_ds, output_count=_state_lines)

    with telemetry.stage("substance_gate", input_count=len(atoms)) as stage:
        gate_dropped = 0
        _gate_notes: list[str] = []
        try:
            from app.core.atom_substance_gate import apply_substance_gate

            before_gate = list(atoms)
            atoms, dropped_gate = apply_substance_gate(atoms)
            if dropped_gate:
                merge_suppressed(
                    suppressed_atoms,
                    capture_suppressed(
                        before_gate,
                        atoms,
                        stage="substance_gate",
                        reason="context-free fragment (bare-name stakeholder or backchannel filler) — not actionable to any head",
                    ),
                )
                gate_dropped = len(dropped_gate)
                warnings.append(
                    f"INFO: substance_gate diverted {gate_dropped} context-free "
                    f"fragment(s) (bare-name stakeholders / backchannel filler)"
                )
                _gate_notes = _dropped_atom_notes("substance_gate", dropped_gate)
        except Exception as exc:
            warnings.append(f"WARNING: substance_gate failed: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=gate_dropped, warnings=_gate_notes)

    # v53 SMART CONFIDENCE — recalibrate every atom from hardcoded
    # provenance defaults (0.82/0.85) to content-aware scoring:
    # semantic-key anchored + value completeness + cross-doc
    # corroboration + source authority tier + receipts verified
    # + text-length quality. PMs get a confidence score that
    # actually correlates with truth.
    with telemetry.stage("confidence_recalibration", input_count=len(atoms)) as stage:
        recal_count = 0
        _stage_notes: list[str] = []
        try:
            from app.core.confidence_recalibration import recalibrate_confidence
            from app.core.authority import classify_artifact_authority
            # Build artifact_id → tier from filenames
            _artifact_tier: dict[str, str] = {}
            for _a in atoms:
                _aid = getattr(_a, "artifact_id", None)
                if not _aid or _aid in _artifact_tier:
                    continue
                _refs = getattr(_a, "source_refs", None) or []
                _fname = ""
                if _refs:
                    _fname = getattr(_refs[0], "filename", "") or ""
                _artifact_tier[_aid] = (
                    classify_artifact_authority(_fname)
                    if _fname else "supporting_evidence"
                )
            recal_count = recalibrate_confidence(
                atoms, artifact_authority=_artifact_tier, edges=[],
                abstain_threshold=abstain_threshold,
            )
        except Exception as exc:
            _stage_notes.append(f"WARNING: confidence_recalibration failed: {type(exc).__name__}: {exc}")
            warnings.append(f"WARNING: confidence_recalibration failed: {type(exc).__name__}: {exc}")
        # Review is a signal only when scarce: a verbatim atom with every
        # receipt verified and high confidence has nothing left to review.
        # Independent of recalibration: live 010300 rounds 21-23 the
        # recalibrator raised (a None confidence on a vision atom) and this
        # step, nested inside its try, silently never ran -- 137 of 168
        # atoms queued with nothing to doubt.
        try:
            from app.core.confidence_recalibration import (
                LAST_ACCEPT_STATS,
                accept_verified_high_confidence,
            )
            _accepted = accept_verified_high_confidence(atoms)
            _stage_notes.append(
                "INFO: review-queue acceptance "
                + ", ".join(f"{k}={v}" for k, v in LAST_ACCEPT_STATS.items())
            )
            if _accepted:
                warnings.append(f"INFO: accepted {_accepted} verified high-confidence atom(s) out of the review queue")
        except Exception as exc:
            _stage_notes.append(f"WARNING: accept_verified_high_confidence failed: {type(exc).__name__}: {exc}")
            warnings.append(f"WARNING: accept_verified_high_confidence failed: {type(exc).__name__}: {exc}")
        if recal_count:
            warnings.append(f"INFO: recalibrated confidence on {recal_count} atoms")
        telemetry.end_stage(stage, output_count=recal_count, warnings=_stage_notes)

    with telemetry.stage("entity_resolution", input_count=len(atoms)) as stage:
        entities = resolve_aliases(
            extract_entity_records(resolved_project_id, atoms, pack=resolved_domain_pack)
        )
        # Cross-mention alias fusion: collapse `site:atl_hq +
        # site:atlanta_headquarters + site:innovation_tower` (three
        # surface names for one physical place) into a single
        # EntityRecord whose `aliases` field carries all three keys.
        # Detected via co-mention patterns in atom raw_text (copular
        # "is the", em-dash, slash, parenthetical aliasing, ...).
        site_alias_groups = collect_site_alias_groups(atoms)
        # D3: collapse multiple surface forms of the same person
        # (``stakeholder:watkins`` + ``stakeholder:r_watkins`` →
        # ``stakeholder:renee_watkins``) using the same fusion
        # mechanism. Key-shape based, so no false positives across
        # documents.
        stakeholder_alias_groups = collect_stakeholder_alias_groups(atoms)
        entities = fuse_alias_groups(
            entities, site_alias_groups + stakeholder_alias_groups
        )
        backfill_n = 0
        try:
            from app.core.site_atom_backfill import backfill_physical_sites_from_entities

            atoms, backfill_n = backfill_physical_sites_from_entities(
                atoms, entities, project_id=resolved_project_id
            )
            if backfill_n:
                warnings.append(
                    f"INFO: site_atom_backfill minted {backfill_n} physical_site atom(s) "
                    f"from site entities (roster was empty after dedup)"
                )
                # A site minted here was not there when site_geo_fallback ran,
                # so the place the document names for it was never filled in.
                # Live 000061 (compile 9a6aacfc): "highland park warehouse
                # office", minted from the call, reached the brief with no
                # city or state while the transcript said "Highland Park,
                # Michigan" in the next breath. Same pass, same rules, on the
                # sites that exist now -- before the dedup below, so two
                # mentions of one place can be seen to be one place.
                try:
                    from app.core.site_geo_fallback import enrich_site_geo as _enrich_late

                    late_geo = _enrich_late(atoms)
                    if late_geo:
                        warnings.append(
                            f"INFO: site_geo_fallback enriched {late_geo} late-minted "
                            f"physical_site atom(s) with city/state/ZIP recovered from the document"
                        )
                except Exception as exc:
                    warnings.append(f"WARNING: late site_geo enrichment failed: {type(exc).__name__}: {exc}")
        except Exception as exc:
            warnings.append(f"WARNING: site_atom_backfill failed: {type(exc).__name__}: {exc}")
        try:
            # Backfill runs after semantic_dedup; minted entity_backfill sites can
            # duplicate the roster atom for the same address (MBrany-class).
            from app.core.semantic_dedup import _dedupe_physical_site_atoms

            phys_before = sum(
                1
                for a in atoms
                if getattr(getattr(a, "atom_type", None), "value", getattr(a, "atom_type", None))
                == "physical_site"
            )
            atoms = _dedupe_physical_site_atoms(atoms)
            phys_after = sum(
                1
                for a in atoms
                if getattr(getattr(a, "atom_type", None), "value", getattr(a, "atom_type", None))
                == "physical_site"
            )
            if phys_after < phys_before:
                warnings.append(
                    f"INFO: post_backfill physical_site dedup collapsed "
                    f"{phys_before} -> {phys_after} site atom(s)"
                )
            before_post_vendor = list(atoms)
            from app.core.site_geo_fallback import suppress_vendor_sites as _suppress_vendor_sites

            atoms, post_vendor_dropped = _suppress_vendor_sites(
                atoms, project_id=resolved_project_id
            )
            if post_vendor_dropped:
                merge_suppressed(
                    suppressed_atoms,
                    capture_suppressed(
                        before_post_vendor,
                        atoms,
                        stage="entity_resolution",
                        reason="vendor / selling-party letterhead address, not a job site",
                    ),
                )
                warnings.append(
                    f"INFO: post_backfill suppressed {post_vendor_dropped} "
                    f"vendor/letterhead address(es) misread as job sites"
                )
            try:
                from app.core.site_facility_head import annotate_site_facility_labels

                atoms, post_facility_n = annotate_site_facility_labels(
                    atoms, project_id=resolved_project_id
                )
                if post_facility_n:
                    warnings.append(
                        f"INFO: site_facility_head labeled {post_facility_n} post-backfill physical_site atom(s)"
                    )
            except Exception as exc:
                warnings.append(
                    f"WARNING: post_backfill site_facility_head failed: {type(exc).__name__}: {exc}"
                )
        except Exception as exc:
            warnings.append(
                f"WARNING: post_backfill physical_site dedup failed: {type(exc).__name__}: {exc}"
            )
        telemetry.end_stage(stage, output_count=len(entities))

    # v48 FIX 6: Cross-doc conflict detection.
    # Build artifact_id → authority_tier map from filenames, then scan
    # atoms for the same entity_key appearing with contradictory numeric
    # values across docs with different tiers.
    cross_doc_conflicts: list[dict[str, Any]] = []
    try:
        from app.core.authority import classify_artifact_authority
        from collections import defaultdict
        # Map artifact_id → filename → tier
        artifact_tier: dict[str, str] = {}
        for atom in atoms:
            aid = getattr(atom, "artifact_id", None)
            if not aid or aid in artifact_tier:
                continue
            refs = getattr(atom, "source_refs", None) or []
            fname = ""
            if refs:
                fname = getattr(refs[0], "filename", "") or ""
            artifact_tier[aid] = classify_artifact_authority(fname) if fname else "supporting_evidence"
        # Group (entity_key, first_int_in_atom) → list of (artifact_id, tier, raw_text)
        groups: dict[tuple, list] = defaultdict(list)
        for atom in atoms:
            ekeys = getattr(atom, "entity_keys", None) or []
            if not ekeys:
                continue
            rt = getattr(atom, "raw_text", "") or ""
            nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', rt)
            if not nums:
                continue
            aid = getattr(atom, "artifact_id", "unknown")
            tier = artifact_tier.get(aid, "supporting_evidence")
            for ekey in ekeys:
                groups[(ekey, nums[0])].append({
                    "value": nums[0],
                    "artifact_id": aid,
                    "authority_tier": tier,
                    "raw_text": rt[:200],
                })
        # Look for entity keys with multiple distinct values across multiple tiers
        by_entity: dict[str, list] = defaultdict(list)
        for (ekey, val), entries in groups.items():
            for entry in entries:
                by_entity[ekey].append({**entry, "value": val})
        for ekey, entries in by_entity.items():
            tiers = {e["authority_tier"] for e in entries}
            values = {e["value"] for e in entries}
            if len(values) > 1 and len(tiers) > 1:
                if "contractual_final" in tiers:
                    severity = "high"
                elif "approved_scope" in tiers:
                    severity = "medium"
                else:
                    severity = "low"
                cross_doc_conflicts.append({
                    "entity_key": ekey,
                    "values": entries,
                    "severity": severity,
                })
        if cross_doc_conflicts:
            warnings.append(f"INFO: detected {len(cross_doc_conflicts)} cross-doc conflicts")
    except Exception as exc:
        warnings.append(f"WARNING: cross_doc_conflicts failed: {type(exc).__name__}: {exc}")

    # v48 FIX 8: BOM arithmetic cross-check.
    bom_arithmetic_check: dict[str, Any] | None = None
    try:
        def _extract_dollars(text: str) -> float | None:
            m = re.search(r'\$\s*([\d,]+(?:\.\d+)?)', text)
            if m:
                return float(m.group(1).replace(",", ""))
            return None
        line_items: list[float] = []
        stated_total: float | None = None
        for atom in atoms:
            atype = str(getattr(atom, "atom_type", "") or "")
            if hasattr(atom.atom_type, "value"):
                atype = atom.atom_type.value
            rt = getattr(atom, "raw_text", "") or ""
            rt_lower = rt.lower()
            if "vendor_line_item" in atype or "bom_line" in atype:
                v = _extract_dollars(rt)
                if v and v > 0:
                    line_items.append(v)
            elif any(kw in rt_lower for kw in ("grand total", "total price", "contract total", "project total")):
                v = _extract_dollars(rt)
                if v and v > 0:
                    stated_total = v
        if len(line_items) >= 2 and stated_total is not None:
            line_sum = sum(line_items)
            discrepancy = abs(line_sum - stated_total)
            pct = (discrepancy / stated_total * 100) if stated_total else 0
            if pct >= 0.5:
                bom_arithmetic_check = {
                    "line_item_sum": round(line_sum, 2),
                    "stated_total": round(stated_total, 2),
                    "discrepancy": round(discrepancy, 2),
                    "discrepancy_pct": round(pct, 2),
                    "severity": "high" if pct > 5 else "medium" if pct > 1 else "low",
                }
                warnings.append(
                    f"INFO: BOM arithmetic discrepancy: line-item sum ${line_sum:,.2f} "
                    f"vs stated total ${stated_total:,.2f} ({pct:.1f}%)"
                )
    except Exception as exc:
        warnings.append(f"WARNING: bom_arithmetic_check failed: {type(exc).__name__}: {exc}")

    # A PM answering an open question in the brief is usually stating deal truth
    # that exists in no document. Admit it as evidence HERE — before graph_build
    # and packetize — so it behaves like every other atom: it gets edges, lands
    # in packets, can settle a cross-document conflict at pm_confirmed authority,
    # and reaches the SOW. Anything later would only decorate the envelope.
    # Gated + best-effort: no ledger, or no blob, is a normal no-op.
    with telemetry.stage("pm_answers", input_count=len(atoms)) as stage:
        pm_atoms = []
        try:
            from app.core.pm_answer_blob import load_pm_answer_atoms

            existing_ids = {a.id for a in atoms}
            pm_atoms = [
                a
                for a in load_pm_answer_atoms(
                    project_id=resolved_project_id, deal_id=resolved_project_id
                )
                if a.id not in existing_ids
            ]
            atoms = atoms + pm_atoms
        except Exception as exc:  # never let the ledger break a compile
            warnings.append(f"WARNING: pm_answer atoms skipped: {type(exc).__name__}: {exc}")
        telemetry.end_stage(stage, output_count=len(pm_atoms))

    with telemetry.stage("declared_scope", input_count=len(atoms)) as stage:
        # Declared-vs-found reconciliation (Marion County lesson): a customer
        # sentence like "SOW's for each of the ten locations" must surface as
        # an open question when the parse found fewer sites or none of the
        # referenced documents. Additive only -- never breaks a compile.
        ds_atoms: list = []
        try:
            from app.core.declared_scope import declared_scope_questions

            ds_atoms = declared_scope_questions(
                project_id=resolved_project_id, atoms=atoms
            )
            atoms = atoms + ds_atoms
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"WARNING: declared_scope skipped: {type(exc).__name__}: {exc}"
            )
        telemetry.end_stage(stage, output_count=len(ds_atoms))

    with telemetry.stage("graph_build", input_count=len(atoms)) as stage:
        edges = build_edges(project_id=resolved_project_id, atoms=atoms, entities=entities)
        telemetry.end_stage(stage, output_count=len(edges))

    with telemetry.stage("packetize", input_count=len(edges)) as stage:
        packets = build_packets(
            project_id=resolved_project_id,
            atoms=atoms,
            entities=entities,
            edges=edges,
            attach_metadata=False,
        )
        telemetry.end_stage(stage, output_count=len(packets))

    with telemetry.stage("packet_certificates", input_count=len(packets)) as stage:
        atom_by_id = {atom.id: atom for atom in atoms}
        edge_by_id = {edge.id: edge for edge in edges}
        for packet in packets:
            packet.certificate = build_packet_certificate(packet, atom_by_id, edge_by_id=edge_by_id)
            packet_atoms = [
                atom_by_id[atom_id]
                for atom_id in (packet.supporting_atom_ids + packet.contradicting_atom_ids)
                if atom_id in atom_by_id
            ]
            packet.risk = score_packet_risk(packet, packet_atoms, edges)
        telemetry.end_stage(stage, output_count=len(packets))

    # The held chatter atoms come back now: after every head has run, before
    # coverage, so their lines count as claimed by an atom.
    if held_chatter:
        try:
            for _atom in held_chatter:
                if getattr(_atom, "source_refs", None) and not getattr(_atom, "receipts", None):
                    _atom.receipts = replay_atom_receipts(_atom, artifact_paths)
        except Exception as exc:  # never fail a compile over a chatter receipt
            warnings.append(f"WARNING: chatter receipts failed: {type(exc).__name__}: {exc}")
        _seen_ids = {a.id for a in atoms}
        _back: list = []
        for _atom in held_chatter:
            if _atom.id not in _seen_ids:
                _seen_ids.add(_atom.id)
                _back.append(_atom)
        atoms = atoms + _back

    # The cross-document copies come back the same way: after every head, so
    # nothing counted, priced or packetized them; before coverage, so each
    # document's line counts as read by its own copy.
    #
    # Not every fold runs through _hold_copies (type-specific passes, site
    # merges, list-split pieces, a quoted note line), but every fold that
    # credits a line to another document leaves that document's ref on the
    # survivor. One sweep over the final atoms -- after the held copies point
    # at a canonical atom still in the result -- gives each such document its
    # own copy, so no line of a document is shown only as another's atom.
    try:
        from app.core.cross_doc_copies import (
            drop_unheld_copies,
            ensure_own_copies,
            resolve_canonical,
            source_lines_reader,
        )

        # A copy is a line its document HOLDS: never one minted only because
        # the document names the same person, or names them in its header.
        _doc_lines = source_lines_reader(artifact_paths)
        if held_copies:
            resolve_canonical(held_copies, atoms)
            _kept_copies, _refused = drop_unheld_copies(held_copies, atoms, _doc_lines)
            if _refused:
                held_copies[:] = _kept_copies
                merge_suppressed(
                    suppressed_atoms,
                    capture_suppressed(
                        _refused, [], stage="own_copy_gate",
                        reason="folded onto another document's atom; this document's text does not hold the line",
                    ),
                )
                warnings.append(f"INFO: own_copy_gate refused {len(_refused)} copy(ies) their document does not hold")
        _swept = ensure_own_copies(atoms, held_copies, doc_lines=_doc_lines, dropped=suppressed_atoms)
        if _swept:
            # A folded line its own document got back as its copy is no
            # longer suppressed.
            _back_ids = {id(a) for a in _swept}
            suppressed_atoms[:] = [a for a in suppressed_atoms if id(a) not in _back_ids]
        if _swept:
            held_copies.extend(_swept)
            warnings.append(f"INFO: own_copy_sweep gave {len(_swept)} document line(s) their own copy")
    except Exception as exc:  # never fail a compile over a copy
        warnings.append(f"WARNING: own_copy_sweep failed: {type(exc).__name__}: {exc}")
    if held_copies:
        try:
            for _atom in held_copies:
                if getattr(_atom, "source_refs", None) and not getattr(_atom, "receipts", None):
                    _atom.receipts = replay_atom_receipts(_atom, artifact_paths)
        except Exception as exc:  # never fail a compile over a copy's receipt
            warnings.append(f"WARNING: copy receipts failed: {type(exc).__name__}: {exc}")
        try:
            from app.core.cross_doc_copies import resolve_canonical

            resolve_canonical(held_copies, atoms)
        except Exception as exc:
            warnings.append(f"WARNING: copy canonical resolution failed: {type(exc).__name__}: {exc}")
        _seen_ids = {a.id for a in atoms}
        _back_copies: list = []
        for _atom in held_copies:
            if _atom.id not in _seen_ids:
                _seen_ids.add(_atom.id)
                _back_copies.append(_atom)
        atoms = atoms + _back_copies
        warnings.append(
            f"INFO: cross_doc_copies kept {len(_back_copies)} later-document copy(ies) "
            f"of lines an earlier document owns"
        )

    # Relationship talk is typed as what it is now that every stage has run:
    # small_talk, the labeler's reject type, not deal_metadata (010087).
    try:
        from app.core.deal_chatter import retype_small_talk

        _st = retype_small_talk(atoms)
        if _st:
            warnings.append(f"INFO: {_st} small-talk line(s) typed small_talk")
    except Exception as exc:
        warnings.append(f"WARNING: small_talk typing failed: {type(exc).__name__}: {exc}")

    # What did we NOT read? Diff every text artifact against its own atoms, so
    # a paragraph that produced nothing is visible instead of silent.
    try:
        from app.core.text_coverage import build_text_coverage

        _coverage = build_text_coverage(artifact_paths, atoms, suppressed_atoms)
        _unread = sum(int(r.get("unread_count") or 0) for r in _coverage)
        if _unread:
            warnings.append(f"INFO: text_coverage: {_unread} source line(s) produced no atom")
    except Exception as exc:  # coverage reporting can never fail a compile
        _coverage = []
        warnings.append(f"WARNING: text_coverage failed: {type(exc).__name__}: {exc}")

    result = CompileResult(
        project_id=resolved_project_id,
        atoms=atoms,
        suppressed_atoms=sorted(suppressed_atoms, key=lambda x: x.id),
        text_coverage=_coverage,
        entities=entities,
        edges=edges,
        packets=packets,
        warnings=sorted(warnings),
        schema_version=SCHEMA_VERSION,
        compiler_version=COMPILER_VERSION,
        compile_id=manifest.compile_id,
        manifest=manifest,
        project_dir=str(project_dir),
        ranked_atoms=[],
        entity_edges=[],
        candidate_summary=summarize_candidate_outcomes(
            candidates=candidates,
            accepted_atoms=adjudication.accepted_atoms if candidates else [],
            rejected_candidates=rejected_candidates,
        ),
    )
    result.atoms = sorted(result.atoms, key=lambda x: x.id)
    result.entities = sorted(result.entities, key=lambda x: x.id)
    result.edges = sorted(result.edges, key=lambda x: x.id)
    result.packets = sorted(
        result.packets,
        key=lambda p: packet_pm_sort_key(p) if p.risk is not None else (50, 50, 0.0, p.anchor_key, p.id),
    )
    if calibrator_path is not None:
        with telemetry.stage("confidence_calibration", input_count=len(result.packets)) as stage:
            try:
                result = apply_calibration(result, calibrator_path, abstain_threshold=abstain_threshold)
                telemetry.end_stage(stage, output_count=len(result.packets))
            except Exception as exc:
                warning = f"WARNING: Failed to apply calibrator {calibrator_path}: {exc}"
                result.warnings = sorted(set(result.warnings + [warning]))
                telemetry.end_stage(stage, output_count=len(result.packets), warnings=[warning], errors=[])
    # Finalize the manifest (including output_signature) BEFORE quality gates so
    # the validator doesn't fire a spurious "missing output_signature" warning
    # on every compile.  output_signature is content-addressed over the result
    # pre-validation; validation messages are excluded from the signature so a
    # warning later doesn't recursively change the signature.
    # Final receipt sweep. The backfill above runs mid-pipeline, and several
    # stages after it still MINT atoms -- quote_line_head consolidating a
    # bom_line is the one that surfaced this. Such an atom carries source_refs
    # it never got receipts for, and the validator treats "source_refs but no
    # receipts while source files are available" as a hard ERROR, so a single
    # late-minted atom fails the whole compile. Observed on a real deal: one
    # bom_line out of ~2,000 atoms, and nothing else wrong with the run.
    #
    # Idempotent and free when there is nothing to do -- it only touches atoms
    # that have source_refs and no receipts -- so running it once more here
    # costs a no-op pass and closes the window for every future late stage
    # rather than for this one caller.
    late_backfilled = 0
    try:
        for atom in result.atoms:
            if getattr(atom, "source_refs", None) and not getattr(atom, "receipts", None):
                atom.receipts = replay_atom_receipts(atom, artifact_paths)
                late_backfilled += 1
    except Exception as exc:  # never fail a compile inside the safety net
        warnings.append(
            f"WARNING: final receipt sweep failed: {type(exc).__name__}: {exc}"
        )
    if late_backfilled:
        warnings.append(
            f"INFO: final receipt sweep attached receipts to {late_backfilled} "
            "atom(s) minted after receipt_backfill"
        )

    output_signature = compute_output_signature(result)
    result.manifest = finalize_manifest(manifest, output_signature)
    with telemetry.stage("quality_gates", input_count=len(result.packets)) as stage:
        validation_messages = validate_compile_result(result, source_files_available=True)
        hard_errors = [m for m in validation_messages if m.startswith("ERROR:")]
        validation_warnings = [m for m in validation_messages if m.startswith("WARNING:")]
        # Every atom any stage suppressed, one line each, so a live compile's
        # losses are readable from its trace (live 010300 round 26: two
        # clauses vanished with no stage owning the loss).
        try:
            from app.core.suppression_ledger import DROP_NOTES as _DROP_NOTES
            _ledger_notes = list(_DROP_NOTES)
        except Exception:
            _ledger_notes = []
        telemetry.end_stage(
            stage,
            output_count=len(validation_messages),
            warnings=validation_warnings + _ledger_notes,
            errors=hard_errors,
        )
    if allow_unverified_receipts:
        receipt_hard_errors = [m for m in hard_errors if "receipt" in m.lower()]
        if receipt_hard_errors:
            validation_warnings.extend(
                [f"WARNING: downgraded receipt validation under --allow-unverified-receipts: {m}" for m in receipt_hard_errors]
            )
        hard_errors = [m for m in hard_errors if "receipt" not in m.lower()]
    result.warnings = sorted(set(result.warnings + validation_warnings))

    if persistence_hook is not None:
        with telemetry.stage("persistence", input_count=1) as stage:
            persistence_hook(result)
            telemetry.end_stage(stage, output_count=1)

    # The rule-decision log goes to blob before the container is recycled. It
    # is the only record of the decisions where a rule did NOT fire, and those
    # are the negatives nothing else can supply: an atom exists only when the
    # rule fired, so labelling alone yields positives and nothing else
    # (measured: 65 rows, 65 positives, across 010180 and 010288). Gated on
    # SOWSMITH_RULE_LOG + SOWSMITH_FEEDBACK_BLOB; best-effort, and a failure
    # here must never fail a compile that has already succeeded.
    try:
        _rule_log = os.environ.get("SOWSMITH_RULE_LOG")
        if _rule_log:
            from app.core import feedback_blob as _fb

            _n = _fb.upload_rule_decisions(str(result.compile_id or "compile"), _rule_log)
            if _n:
                warnings.append(f"INFO: mirrored {_n} rule decision(s) to blob for threshold training")
    except Exception:  # pragma: no cover - never break a finished compile
        pass

    packet_family_counts = Counter(packet.family.value for packet in result.packets)
    result.trace = telemetry.build_trace(
        artifact_count=len(artifacts),
        atom_count=len(result.atoms),
        entity_count=len(result.entities),
        edge_count=len(result.edges),
        packet_count=len(result.packets),
        parser_atom_counts=dict(parser_atom_counts),
        packet_family_counts=dict(packet_family_counts),
        parser_routing=manifest.parser_routing,
    )

    if hard_errors and not allow_errors:
        raise ValueError("Compile validation failed:\n" + "\n".join(hard_errors))
    if hard_errors and allow_errors:
        result.warnings = sorted(set(result.warnings + hard_errors))

    # PRODUCTION_GAPS P3.4 / P3.5: compute quality metrics + fail-loud
    # warnings.  Metrics are deterministic over the finalized result so
    # they're safe to surface in the JSON output and as telemetry.
    routing_source = "unknown"
    routing_confidence_value = 0.0
    if pack_routing_decision is not None:
        routing_source = pack_routing_decision.source
        routing_confidence_value = pack_routing_decision.confidence
    result.quality = compute_quality(
        result,
        pack_routing_source=routing_source,
        pack_routing_confidence=routing_confidence_value,
    )

    # Fail-loud: a parser that successfully routed a file but produced
    # zero atoms is a regression signal (XLSX header detection bug,
    # PDF table extraction failure, etc.).  We surface a clear ERROR-
    # adjacent warning so operators see it without diffing JSON.
    if result.quality.parsers_with_zero_atoms:
        names = ", ".join(result.quality.parsers_with_zero_atoms[:5])
        suffix = (
            f" (and {len(result.quality.parsers_with_zero_atoms) - 5} more)"
            if len(result.quality.parsers_with_zero_atoms) > 5 else ""
        )
        result.warnings = sorted(
            set(
                result.warnings
                + [f"WARNING: parser produced 0 atoms for: {names}{suffix}"]
            )
        )
    if result.quality.entity_resolution_rate < 0.30 and result.quality.atom_count >= 20:
        result.warnings = sorted(
            set(
                result.warnings
                + [
                    f"WARNING: low entity_resolution_rate "
                    f"({result.quality.entity_resolution_rate:.2f}); "
                    "atoms aren't getting entity_keys — review pack vocabulary"
                ]
            )
        )
    if result.quality.packet_specificity < 0.50 and result.quality.packet_count >= 5:
        result.warnings = sorted(
            set(
                result.warnings
                + [
                    f"WARNING: low packet_specificity "
                    f"({result.quality.packet_specificity:.2f}); "
                    "many packets anchor on `*:unknown` — review entity extraction"
                ]
            )
        )

    try:
        from app.core.ml_capabilities import build_compile_capabilities

        result.compile_capabilities = build_compile_capabilities()
    except Exception:
        result.compile_capabilities = None

    try:
        from app.learning.worker_retrain import maybe_retrain_after_compile

        maybe_retrain_after_compile(
            compile_id=result.compile_id,
            project_id=result.project_id,
        )
    except Exception:
        pass

    # Mirror what this compile read, so the next container starts warm instead
    # of re-paying Document Intelligence for pictures it has already read. One
    # immutable blob per batch: three workers can run at once and none can
    # overwrite another's entries. Gated on SOWSMITH_FEEDBACK_BLOB like the
    # training-row and correction mirrors.
    try:
        from app.core import cache_blob as _cb

        _cb.mirror_ocr()
        _cb.mirror_embed()
    except Exception:  # pragma: no cover - mirroring must never break a compile
        pass

    return result
