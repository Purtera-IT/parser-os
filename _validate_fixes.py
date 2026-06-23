"""Assertion harness: re-verify EVERY parser fix from this session produces the
exact expected output. Run offline (lexical fallbacks). Prints PASS/FAIL per check."""
import warnings; warnings.filterwarnings("ignore")
import glob, re
from pathlib import Path
from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser
from _labeler_core import parse_deal

P = OrbitBriefPdfParser()
FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def bk_pdf():
    f = "_blob_pool/burger_king_hme/557-04-08-2026.pdf"
    return P.parse_artifact("t", "t", Path(f)).atoms


def deal(slug):
    fs = [f for f in sorted(glob.glob(f"_blob_pool/{slug}/*"))
          if f.lower().endswith((".pdf", ".docx", ".xlsx"))]
    return parse_deal(fs)[0]


def cap(a):  # image expected_content
    return ((a.value if isinstance(a.value, dict) else {}) or {}).get("expected_content")


def body(a):  # works for both EvidenceAtom and labeler-dict
    return (getattr(a, "raw_text", None) or (a.get("body") if isinstance(a, dict) else "") or "").strip()


print("=== Burger King PDF (photo captions, Q&A, signature, fragments) ===")
bk = bk_pdf()
imgs = {}
for a in bk:
    if "binary_region_marker" in (a.review_flags or []):
        m = re.search(r"page(\d+)/image(\d+)", a.raw_text or "")
        if m:
            imgs[(int(m.group(1)), int(m.group(2)))] = cap(a)
check("p3 img19 -> 'Battery Charger Mounting'", imgs.get((3, 19)) == "Upload photo showing Battery Charger Mounting")
check("p3 img20 -> 'Headset Holder Mounting'", imgs.get((3, 20)) == "Upload photo showing Headset Holder Mounting")
check("p7 img34 (cable tester) -> carried 'POS 1' (not below 'POS 3')", imgs.get((7, 34), "").endswith("POS 1"))
check("p7 img35 (wall plate) -> 'POS 3'", imgs.get((7, 35), "").endswith("POS 3"))
check("p4 img23 != img24 (two distinct requests split)", imgs.get((4, 23)) != imgs.get((4, 24)))
btxt = [body(a) for a in bk]
check("no 'Mounting 4' junk atom", "Mounting 4" not in btxt)
check("no bare 'Mounting' fragment", "Mounting" not in btxt)
check("first question '2 LANE' present", any("2 LANE" in t for t in btxt))
check("POS Q has its answer ('...POS  Yes')", any("pull 2 cables" in t and "Yes" in t for t in btxt))
check("'How many total Cables...  8' paired", any("How many total Cables" in t and t.rstrip().endswith("8") for t in btxt))
p15 = [a for a in bk if (a.source_refs[0].locator.get("page") if a.source_refs else None) == 15]
p15b = [body(a) for a in p15]
check("p15 'Managers Name: Diedra Kennedy' merged", any(t == "Managers Name: Diedra Kennedy" for t in p15b))
check("p15 signature image captioned 'Signature'", any("binary_region_marker" in (a.review_flags or []) and cap(a) == "Signature" for a in p15))
check("p5 tablet photo request folded (no split text atom)", not any("showing the correct screen" in t and "awaiting OCR" not in t for t in btxt))
# section headers became breadcrumbs
secs = set()
for a in bk:
    for s in (a.source_refs[0].locator.get("section_path") or [] if a.source_refs else []):
        secs.add(s)
check("form sub-headers as sections (Tablet Install / POS Cabling)", "Tablet Install" in secs and "POS Cabling" in secs)
check("no time-as-section ('09:30 AM')", not any(re.fullmatch(r"\d{1,2}:\d{2}( ?[AP]M)?", s or "") for s in secs))

print("=== columbus (Q&A pairing, dotted SOW sections) ===")
col = deal("columbus_afb_norvet_0149")
colb = [body(a) for a in col]
check("Q&A paired ('wall drops...  Answer:')", any("wall drops" in t and "Answer:" in t for t in colb))
check("no orphan 'Answer:' atoms", not any(t.lower().startswith("answer:") for t in colb))
colsecs = set()
for a in col:
    for s in (a.get("section") or "").split(" > "):
        colsecs.add(s.strip())
check("dotted section '1.0 SCOPE' present", "1.0 SCOPE" in colsecs)
check("dotted section '2.1 GENERAL REQUIREMENTS' present", "2.1 GENERAL REQUIREMENTS" in colsecs)
check("no stale 'LIST OF TABLES' on SCOPE content", not any("SCOPE" in t and "This Statement of Work" in t and "LIST OF TABLES" in (a.get("section") or "") for a, t in zip(col, colb)))

print("=== cross-deal hygiene (junk, lead-ins, dates) ===")
for slug in ["ace_school_bc84", "anywair_uga", "burger_king_hme", "columbus_afb_norvet_0149",
             "fortinet_locations_8aa9", "sow_for_project_c869", "summit_dfd1"]:
    c = deal(slug)
    b = [body(a) for a in c]
    nb = [x for x in b if x and "awaiting OCR" not in x and not x.startswith("[")]
    glyph = [x for x in nb if not any(ch.isalnum() for ch in x)]
    barenum = [x for x in nb if all(not ch.isalpha() for ch in x) and any(ch.isdigit() for ch in x)]
    iso = [x for x in nb if "T00:00:00" in x]
    leadin = [x for x in nb if x.rstrip(":").lower() in ("services include", "this support is limited to")]
    url = [x for x in nb if re.fullmatch(r"(?i)(https?://)?www\.[\w.-]+", x)]
    ok = not (glyph or barenum or iso or leadin or url)
    check(f"{slug}: clean (glyph={len(glyph)} barenum={len(barenum)} iso={len(iso)} leadin={len(leadin)} url={len(url)})", ok)

print()
print("RESULT:", "ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
