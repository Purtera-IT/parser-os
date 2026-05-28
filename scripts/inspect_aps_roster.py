"""See what the APS Attachment A roster actually looks like to structural detectors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz

APS = Path(r"C:\Users\lilli\test_deals\aps_fiber\artifacts\APS_fiber_RFP.pdf")
doc = fitz.open(str(APS))

print(f"APS Attachment A — {len(doc)} pages\n")

# 1. Page-by-page text sample
print("=" * 72)
print("PAGE TEXT (first 3 pages)")
print("=" * 72)
for i, page in enumerate(doc):
    if i >= 3:
        break
    txt = page.get_text() or ""
    print(f"\n--- page {i+1} ({len(txt)} chars) ---")
    print(txt[:2000])

# 2. Try fitz find_tables on each page
print("\n" + "=" * 72)
print("FITZ TABLE DETECTION")
print("=" * 72)
for i, page in enumerate(doc):
    try:
        tabs = page.find_tables()
    except Exception as e:
        print(f"page {i+1}: find_tables ERROR {type(e).__name__}: {e}")
        continue
    tables = list(tabs)
    if tables:
        print(f"page {i+1}: {len(tables)} table(s) found")
        for j, t in enumerate(tables):
            try:
                hdr = t.header.names if t.header else []
                data = t.extract()
                rows = data[1:] if hdr else data
                print(f"  table {j}: header={hdr} rows={len(rows)}")
                if rows:
                    print(f"    first row: {rows[0]}")
                    if len(rows) > 1:
                        print(f"    second row: {rows[1]}")
            except Exception as e:
                print(f"  table {j}: extract ERROR {type(e).__name__}: {e}")
    else:
        # Only show pages with no tables that DO have text
        txt = page.get_text() or ""
        if "APS-" in txt or "Site" in txt or "School" in txt:
            print(f"page {i+1}: NO tables but text mentions site/school ({len(txt)} chars text)")

# 3. Check what triggers the v53.10 "explicit roster declaration" gate
print("\n" + "=" * 72)
print("ROSTER DECLARATION CHECK")
print("=" * 72)
full_text = ""
for page in doc:
    full_text += (page.get_text() or "") + "\n\n"

import re
declarations = re.findall(r"(?i)(kind\s*=\s*physical_site|site\s+roster|facilit(?:y|ies)\s+list|location\s+list|school\s+list|site\s+inventory)", full_text)
print(f"Declaration matches found: {declarations[:10]}")

# 4. How many APS-NNN tokens are in the raw text?
aps_ids = re.findall(r"\bAPS-\d{2,4}\b", full_text)
print(f"\nAPS-NNN tokens in raw text: {len(aps_ids)} (unique: {len(set(aps_ids))})")
print(f"Sample: {sorted(set(aps_ids))[:10]}")

doc.close()
