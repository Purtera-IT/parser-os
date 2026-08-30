"""Read the PDFs that have no text layer, by looking at them.

16 of 248 PDFs carry almost no extractable text -- 36 pages -- and they are the
site evidence: floor plans, AP placement drawings, camera location maps, rack
layouts. Every text-based stage of the pipeline is blind to them, so they have
been sitting in the corpus contributing nothing.

They are drawings. So render each page and read it with a vision model, then feed
the result through the same classifier as everything else. The transcription is
stored as the document's text, which means it is subject to the same receipt rule:
a claim about one of these has to quote something that was actually seen.
"""
import base64, io, json, os, sys, time, urllib.request, urllib.error, concurrent.futures as cf
# Credentials come from the environment, as everywhere else in this service.
EP = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
KEY = os.environ.get("AZURE_OPENAI_API_KEY") or ""
DEPLOYMENT = os.environ.get("AZURE_OPENAI_VISION_DEPLOYMENT", "gpt-4.1-mini")

#: A page of real prose runs 1,500-3,000 characters. Below this, the text layer is
#: decoration and the document is effectively a picture.
EMPTY_TEXT_LAYER_CHARS_PER_PAGE = 120


def has_empty_text_layer(text: str, pages: int) -> bool:
    """True when a PDF is a drawing wearing a text layer, and needs to be looked at."""
    if not pages:
        return False
    return len(text or "") / pages < EMPTY_TEXT_LAYER_CHARS_PER_PAGE

PROMPT = """This page is from a document whose text layer is empty — it is a drawing, scan or
image. Transcribe what is actually on it, for someone pricing and delivering the work.

Capture, when present and only when you can actually see it:
  * the title block and any site, building, floor or room names
  * every legend entry and what each symbol means
  * COUNTS of anything countable: access points, cameras, drops, racks, TVs, panels
  * dimensions, port counts, rack units, labelled equipment
  * any note written on the drawing

Write plain text. Do not interpret or summarise the project. Do not guess at anything
illegible — write [illegible] instead. If the page is essentially blank or is pure
branding, say exactly that in one line."""


def pages_as_png(path, max_pages=6, dpi=140):
    import fitz
    doc = fitz.open(path)
    out = []
    for i in range(min(max_pages, len(doc))):
        pix = doc[i].get_pixmap(dpi=dpi)
        out.append(pix.tobytes("png"))
    doc.close()
    return out


def read_page(png):
    b64 = base64.b64encode(png).decode()
    body = json.dumps({"messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}],
        "temperature": 0, "max_tokens": 1200}).encode()
    url = f"{EP}/openai/deployments/{DEPLOYMENT}/chat/completions?api-version=2024-10-21"
    last = None
    for a in range(5):
        try:
            req = urllib.request.Request(url, body, {"content-type": "application/json", "api-key": KEY})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            return d["choices"][0]["message"]["content"], d.get("usage", {})
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in (429, 500, 502, 503, 504): raise
            time.sleep(int(e.headers.get("Retry-After") or 0) or min(45, 2 ** a * 3))
    raise last


def run(doc):
    try:
        pngs = pages_as_png(doc["path"])
    except Exception as e:
        return {**doc, "error": f"render: {type(e).__name__}"}
    texts, tok = [], 0
    for i, png in enumerate(pngs, 1):
        try:
            t, u = read_page(png)
            texts.append(f"[page {i}]\n{t.strip()}")
            tok += (u.get("prompt_tokens", 0), u.get("completion_tokens", 0))[0]
        except Exception as e:
            texts.append(f"[page {i}] (unread: {type(e).__name__})")
    return {**doc, "vision_text": "\n\n".join(texts), "pages_read": len(pngs), "prompt_tokens": tok}


if __name__ == "__main__":  # pragma: no cover - operational entry point
    SP = os.environ.get("LIFECYCLE_WORKDIR", ".")
    docs = json.load(open(f"{SP}/blind_pdfs.json"))
    print(f"reading {len(docs)} PDFs ({sum(d['pages'] for d in docs)} pages) with vision…", flush=True)
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        out = list(ex.map(run, docs))
    json.dump(out, open(f"{SP}/vision_text.json", "w"))
    ok = [r for r in out if r.get("vision_text")]
    print(f"read: {len(ok)}/{len(out)}")
    print(f"median chars recovered: {sorted(len(r['vision_text']) for r in ok)[len(ok)//2]:,}")
    print(f"total prompt tokens: {sum(r.get('prompt_tokens',0) for r in ok):,}")
