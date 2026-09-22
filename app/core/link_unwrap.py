"""Get the real URL back out of mail-gateway link wrappers.

A link that travels through Outlook ATP and Proofpoint arrives as
``https://nam13.safelinks.protection.outlook.com/?url=https%3A%2F%2Furldefense.com
%2Fv3%2F__https%3A%2Fhuzzard.com%2F...png__%3B!!...%24&data=...`` — 500 characters
of gateway, wrapping the one thing the PM wants: the diagram.

Live 010289: the ask's "Diagram:" link is the vendor's door-access drawing, and
the deal is unreadable without it.
"""
from __future__ import annotations

import re
from urllib.parse import parse_qs, unquote, urlsplit

#: What a browser will render inline.
IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")

_SAFELINKS_HOST_RE = re.compile(r"safelinks\.protection\.outlook\.com$", re.I)
# urldefense v3: .../v3/__<real url>__;<base64 marker>$
_URLDEFENSE_V3_RE = re.compile(r"/v3/__(?P<url>.+?)__;", re.S)
_URLDEFENSE_V2_RE = re.compile(r"/v2/url\?u=(?P<url>[^&]+)")
# urldefense escapes "/" as "*" inside the wrapped url
_MAX_HOPS = 4


def unwrap_link(url: str) -> str:
    """Peel gateway wrappers until a plain URL is left. Idempotent; a URL that
    is not wrapped comes back unchanged. Never raises."""
    current = (url or "").strip().strip("<>\"'")
    for _ in range(_MAX_HOPS):
        nxt = _unwrap_once(current)
        if nxt == current:
            break
        current = nxt
    return current


def _unwrap_once(url: str) -> str:
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if _SAFELINKS_HOST_RE.search(parts.netloc or ""):
        inner = (parse_qs(parts.query).get("url") or [""])[0]
        if inner:
            return unquote(inner)
    if "urldefense" in (parts.netloc or "").lower():
        m = _URLDEFENSE_V3_RE.search(url)
        if m:
            # urldefense writes the scheme with ONE slash ("https:/host/…") and
            # escapes further slashes as "*".
            inner = unquote(m.group("url")).replace("*", "/")
            return re.sub(r"^(https?:)/(?!/)", r"\1//", inner)
        m = _URLDEFENSE_V2_RE.search(url)
        if m:
            return unquote(m.group("url")).replace("-", "%").replace("_", "/")
    return url


def is_image_url(url: str) -> bool:
    """Does this URL point at something a browser can show inline?"""
    try:
        path = urlsplit((url or "").strip()).path.lower()
    except ValueError:
        return False
    return path.endswith(IMAGE_SUFFIXES)


__all__ = ["unwrap_link", "is_image_url", "IMAGE_SUFFIXES"]
