"""Text of a Word 97-2003 (.doc) file, with no dependency.

010279's whole scope -- the RFP's task list, equipment counts and checklists --
sat in "785 POS Termainal Refresh RFP.doc" and the compile read it as 0 lines:
nothing in the pipeline opened the old binary format. This reads it the way
Word does: the OLE compound file, the WordDocument and table streams, and the
piece table (Clx) that maps text positions to bytes, each piece either cp1252
or UTF-16. Paragraphs end in \\r; a table cell ends in \\x07.
"""
from __future__ import annotations

import struct

_FREE, _END = 0xFFFFFFFF, 0xFFFFFFFE


class _Ole:
    def __init__(self, data: bytes):
        if data[:8] != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            raise ValueError("not an OLE compound file")
        self.d = data
        self.sec = 1 << struct.unpack_from("<H", data, 0x1E)[0]
        self.mini = 1 << struct.unpack_from("<H", data, 0x20)[0]
        n_fat, dir_start = struct.unpack_from("<II", data, 0x2C)
        self.mini_cutoff, mini_fat_start, n_mini_fat, difat_start, n_difat = struct.unpack_from("<IIIII", data, 0x38)
        difat = list(struct.unpack_from("<109I", data, 0x4C))
        s = difat_start
        for _ in range(n_difat):
            if s in (_FREE, _END):
                break
            vals = struct.unpack_from(f"<{self.sec // 4}I", self._sector(s))
            difat += vals[:-1]
            s = vals[-1]
        self.fat: list[int] = []
        for fs in difat[:n_fat]:
            if fs in (_FREE, _END):
                continue
            self.fat += struct.unpack_from(f"<{self.sec // 4}I", self._sector(fs))
        self.entries = self._dirs(self._chain(dir_start))
        root = self.entries[0]
        self.ministream = self._chain(root[1])[: root[2]]
        self.minifat: list[int] = []
        if n_mini_fat:
            raw = self._chain(mini_fat_start)
            self.minifat = list(struct.unpack_from(f"<{len(raw) // 4}I", raw))

    def _sector(self, i: int) -> bytes:
        off = (i + 1) * self.sec
        return self.d[off: off + self.sec]

    def _chain(self, start: int) -> bytes:
        out, s, seen = [], start, set()
        while s not in (_FREE, _END) and s < len(self.fat) and s not in seen:
            seen.add(s)
            out.append(self._sector(s))
            s = self.fat[s]
        return b"".join(out)

    def _minichain(self, start: int) -> bytes:
        out, s, seen = [], start, set()
        while s not in (_FREE, _END) and s < len(self.minifat) and s not in seen:
            seen.add(s)
            out.append(self.ministream[s * self.mini: (s + 1) * self.mini])
            s = self.minifat[s]
        return b"".join(out)

    @staticmethod
    def _dirs(raw: bytes) -> list[tuple[str, int, int]]:
        out = []
        for off in range(0, len(raw) - 127, 128):
            nlen = struct.unpack_from("<H", raw, off + 0x40)[0]
            name = raw[off: off + max(0, nlen - 2)].decode("utf-16-le", "ignore")
            start, size = struct.unpack_from("<II", raw, off + 0x74)
            out.append((name, start, size))
        return out

    def stream(self, name: str) -> bytes:
        for n, start, size in self.entries:
            if n == name:
                raw = self._minichain(start) if size < self.mini_cutoff else self._chain(start)
                return raw[:size]
        raise KeyError(name)


def doc_text(data: bytes) -> str:
    """The document's text: paragraphs split by \\r, table cells by \\x07."""
    ole = _Ole(data)
    wd = ole.stream("WordDocument")
    flags = struct.unpack_from("<H", wd, 0x0A)[0]
    table = ole.stream("1Table" if flags & 0x0200 else "0Table")
    # FibRgFcLcb97: fcClx / lcbClx sit at fixed offsets after the variable-length
    # csw / cslw blocks.
    csw = struct.unpack_from("<H", wd, 32)[0]
    pos = 34 + csw * 2
    cslw = struct.unpack_from("<H", wd, pos)[0]
    pos += 2 + cslw * 4 + 2
    fc_clx, lcb_clx = struct.unpack_from("<II", wd, pos + 0x108)
    clx = table[fc_clx: fc_clx + lcb_clx]
    i = 0
    while i < len(clx) and clx[i] == 0x01:            # skip Prc (property) blocks
        i += 3 + struct.unpack_from("<H", clx, i + 1)[0]
    if i >= len(clx) or clx[i] != 0x02:
        raise ValueError("no piece table")
    lcb = struct.unpack_from("<I", clx, i + 1)[0]
    plc = clx[i + 5: i + 5 + lcb]
    n = (lcb - 4) // 12
    cps = struct.unpack_from(f"<{n + 1}I", plc, 0)
    parts = []
    for k in range(n):
        fc = struct.unpack_from("<I", plc, (n + 1) * 4 + k * 8 + 2)[0]
        count = cps[k + 1] - cps[k]
        if fc & 0x40000000:                          # compressed: cp1252, one byte per char
            off = (fc & 0x3FFFFFFF) // 2
            parts.append(wd[off: off + count].decode("cp1252", "replace"))
        else:
            parts.append(wd[fc: fc + 2 * count].decode("utf-16-le", "replace"))
    return "".join(parts)


def doc_lines(data: bytes) -> list[str]:
    """One line per paragraph or table row; Word's field codes and control
    characters removed."""
    text = doc_text(data)
    # Field codes: \x13 code \x14 result \x15 -> keep the result.
    out, depth, keep = [], 0, []
    for ch in text:
        if ch == "\x13":
            depth += 1
            keep.append(False)
            continue
        if ch == "\x14" and keep:
            keep[-1] = True
            continue
        if ch == "\x15" and keep:
            keep.pop()
            depth -= 1
            continue
        if depth and not (keep and keep[-1]):
            continue
        out.append(ch)
    text = "".join(out)
    lines, row = [], []
    for para in text.replace("\x0b", "\r").split("\r"):
        cells = para.split("\x07")
        if len(cells) > 1:
            row += [c.strip() for c in cells[:-1]]
            tail = cells[-1].strip()
            if not tail:                              # a row ends in a doubled \x07
                if any(row):
                    lines.append(" | ".join(c for c in row if c))
                row = []
            continue
        if row:
            lines.append(" | ".join(c for c in row if c))
            row = []
        s = "".join(ch for ch in para if ch >= " " or ch == "\t").strip()
        if s:
            lines.append(s)
    if row:
        lines.append(" | ".join(c for c in row if c))
    return lines


__all__ = ["doc_text", "doc_lines"]
