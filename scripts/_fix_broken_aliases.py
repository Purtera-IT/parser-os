"""One-shot script that repairs broken ``aliases_from`` references in the
domain packs after PR2.

The boss-review audit found 21 references pointing to ``device_aliases.X``
keys that don't actually exist in the corresponding pack. For each one we
either:

  - rewrite the path to the correct existing key, OR
  - drop the broken ``aliases_from`` entry and append a deterministic
    explicit ``aliases:`` list so the target still has a non-empty alias bag.

Idempotent: re-running on already-fixed files is a no-op.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOMAIN = REPO / "app" / "domain"


# Each entry: (file, target_key, broken_path, fix). ``fix`` is either
# ('rewrite', '<new device_alias key>') or
# ('replace_with_aliases', [<explicit aliases list>]).
FIXES: list[tuple[str, str, str, tuple]] = [
    ("access_control_pack.yaml", "card_reader",            "device_aliases.reader",       ("rewrite", "card_reader")),
    ("access_control_pack.yaml", "rex_device",             "device_aliases.rex",          ("rewrite", "rex_button")),
    ("av_pack.yaml",             "av_vc_camera",           "device_aliases.camera",       ("rewrite", "ptz_camera")),
    ("av_pack.yaml",             "av_rack",                "device_aliases.rack",         ("replace_with_aliases", ["rack", "equipment rack", "av rack"])),
    ("das_pack.yaml",            "das_antenna",            "device_aliases.antenna",      ("rewrite", "remote_antenna")),
    ("datacenter_field_pack.yaml","dc_rack",               "device_aliases.rack",         ("rewrite", "server_rack")),
    ("datacenter_field_pack.yaml","rack_pdu",              "device_aliases.pdu",          ("rewrite", "rack_pdu")),
    ("datacenter_field_pack.yaml","tor_switch",            "device_aliases.tor",          ("rewrite", "tor_switch")),
    ("edge_iot_security_pack.yaml","edge_door_reader",     "device_aliases.reader",       ("rewrite", "door_reader")),
    ("endpoint_imac_pack.yaml",  "laptop_dock",            "device_aliases.dock",         ("rewrite", "docking_station")),
    ("endpoint_imac_pack.yaml",  "monitor",                "device_aliases.monitor",      ("replace_with_aliases", ["monitor", "display", "screen"])),
    ("endpoint_imac_pack.yaml",  "user_printer",           "device_aliases.printer",      ("replace_with_aliases", ["printer", "user printer", "desktop printer"])),
    ("fire_safety_pack.yaml",    "fire_panel",             "device_aliases.panel",        ("rewrite", "fire_panel")),
    ("network_modernization_pack.yaml","nm_switch",        "device_aliases.switch",       ("rewrite", "l2_switch")),
    ("network_modernization_pack.yaml","nm_router",        "device_aliases.router",       ("rewrite", "branch_router")),
    ("paging_pack.yaml",         "paging_ceiling_speaker", "device_aliases.speaker",      ("rewrite", "ceiling_speaker")),
    ("paging_pack.yaml",         "paging_mic",             "device_aliases.microphone",   ("rewrite", "paging_mic")),
    ("pos_commerce_pack.yaml",   "pos_terminal",           "device_aliases.pos_terminal", ("rewrite", "pos_register")),
    ("pos_commerce_pack.yaml",   "pin_pad",                "device_aliases.pin_pad",      ("rewrite", "pos_pin_pad")),
    ("structured_backbone_fiber_pack.yaml","fiber_trunk",  "device_aliases.trunk",        ("rewrite", "fiber_trunk")),
    ("structured_backbone_fiber_pack.yaml","fiber_patch_panel","device_aliases.patch_panel",("rewrite","fiber_patch_panel")),
]


def _find_target_block(text: str, target_key: str) -> tuple[int, int] | None:
    """Locate the ``- key: <target_key>`` block in a domain YAML pack.

    Returns (start, end) line indices inclusive-exclusive, or None.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^\s*-\s*key:\s*{re.escape(target_key)}\s*$", line):
            start = i
            break
    if start is None:
        return None
    # Block ends at the next "- key:" sibling at the same indent or EOF.
    base_indent = len(lines[start]) - len(lines[start].lstrip())
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if not lines[j].strip():
            continue
        indent = len(lines[j]) - len(lines[j].lstrip())
        if lines[j].lstrip().startswith("- ") and indent == base_indent:
            end = j
            break
    return (start, end)


def _rewrite_aliases_from(block_lines: list[str], broken_path: str, new_key: str) -> tuple[list[str], bool]:
    new_full = f"device_aliases.{new_key}"
    out = []
    changed = False
    for line in block_lines:
        # Match `- device_aliases.X` inside aliases_from
        m = re.match(r"^(\s*-\s*)(device_aliases\.[A-Za-z0-9_]+)\s*$", line)
        if m and m.group(2) == broken_path:
            out.append(f"{m.group(1)}{new_full}")
            changed = True
        else:
            out.append(line)
    return out, changed


def _replace_aliases_from_with_aliases(
    block_lines: list[str], broken_path: str, explicit: list[str]
) -> tuple[list[str], bool]:
    """Drop the broken aliases_from entry and ensure an aliases: list is present.

    If the aliases_from list contained only the broken entry, remove the entire
    aliases_from key. Then add or merge an aliases: block containing ``explicit``.
    """
    if not block_lines:
        return block_lines, False

    out = list(block_lines)
    base_indent = len(out[0]) - len(out[0].lstrip())
    field_indent = base_indent + 2
    item_indent = field_indent + 2

    af_start = None
    for i, line in enumerate(out):
        if line.strip() == "aliases_from:":
            indent = len(line) - len(line.lstrip())
            if indent == field_indent:
                af_start = i
                break
    if af_start is None:
        return out, False

    # Collect aliases_from items.
    af_end = af_start + 1
    while af_end < len(out):
        line = out[af_end]
        if not line.strip():
            af_end += 1
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= field_indent:
            break
        af_end += 1
    items = [
        out[i].strip().lstrip("-").strip()
        for i in range(af_start + 1, af_end)
        if out[i].strip().startswith("-")
    ]
    items_remaining = [it for it in items if it != broken_path]
    if items == items_remaining:
        return out, False  # nothing to drop here

    # Rebuild the aliases_from region.
    new_region: list[str] = []
    if items_remaining:
        new_region.append(" " * field_indent + "aliases_from:")
        for it in items_remaining:
            new_region.append(" " * item_indent + f"- {it}")
    out = out[:af_start] + new_region + out[af_end:]

    # Now ensure aliases: list exists and includes ``explicit``.
    aliases_start = None
    for i, line in enumerate(out):
        if line.strip() == "aliases:":
            indent = len(line) - len(line.lstrip())
            if indent == field_indent:
                aliases_start = i
                break
    if aliases_start is None:
        # Insert at start of the field region (right after `- key:` and the line after it).
        insert_at = 1  # after the `- key:` line
        # Find first non-key field; insert before it.
        block_lines2 = out
        if len(block_lines2) > 1:
            insert_at = 1
        new_lines = [" " * field_indent + "aliases:"]
        for alias in explicit:
            new_lines.append(" " * item_indent + f"- {alias!r}")
        out = out[:insert_at] + new_lines + out[insert_at:]
        return out, True

    # aliases: already exists; collect existing items and merge.
    a_end = aliases_start + 1
    while a_end < len(out):
        line = out[a_end]
        if not line.strip():
            a_end += 1
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= field_indent:
            break
        a_end += 1
    existing = [
        out[i].strip().lstrip("-").strip().strip("'").strip('"')
        for i in range(aliases_start + 1, a_end)
        if out[i].strip().startswith("-")
    ]
    merged = list(existing)
    for alias in explicit:
        if alias not in merged:
            merged.append(alias)
    if merged == existing:
        return out, True  # alias list already had everything
    new_region = [" " * field_indent + "aliases:"]
    for alias in merged:
        new_region.append(" " * item_indent + f"- {alias!r}")
    out = out[:aliases_start] + new_region + out[a_end:]
    return out, True


def apply_fix(path: Path, target_key: str, broken_path: str, fix: tuple) -> bool:
    text = path.read_text(encoding="utf-8")
    span = _find_target_block(text, target_key)
    if span is None:
        return False
    lines = text.splitlines()
    start, end = span
    block_lines = lines[start:end]
    kind = fix[0]
    if kind == "rewrite":
        new_block, changed = _rewrite_aliases_from(block_lines, broken_path, fix[1])
    elif kind == "replace_with_aliases":
        new_block, changed = _replace_aliases_from_with_aliases(block_lines, broken_path, fix[1])
    else:
        raise ValueError(f"unknown fix kind: {kind}")
    if not changed:
        return False
    new_lines = lines[:start] + new_block + lines[end:]
    new_text = "\n".join(new_lines)
    if not text.endswith("\n"):
        # preserve trailing-newline behavior of the original
        pass
    else:
        new_text += "\n"
    path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    changed = 0
    for fname, target, broken, fix in FIXES:
        p = DOMAIN / fname
        if not p.exists():
            print(f"  SKIP (missing): {fname}")
            continue
        if apply_fix(p, target, broken, fix):
            changed += 1
            print(f"  FIXED {fname}: {target} ({broken} -> {fix[0]} {fix[1]})")
        else:
            print(f"  NOOP  {fname}: {target}")
    print(f"\ntotal: {changed}/{len(FIXES)} broken references repaired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
