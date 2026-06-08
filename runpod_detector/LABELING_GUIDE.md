# Schematic labeling guide (for Luke) — Roboflow → universal symbol detector

This is the canonical doc. Goal: a detector that finds devices + structure on
**any** firm's floor plan, even ones we've never seen. Read the principle first —
it changes how you label.

---

## 0. The one principle that drives everything

**We do NOT teach the model device *types*.** Every firm draws a camera/WAP/jack
differently, so a "camera class" trained on Firm A fails on Firm B. Instead:

1. Detect **one generic `device`** (just "a device symbol is here").
2. Detect + read the **legend** (the notes box) — it defines what each glyph means
   *on that sheet*.
3. Match a device's **tag/letter** to the legend → the real type, per drawing, with
   **no per-firm training**.

So the detector stays firm-agnostic forever; meaning comes from the legend each
time. Label with that in mind: be consistent about *what's a device*, don't stress
about *what kind*.

> This is also why "multi-label" is a second, smaller step: the detector boxes the
> device; a tiny **attribute head** on the crop predicts multi-label attributes
> (new/existing, ceiling/wall) and the **legend** supplies the type. Detection +
> attribute head + legend = the full picture. You only annotate boxes (+ a couple
> optional attributes); you do not hand-label every device type.

---

## 1. Core classes — label these (strict list, no synonyms)

| Class | What it is | Box |
|---|---|---|
| `device` | the placed symbol glyph (square/circle/triangle marker in a room). **ALL device types = this one class.** | tight, glyph only |
| `device_tag` | short text/letter by a glyph ("WAP", "C", "AP1") — feeds legend matching | tight on text |
| `room` | room boundary | polygon (or box if you only need presence) |
| `room_tag` | room name + number text | box |
| `legend` | the legend / notes block (symbol definitions) | box around the whole block |
| `legend_item` | one row inside the legend (glyph + its text definition) | box per row |
| `leader` | line/arrow connecting a tag/callout to a symbol or room | box or polyline |
| `cable_path` | conduit / cable run (long thin lines) | polyline/polygon |

**Naming is literal in Roboflow** — a typo makes a NEW class. Use these exact
strings, lowercase, every time.

---

## 2. The "extra stuff that shows up" — what to do with each

You WILL hit these. Here's the call so you never have to guess:

| You see… | Do this |
|---|---|
| Camera / WAP / speaker / TV / data jack / card reader / sensor — any device | `device` (+ `device_tag` if it has a code). **Don't make a class per type.** |
| WAP specifically | still `device`. (Only break out a `wap` class if Lilli says WAPs are the priority deliverable — ask first.) |
| The legend / "general notes" / symbol key | `legend`, and box each line as `legend_item` |
| Arrow / leader line / callout line | `leader` |
| Cable, conduit, homerun, raceway, long thin line | `cable_path` (NOT `device`) |
| Detail/section bubble (e.g. "3 / T4.04" circle with a tail) | `detail_ref` — *optional*, label only if Lilli wants cross-sheet refs; else skip |
| Title block (sheet name/number bottom-right) | `title_block` — *optional*, useful for metadata; else skip |
| **Stairs, elevators, doors, windows, walls, furniture, toilets** | **IGNORE — do not label.** Architectural, not in scope. |
| Grid bubbles (A–G, 1–5 circles in the margins) | IGNORE |
| Dimension lines, dimension text, scale bar, north arrow | IGNORE |
| Revision cloud, stamp, match line, hatching/fill | IGNORE |
| Random free text / general notes (not the legend) | IGNORE (unless it's a `room_tag`) |

Rule of thumb: **if we'd never count it, place it, or quote it in a deal, don't
label it.** When unsure, leave a Roboflow comment and tag the image `needs-review`.

---

## 3. Fix these in the existing annotations (current convention is inconsistent)

1. `DeviceSymbols`, `symbol`, `wap` are three names for the same thing → use Roboflow
   **Classes → Rename/Merge** to fold them into **`device`** (and the text ones into
   `device_tag`).
2. The long thin shapes currently tagged as symbols (wide/flat or tall/skinny
   polygons) are **cable runs** → reclass to `cable_path`. They're poisoning the
   device class.
3. The yellow "WAP" boxes are the **text tag above the glyph** → `device_tag`; the
   square below is `device`.
4. `stair` labels → **delete**.

---

## 4. Annotation rules for the best model (non-negotiables)

1. **One class list, defined.** Keep a pinned reference image with 2 example crops
   per class. Ambiguity is the #1 model killer.
2. **Label EVERY instance on every page.** A missed symbol trains the model that
   that spot is "background" → kills recall directly. Completeness beats speed.
3. **Tight boxes, glyph only** (not the surrounding text — that's `device_tag`).
4. **Boxes for symbols; polygons only for rooms/cables.** Don't polygon tiny glyphs.
5. **Tag every image with its FIRM** (Roboflow Tags, e.g. `firm:uri-telecom`). We
   split train/test **by firm**, so this tag is required for an honest score.
6. **Diversity > volume.** 25–30 *different firms* beats 300 pages of one firm. The
   model fails on unseen drawing *styles*, so variety is the single biggest lever
   (it's what moves held-out-firm mAP from ~0.1 toward 0.7+).
7. Always label the `legend` — the type-resolution step needs it on every sheet.

---

## 5. Roboflow setup (do it this way)

- **Project type:** Object Detection.
- **Ontology:** create the 8 classes up front; lock naming. Use **Classes** to
  merge the legacy dupes.
- **Image tags:** add `firm:<name>` and `sheet:<type>` to every image (for splitting
  + analysis).
- **Preprocessing:** Auto-Orient ON. **Tiling** ON (2×2 or 3×3) — symbols are ~30px
  on a 4680px sheet, tiling makes them big enough to learn. Resize tiles to 1280.
- **Augmentations (light — these are clean line drawings):** small rotation (±5°),
  brightness/contrast ±15%, slight blur/noise (simulates scans). **NO horizontal/
  vertical flips** — they mirror text and break `device_tag`/`room_tag`.
- **Split:** train/valid/test, but make the **test set whole held-out firms** (use
  the firm tag), not a random split.

---

## 6. The automation flywheel (your "hack" — set it up early)

Once you have a first model, you stop labeling from scratch. Each round the human
does less:

1. **Label ~200–300 boxes by hand** across a few firms → train **v1** (rough is fine).
2. **Model-assisted labeling:** in Roboflow, upload the next batch and run **Label
   Assist** with v1 — it pre-draws the boxes, you just **correct** (5–10× faster).
3. **Train v2** on the bigger set → it pre-labels the next, larger batch → train v3 →
   each round the corrections shrink. This compounding loop is the whole game.
4. **Active learning (label only what's hard):** deploy the model, push its
   predictions back to Roboflow, and **prioritize the LOW-CONFIDENCE images** for
   human review. Don't re-label what the model already nails — spend labels only
   where it's unsure. Max information per click.
5. **Legend auto-resolve (kills type labeling entirely):** detect `legend` + OCR its
   rows → build the glyph→meaning map per sheet → match floor devices to it. After
   this, you almost never hand-label device *types* — the legend does it.

> Foundation-model auto-label (Grounding DINO / SAM in Roboflow) is weak on tiny
> schematic glyphs — skip it. Your own v1 as the pre-labeler is far better here.

---

## 7. TL;DR to start today
Object Detection in Roboflow. 8 classes: `device, device_tag, room, room_tag,
legend, legend_item, leader, cable_path`. **One generic `device`** for all symbol
types — type comes from the **legend**, so don't class-per-device. **Ignore stairs/
doors/walls/grid/dimensions/furniture.** Tile images, light aug, NO flips, tag every
image with the firm, hold out whole firms for test. Label every instance, tight
boxes. After ~300 labels, train a rough model and switch to **Label Assist** so it
pre-labels and you just correct — then prioritize low-confidence images. **Get many
different firms — diversity is what makes it work on schematics we've never seen.**
