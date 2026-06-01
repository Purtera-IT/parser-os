/*
 * OrbitBrief PM User Guide — beautiful Word handout (OPTBOT example)
 *
 * Generated content based on the actual live system as of 2026-05-31:
 *   deal:        841ea7e0-0e2f-412a-aebc-5794c199b85c (OPTBOT - Atlanta Office Refresh)
 *   compile_id:  83e8ae1a-24bc-4097-bd3e-7215dbee9399
 *   atoms:       433  · 241 typed · 192 prose
 *   entities:    155
 *   edges:       182  (127 cross-file)
 *   packets:     31
 *   files:       9    (5 pdf, 2 docx, 2 xlsx)
 *   sites:       5    canonical (ATL-CP-05, ATL-WEST-02, ATL-HQ-01, ATL-047-04, ATL-AIR-03)
 *   pipeline:    16/14 stages · 3m49s total
 */

const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Header, Footer, Table, TableRow, TableCell,
  AlignmentType, PageOrientation, LevelFormat, ExternalHyperlink, Bookmark,
  InternalHyperlink, HeadingLevel, BorderStyle, WidthType, ShadingType,
  TableOfContents, PageNumber, PageBreak, TabStopType, TabStopPosition,
} = require("docx");

// ───────── design tokens ─────────
const INK    = "1A1815";  // body text
const INK2   = "5C5648";  // secondary
const INK3   = "9B9384";  // tertiary
const PAPER  = "FAF6EE";  // page bg
const SURF   = "FFFFFF";  // panels
const ACCENT = "B8722C";  // burnt orange
const GREEN  = "2E7D5C";  // success
const AMBER  = "B8722C";  // warn
const RED    = "B33A3A";  // error
const RULE   = "E8E2D5";  // borders
const RULES  = "C8BFA8";  // stronger borders

const CONTENT_W = 9360;      // 6.5" at 1" margins
const HALF      = 4680;
const THIRD     = 3120;
const QUARTER   = 2340;

// ───────── helpers ─────────
const border = (color = RULE, sz = 4) => ({ style: BorderStyle.SINGLE, size: sz, color });
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const cellBorders = (c = RULE) => ({
  top: border(c), bottom: border(c), left: border(c), right: border(c),
});
const margins = (t = 100, b = 100, l = 140, r = 140) => ({ top: t, bottom: b, left: l, right: r });

function p(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    spacing: { before: opts.before ?? 0, after: opts.after ?? 80 },
    alignment: opts.align,
  });
}
function pMulti(runs, opts = {}) {
  return new Paragraph({
    children: runs,
    spacing: { before: opts.before ?? 0, after: opts.after ?? 80 },
    alignment: opts.align,
  });
}
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, color: INK, bold: true, font: "Inter", size: 40 })],
    spacing: { before: 360, after: 200 },
  });
}
function h2(text, color = ACCENT) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, color, bold: true, font: "Inter", size: 30 })],
    spacing: { before: 320, after: 160 },
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, color: INK, bold: true, font: "Inter", size: 24 })],
    spacing: { before: 240, after: 120 },
  });
}
function body(text, size = 22) {
  return new Paragraph({
    children: [new TextRun({ text, color: INK, font: "Inter", size })],
    spacing: { after: 120 },
  });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, color: INK3, font: "Inter", size: 18, italics: true })],
    spacing: { after: 160 },
  });
}
function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    children: [new TextRun({ text, color: INK, font: "Inter", size: 22 })],
    spacing: { after: 80 },
  });
}
function num(text) {
  return new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    children: [new TextRun({ text, color: INK, font: "Inter", size: 22 })],
    spacing: { after: 80 },
  });
}
function cell(content, opts = {}) {
  const {
    width = HALF, fill = SURF, color = INK, bold = false, size = 22,
    align = AlignmentType.LEFT, mono = false, mute = false,
    borderColor = RULE, vert = false, padTop = 100, padBottom = 100,
  } = opts;
  const useColor = mute ? INK3 : color;
  return new TableCell({
    borders: cellBorders(borderColor),
    width: { size: width, type: WidthType.DXA },
    shading: { fill, type: ShadingType.CLEAR },
    margins: margins(padTop, padBottom, 160, 160),
    children: Array.isArray(content) ? content : [
      new Paragraph({
        children: [new TextRun({
          text: String(content),
          font: mono ? "JetBrains Mono" : "Inter",
          size, bold, color: useColor,
        })],
        alignment: align,
      }),
    ],
  });
}
function statTile(label, value, sub, accent = INK) {
  return new TableCell({
    borders: cellBorders(RULE),
    width: { size: THIRD, type: WidthType.DXA },
    shading: { fill: SURF, type: ShadingType.CLEAR },
    margins: margins(160, 160, 200, 200),
    children: [
      new Paragraph({
        spacing: { after: 40 },
        children: [new TextRun({
          text: label.toUpperCase(), font: "JetBrains Mono", size: 16,
          color: INK3, bold: true, characterSpacing: 30,
        })],
      }),
      new Paragraph({
        spacing: { after: 40 },
        children: [new TextRun({ text: String(value), font: "Fraunces", size: 44, bold: true, color: accent })],
      }),
      new Paragraph({
        children: [new TextRun({ text: sub || "", font: "Inter", size: 16, color: INK3 })],
      }),
    ],
  });
}
function calloutBox(title, body, color = ACCENT, fill = "FFFBF2") {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({
      children: [new TableCell({
        borders: {
          top: { style: BorderStyle.SINGLE, size: 24, color },
          bottom: border(color, 6),
          left: border(color, 6),
          right: border(color, 6),
        },
        width: { size: CONTENT_W, type: WidthType.DXA },
        shading: { fill, type: ShadingType.CLEAR },
        margins: margins(200, 200, 240, 240),
        children: [
          new Paragraph({
            spacing: { after: 100 },
            children: [new TextRun({ text: title, font: "Inter", size: 22, bold: true, color })],
          }),
          ...(Array.isArray(body) ? body : [new Paragraph({
            children: [new TextRun({ text: body, font: "Inter", size: 20, color: INK })],
          })]),
        ],
      })],
    })],
  });
}
function rule(color = RULE) {
  return new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color, space: 1 } },
    spacing: { before: 80, after: 80 },
    children: [new TextRun("")],
  });
}
function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}
function chip(text, color = INK, fill = "F0E9DA") {
  return new TextRun({
    text: ` ${text} `, font: "JetBrains Mono", size: 16,
    color, highlight: undefined, bold: true,
    shading: { fill, type: ShadingType.CLEAR },
  });
}

// ═════════════════════════════════════════════════════════════════
//  COVER PAGE
// ═════════════════════════════════════════════════════════════════
const cover = [
  new Paragraph({ spacing: { before: 1600 }, children: [new TextRun("")] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({
      text: "PURPULSE · ORBITBRIEF",
      font: "JetBrains Mono", size: 22, color: ACCENT, bold: true, characterSpacing: 60,
    })],
    spacing: { after: 240 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({
      text: "The PM Field Guide",
      font: "Fraunces", size: 84, color: INK, bold: true,
    })],
    spacing: { after: 200 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({
      text: "How to read a parsed deal, work the OrbitBrief, and ship a clean SOW —",
      font: "Inter", size: 26, color: INK2, italics: true,
    })],
    spacing: { after: 60 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({
      text: "with every screen explained.",
      font: "Inter", size: 26, color: INK2, italics: true,
    })],
    spacing: { after: 600 },
  }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: "WALKTHROUGH DEAL", font: "JetBrains Mono", size: 16, color: INK3, characterSpacing: 60 })],
  }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: "OPTBOT · Atlanta Office Refresh", font: "Fraunces", size: 36, color: ACCENT, bold: true })],
  }),

  // The stat strip mockup that matches what they'll see
  new Table({
    alignment: AlignmentType.CENTER,
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [HALF / 2, HALF / 2, HALF / 2, HALF / 2],
    rows: [new TableRow({
      children: [
        statTile("Files", 9, "5 pdf · 2 docx · 2 xlsx"),
        statTile("Atoms", 433, "241 typed · 192 prose", GREEN),
        statTile("Entities", 155, "26 device · 18 money"),
        statTile("Sites", 5, "canonical"),
      ],
    })],
  }),

  new Paragraph({ spacing: { before: 1200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "v57.11 · current as of June 1, 2026", font: "JetBrains Mono", size: 16, color: INK3 })],
  }),
  pageBreak(),
];

// ═════════════════════════════════════════════════════════════════
//  TOC / what's inside
// ═════════════════════════════════════════════════════════════════
const tocSection = [
  h1("What's in this guide"),
  body("Six sections. Skim, or read end-to-end in 12 minutes. Every screen the PM will see — explained, with the OPTBOT deal as the running example."),
  new Paragraph({ spacing: { after: 80 } }),

  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [800, 2400, 6160],
    rows: [
      ["1", "Finding deals", "The quoting list, filters, jumping into a deal page"],
      ["2", "The deal page layout", "7-step workflow strip + 6 content tabs"],
      ["3", "Deal Artifacts — the parser", "Where 9 files become 433 typed atoms and 5 canonical sites"],
      ["4", "OrbitBrief — the brief", "What the LLM does with those atoms (PM handoff, customer questions, gaps)"],
      ["5", "SowSmith & SOW Versions", "Render the final Word doc and track revisions"],
      ["6", "How to report a bug", "What to include so the dev team can fix it the same day"],
    ].map(([n, title, desc]) =>
      new TableRow({
        children: [
          cell(n, { width: 800, fill: ACCENT, color: "FFFFFF", bold: true, align: AlignmentType.CENTER, size: 28, padTop: 160, padBottom: 160 }),
          cell(title, { width: 2400, bold: true, size: 22, fill: SURF }),
          cell(desc, { width: 6160, size: 20, mute: true, fill: SURF }),
        ],
      })
    ),
  }),
  pageBreak(),
];

// ═════════════════════════════════════════════════════════════════
//  SECTION 1 — finding deals
// ═════════════════════════════════════════════════════════════════
const section1 = [
  new Paragraph({ children: [new Bookmark({ id: "sec1", children: [new TextRun("")] })] }),
  new Paragraph({ spacing: { before: 0, after: 0 },
    children: [new TextRun({ text: "SECTION 1", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 60 })],
  }),
  h1("Finding deals"),
  body("Every deal lives under the Quoting tab in the top nav. The list is filterable by stage, owner, SA assignment, account, and pipeline."),

  h3("Top nav (always visible)"),
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({
      children: [new TableCell({
        borders: cellBorders(RULES),
        shading: { fill: INK, type: ShadingType.CLEAR },
        margins: margins(200, 200, 240, 240),
        width: { size: CONTENT_W, type: WidthType.DXA },
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "PurPulse  ", font: "Fraunces", size: 24, color: "FFFFFF", bold: true }),
            new TextRun({ text: " │  Dashboard  │  Accounts  │  ", font: "Inter", size: 22, color: "C9C2B1" }),
            new TextRun({ text: "Quoting", font: "Inter", size: 22, color: ACCENT, bold: true }),
            new TextRun({ text: "  │  Projects  │  Planning  │  Execution  │  Closeout", font: "Inter", size: 22, color: "C9C2B1" }),
          ],
        })],
      })],
    })],
  }),
  caption("Click Quoting → you land on the full deal list. ⌘K opens the global search to jump to any deal by name or HubSpot number."),

  h3("The quoting list"),
  body("Each deal card on the board shows the HubSpot stage, days open, amount, and a colored bar indicating SA assignment status (orange = unassigned, neutral = assigned)."),

  bullet("Filter by stage to see only \"Submitted for Quoting\" or \"Open – Awaiting Scope\"."),
  bullet("Filter by owner to see only your deals."),
  bullet("Filter by SA to see deals awaiting SA pickup."),
  bullet("Click any card → opens the deal page (defaults to Project Overview)."),
  bullet("Or paste the URL directly: …/pm/quoting/{dealId}?step=artifacts"),

  calloutBox(
    "Pro tip — the URL is the source of truth",
    [
      new Paragraph({ children: [new TextRun({
        text: "Every tab change updates the URL. So bookmark a tab, share a link, hit browser-back — all of it works. Example for OPTBOT:",
        font: "Inter", size: 20, color: INK,
      })] }),
      new Paragraph({ spacing: { before: 80 }, children: [new TextRun({
        text: "  …/pm/quoting/841ea7e0-…/?step=artifacts",
        font: "JetBrains Mono", size: 18, color: ACCENT,
      })] }),
    ],
  ),

  pageBreak(),
];

// ═════════════════════════════════════════════════════════════════
//  SECTION 2 — deal page layout
// ═════════════════════════════════════════════════════════════════
const section2 = [
  new Paragraph({ children: [new Bookmark({ id: "sec2", children: [new TextRun("")] })] }),
  new Paragraph({ spacing: { before: 0, after: 0 },
    children: [new TextRun({ text: "SECTION 2", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 60 })],
  }),
  h1("The deal page layout"),
  body("Once you're on a deal, the screen has three persistent rails — top header, workflow strip, content tabs. Everything else swaps based on which tab you're in."),

  h3("1 │ Header bar (top)"),
  // Mock the header bar
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [3120, 3120, 3120],
    rows: [new TableRow({ children: [
      cell([
        new Paragraph({ children: [new TextRun({ text: "OPTBOT", font: "JetBrains Mono", size: 18, color: INK3, characterSpacing: 30 })] }),
        new Paragraph({ children: [new TextRun({ text: "#000087", font: "JetBrains Mono", size: 18, color: INK3 })] }),
        new Paragraph({ children: [new TextRun({ text: "Atlanta Office Refresh", font: "Fraunces", size: 26, color: INK, bold: true })] }),
      ], { width: 3120, fill: PAPER, borderColor: RULE }),
      cell([
        new Paragraph({ children: [new TextRun({ text: " OPEN – AWAITING SCOPE ", font: "JetBrains Mono", size: 16, color: ACCENT, bold: true })] }),
        new Paragraph({ children: [new TextRun({ text: " SA UNASSIGNED ", font: "JetBrains Mono", size: 16, color: AMBER, bold: true })] }),
      ], { width: 3120, fill: PAPER, borderColor: RULE, align: AlignmentType.CENTER }),
      cell([
        new Paragraph({ children: [new TextRun({ text: "Owner Max Friedberger", font: "Inter", size: 18, color: INK })] }),
        new Paragraph({ children: [new TextRun({ text: "Close May 31", font: "Inter", size: 18, color: INK3 })] }),
        new Paragraph({ children: [new TextRun({ text: "[ SAVE QUOTE ]", font: "JetBrains Mono", size: 16, color: ACCENT, bold: true })] }),
      ], { width: 3120, fill: PAPER, borderColor: RULE, align: AlignmentType.RIGHT }),
    ] })],
  }),
  caption("Account · HubSpot # · Deal name on the left. Stage + SA status pills in the middle. Owner + close date + save action on the right."),

  h3("2 │ Workflow strip (7 steps)"),
  // mock the workflow strip
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: Array(7).fill(Math.floor(CONTENT_W / 7)),
    rows: [new TableRow({ children: [
      cell("1\nOverview", { width: 1337, size: 16, mono: true, align: AlignmentType.CENTER, fill: "EBE3D0", padTop: 140, padBottom: 140 }),
      cell("2\nScope sources", { width: 1337, size: 16, mono: true, align: AlignmentType.CENTER, fill: ACCENT, color: "FFFFFF", bold: true, padTop: 140, padBottom: 140 }),
      cell("3\nProject Needs", { width: 1337, size: 16, mono: true, align: AlignmentType.CENTER, fill: PAPER, mute: true, padTop: 140, padBottom: 140 }),
      cell("4\nSites & Ingest", { width: 1337, size: 16, mono: true, align: AlignmentType.CENTER, fill: PAPER, mute: true, padTop: 140, padBottom: 140 }),
      cell("5\nScope Review", { width: 1337, size: 16, mono: true, align: AlignmentType.CENTER, fill: PAPER, mute: true, padTop: 140, padBottom: 140 }),
      cell("6\nDeal Kit", { width: 1337, size: 16, mono: true, align: AlignmentType.CENTER, fill: PAPER, mute: true, padTop: 140, padBottom: 140 }),
      cell("7\nSOW Ready", { width: 1337, size: 16, mono: true, align: AlignmentType.CENTER, fill: PAPER, mute: true, padTop: 140, padBottom: 140 }),
    ] })],
  }),
  caption("Active step is highlighted in burnt-orange. Future steps are dimmed. You can jump forwards/back freely — the URL updates with ?step=…"),

  h3("3 │ Content tabs (the actual panels)"),
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [2400, 6960],
    rows: [
      ["Project Overview", "HubSpot deal metadata · stage timing · owner & SA · contacts · checklists. Read-only mirror of the HubSpot card."],
      ["Deal Artifacts", "★ Where the parser output lives. Files → 433 atoms → 155 entities → 5 canonical sites. THIS IS THE STAR OF THE SHOW."],
      ["OrbitBrief", "The LLM's brief: PM handoff, gaps, customer questions, sa focus. Built from the atoms in Deal Artifacts."],
      ["Deal Kit", "BOM, pricing, margins, scenarios. Built from the atoms + your inputs."],
      ["SowSmith", "Render the final SOW Word doc from the envelope."],
      ["SOW Versions", "Compare and promote SOW drafts."],
    ].map(([title, desc]) =>
      new TableRow({
        children: [
          cell(title, { width: 2400, bold: true, size: 22, fill: title === "Deal Artifacts" ? ACCENT : SURF, color: title === "Deal Artifacts" ? "FFFFFF" : INK }),
          cell(desc, { width: 6960, size: 20, fill: SURF, mute: false }),
        ],
      })
    ),
  }),

  pageBreak(),
];

// ═════════════════════════════════════════════════════════════════
//  SECTION 3 — DEAL ARTIFACTS (the parser)
// ═════════════════════════════════════════════════════════════════
const section3 = [
  new Paragraph({ children: [new Bookmark({ id: "sec3", children: [new TextRun("")] })] }),
  new Paragraph({ spacing: { before: 0, after: 0 },
    children: [new TextRun({ text: "SECTION 3", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 60 })],
  }),
  h1("Deal Artifacts — the parser"),
  body("This is the workhorse tab. Drop files in, hit Re-parse, get structured atoms back. Everything downstream (OrbitBrief, Deal Kit, SOW) reads from here."),

  // OPTBOT actual numbers
  h2("OPTBOT at a glance", INK),
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [HALF/2, HALF/2, HALF/2, HALF/2],
    rows: [new TableRow({ children: [
      statTile("Files", 9, "5 pdf · 2 docx · 2 xlsx"),
      statTile("Pages", 23, "total pages"),
      statTile("Atoms", 433, "241 typed · 192 prose", GREEN),
      statTile("Parse", "OK", "no degraded files", GREEN),
    ] })],
  }),
  new Paragraph({ spacing: { after: 40 } }),
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [HALF/2, HALF/2, HALF/2, HALF/2],
    rows: [new TableRow({ children: [
      statTile("Entities", 155, "device · date · money · stakeholder"),
      statTile("Edges", 182, "127 cross-file"),
      statTile("Packets", 31, "bundled scope units"),
      statTile("Sites", 5, "ATL-* canonical"),
    ] })],
  }),

  // Anatomy of the page
  pageBreak(),
  h2("Anatomy — what's on screen, left to right"),

  // LEFT RAIL
  h3("⊢ Left rail: SOURCES"),
  body("The intake tracker. Counts files, transcripts, notes; shows ingestion progress; lists connected sources (Fireflies / Drive / HubSpot)."),

  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [2800, 6560],
    rows: [
      ["Files / Transcripts / Notes", "Three counters across the top. OPTBOT: 9 files · 0 transcripts · 0 notes."],
      ["INGESTION · N/M", "Strip showing Parsed / Processing / Queued / Failed. Should always read N/N on a healthy deal."],
      ["Connected sources", "FF=Fireflies, GD=Drive, HS=HubSpot. \"Manage →\" opens the source picker."],
      ["Inspect lineage", "Per-field provenance — which file/cell each atom came from. Audit trail for legal review."],
    ].map(([k, v]) => new TableRow({
      children: [
        cell(k, { width: 2800, bold: true, size: 20, fill: PAPER }),
        cell(v, { width: 6560, size: 20, fill: SURF }),
      ],
    })),
  }),

  // COMPILE SNAPSHOT
  pageBreak(),
  h3("⊤ Top center: Compile snapshot"),
  body("Single status line confirming the latest compile. PM glances at this before reading anything else."),

  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({
      children: [new TableCell({
        borders: cellBorders(RULES),
        width: { size: CONTENT_W, type: WidthType.DXA },
        shading: { fill: "F0E9DA", type: ShadingType.CLEAR },
        margins: margins(180, 180, 240, 240),
        children: [
          new Paragraph({ spacing: { after: 80 },
            children: [new TextRun({ text: "000087 - (Test) OPTBOT - Atlanta Office Refresh", font: "Fraunces", size: 26, bold: true, color: INK })],
          }),
          new Paragraph({ spacing: { after: 100 },
            children: [new TextRun({ text: "deal 841ea7e0  ·  compile 83e8ae1a-24b  ·  generated 5/29/2026, 2:20 PM", font: "JetBrains Mono", size: 18, color: INK3 })],
          }),
          new Paragraph({ children: [
            new TextRun({ text: " ✓ Compile complete  ", font: "JetBrains Mono", size: 18, color: GREEN, bold: true, shading: { fill: "DEF0E5", type: ShadingType.CLEAR } }),
            new TextRun({ text: "    ", font: "Inter", size: 18 }),
            new TextRun({ text: " RE-PARSE ", font: "JetBrains Mono", size: 18, color: "FFFFFF", bold: true, shading: { fill: INK, type: ShadingType.CLEAR } }),
          ] }),
        ],
      })],
    })],
  }),
  caption("Re-parse triggers the full parser-os pipeline again on the existing files. Use it when files change or after a parser version bump."),

  // PIPELINE
  h3("≣ Top right: Pipeline timeline (16 stages)"),
  body("Live progress while a compile runs; durations once it's done. Each tile = one parser-os stage. Click into a tile to see input/output counts."),

  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: Array(7).fill(Math.floor(CONTENT_W / 7)),
    rows: [
      ["discover","parse","adjudic","replay","conf","prose","enrich"],
      ["2ms","2.4s","0ms","1.4s","0ms","1ms","1m45s"],
      ["classify","collapse","dedup","calib","resolve","graph","packet"],
      ["1m49s","2ms","7ms","4ms","10.4s","615ms","50ms"],
    ].map((row, ri) => new TableRow({ children: row.map(c => cell(c, {
      width: Math.floor(CONTENT_W/7),
      size: ri % 2 === 0 ? 16 : 18,
      mono: true,
      align: AlignmentType.CENTER,
      bold: ri % 2 === 1,
      fill: ri % 2 === 0 ? PAPER : SURF,
      color: ri % 2 === 0 ? INK3 : INK,
      padTop: 100, padBottom: 100,
    })) })),
  }),
  caption("OPTBOT total: 3m49s. The two LLM-heavy stages (enrich · classify) dominate; everything else is structural and runs in milliseconds."),

  // STAT TILES
  pageBreak(),
  h3("⊟ Stat strip: the 7 numbers PMs check first"),
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [2200, 1400, 5760],
    rows: [
      ["Files", "9", "How many source documents (PDFs, DOCX, XLSX) the parser consumed."],
      ["Pages", "23", "Total pages across all files — gives you a rough volume signal."],
      ["Atoms", "433", "Smallest unit of meaning. \"241 typed · 192 prose\": typed = classified, prose = raw text. Higher typed-ratio = richer extraction."],
      ["Entities", "155", "Named things — devices, dates, money, sites, people. The hover hints them: 26 device · 18 money · 16 milestone · 11 stakeholder…"],
      ["Edges", "182", "Relations between atoms and entities. 127 cross-file means the parser tied facts in different files together (e.g. site A in the BOM = site A in the schedule)."],
      ["Packets", "31", "Bundled scope units — \"this site + these atoms + this BOM line\" — ready for the SOW to render."],
      ["Parse", "OK", "If anything failed, you'd see N issue here in amber/red."],
    ].map(([label, val, desc]) => new TableRow({
      children: [
        cell(label.toUpperCase(), { width: 2200, mono: true, bold: true, size: 18, fill: PAPER, color: INK3 }),
        cell(val, { width: 1400, mono: true, bold: true, size: 26, fill: PAPER, color: ACCENT, align: AlignmentType.CENTER }),
        cell(desc, { width: 5760, size: 20, fill: SURF }),
      ],
    })),
  }),

  // ATOM TYPES
  pageBreak(),
  h3("⌬ Atom type pills — click to filter the inspector"),
  body("Below the stat strip, the atom-type filter pills. Click one to filter the source file table and the right-pane inspector."),
  body("OPTBOT's top types (out of 38 distinct types extracted):"),

  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [3120, 1400, 4840],
    rows: [
      ["raw_table_row", "98", "Direct rows from a structured table — typically BOM / pricing / site rosters."],
      ["scope_item", "94", "A discrete deliverable, requirement, or commitment in prose."],
      ["site_allocation", "30", "Which site gets which scope/quantity."],
      ["vendor_line_item", "16", "Pricing line — quantity + sku + price."],
      ["task / dependency / quantity", "15 each", "Project schedule rows — what to do, what blocks it, how many."],
      ["milestone_phase", "14", "Gate dates in the project timeline."],
      ["bom_line", "13", "Pure BOM rows (vs. raw_table_row, which is unclassified)."],
      ["lead_time_constraint", "12", "Vendor lead times the SOW must respect."],
      ["stakeholder", "11", "Person + role + contact."],
      ["risk + mitigation", "10 + 5", "Risk register, with proposed mitigations."],
      ["site_access_window", "8", "When techs can be on-site."],
    ].map(([n, c, d]) => new TableRow({
      children: [
        cell(n, { width: 3120, mono: true, bold: true, size: 18 }),
        cell(c, { width: 1400, mono: true, size: 20, align: AlignmentType.CENTER, color: ACCENT, bold: true }),
        cell(d, { width: 4840, size: 20, mute: false }),
      ],
    })),
  }),

  // SITES
  pageBreak(),
  h3("⊕ Site readiness — canonical sites extracted"),
  body("Bottom-left of the panel. The parser reconciles the same site mentioned in different files (\"OPTBOT HQ\", \"Atlanta HQ\", \"1200 Peachtree\") into one canonical record with confidence."),
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [1800, 4560, 1500, 1500],
    rows: [
      ["SITE ID", "NAME · ADDRESS", "CONF", "STATUS"],
      ["ATL-HQ-01", "OPTBOT Atlanta HQ · 1200 Peachtree St NE", "0.90", "high"],
      ["ATL-WEST-02", "OPTBOT West Campus · 3100 Interstate N Pkwy", "0.85", "high"],
      ["ATL-AIR-03", "OPTBOT Airport Logistics · 6000 N Terminal Pkwy", "0.90", "high"],
      ["ATL-047-04", "OPTBOT Brady Training · 047 Brady Ave NW", "0.90", "high"],
      ["ATL-CP-05", "OPTBOT College Park S · 1850 Sullivan Rd", "0.90", "high"],
    ].map((r, i) => new TableRow({
      children: r.map((c, ci) => cell(c, {
        width: [1800, 4560, 1500, 1500][ci],
        size: i === 0 ? 16 : 20,
        mono: i === 0 || ci === 0 || ci === 2,
        bold: i === 0,
        fill: i === 0 ? INK : (i % 2 === 0 ? SURF : PAPER),
        color: i === 0 ? "FFFFFF" : INK,
        align: i === 0 ? AlignmentType.LEFT : (ci >= 2 ? AlignmentType.CENTER : AlignmentType.LEFT),
      })),
    })),
  }),

  // SOURCE FILES TABLE
  pageBreak(),
  h3("⊞ Source files table — click a row, see its atoms"),
  body("OPTBOT's 9 files, sorted by atom count:"),
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [4800, 1100, 1100, 2360],
    rows: [
      ["FILE", "ATOMS", "STATUS", "TOP TYPES"],
      ["04_hardware_bill_of_materials.xlsx (14.8 KB)", "120", "ok", "30 site_allocation · 16 raw_table_row"],
      ["02_statement_of_work.docx (39.5 KB)", "84", "ok", "30 raw_table_row · 19 scope_item"],
      ["03_site_surveys_and_requirements.docx (38.3 KB)", "76", "ok", "37 raw_table_row · 22 scope_item"],
      ["05_project_schedule_and_cutover_plan.xlsx (15.1 KB)", "63", "ok", "15 task · 15 dependency"],
      ["09_commercial_pricing_acceptance_…final.pdf (4.5 KB)", "34", "ok", "16 scope_item · 4 payment_term"],
      ["01_deal_overview_executive_brief.pdf (7.8 KB)", "21", "ok", "11 scope_item · 6 stakeholder"],
      ["08_site_roster_and_facilities_authoritative.pdf (3.3 KB)", "17", "ok", "5 physical_site · 3 stakeholder"],
      ["07_contracting_procurement_packet.pdf (4.9 KB)", "10", "ok", "6 scope_item"],
      ["06_security_it_integration_notes.pdf", "8", "ok", "4 scope_item · 2 dependency"],
    ].map((r, i) => new TableRow({
      children: r.map((c, ci) => cell(c, {
        width: [4800, 1100, 1100, 2360][ci],
        size: i === 0 ? 14 : 18,
        bold: i === 0,
        mono: ci === 1 || ci === 2,
        fill: i === 0 ? INK : (i % 2 === 0 ? SURF : PAPER),
        color: i === 0 ? "FFFFFF" : INK,
        align: ci === 1 || ci === 2 ? AlignmentType.CENTER : AlignmentType.LEFT,
      })),
    })),
  }),
  caption("Click any row → right pane shows the file's atoms with full text, atom_type, confidence, entity_keys, and source page."),

  // ATOM INSPECTOR
  pageBreak(),
  h3("⌽ Atom inspector — every atom in plain text"),
  body("Right side of the page once you select a file. Each atom is one card showing what was extracted from where."),

  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({
      children: [new TableCell({
        borders: cellBorders(RULES),
        shading: { fill: SURF, type: ShadingType.CLEAR },
        margins: margins(160, 160, 200, 200),
        children: [
          new Paragraph({ spacing: { after: 80 }, children: [
            new TextRun({ text: " scope_item ", font: "JetBrains Mono", size: 16, color: "FFFFFF", bold: true, shading: { fill: ACCENT, type: ShadingType.CLEAR } }),
            new TextRun({ text: "    conf 0.78  ·  verified", font: "JetBrains Mono", size: 16, color: INK3 }),
          ]}),
          new Paragraph({ spacing: { after: 60 }, children: [new TextRun({
            text: "Property has 23 dwellings, approx 8 have second story (walking stairs, no elevator). Programming is pretty easy, but not via thumb drive — several categories and menus within TV to manually set the IP addressing.",
            font: "Inter", size: 20, color: INK,
          })]}),
          new Paragraph({ children: [
            new TextRun({ text: " device:display ", font: "JetBrains Mono", size: 14, color: INK2, shading: { fill: "F0E9DA", type: ShadingType.CLEAR } }),
            new TextRun({ text: "      page 3", font: "JetBrains Mono", size: 14, color: INK3 }),
          ]}),
        ],
      })],
    })],
  }),
  caption("This is a real atom from the parser. atom_type pill on top-left, confidence/verification status on the right, the extracted text in the body, entity_keys + source page at the bottom."),

  pageBreak(),
];

// ═════════════════════════════════════════════════════════════════
//  SECTION 4 — OrbitBrief tab
// ═════════════════════════════════════════════════════════════════
const section4 = [
  new Paragraph({ children: [new Bookmark({ id: "sec4", children: [new TextRun("")] })] }),
  new Paragraph({ spacing: { before: 0, after: 0 },
    children: [new TextRun({ text: "SECTION 4", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 60 })],
  }),
  h1("OrbitBrief — the LLM brief"),
  body("Same source atoms, packaged for PM consumption: \"what's the deal about · what's missing · what should I ask the customer · what should the SA focus on.\""),

  h2("What you'll see when you click into OrbitBrief", INK),

  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [2800, 6560],
    rows: [
      ["PM status", "Red / amber / green. Derived from gap-coverage, risk count, and scope completeness. OPTBOT today: RED (still missing CFO signoff packet)."],
      ["One-line summary", "Single sentence headline: \"5-site Atlanta office refresh, ~$1.5M, May 31 close, awaiting CFO approval for procurement.\""],
      ["Domains", "What scope buckets this deal spans (audio_visual, network, low_voltage, etc). OPTBOT touches audio_visual + procurement."],
      ["Sites", "Same 5 canonical sites from Deal Artifacts, with per-site readiness scores."],
      ["Customer questions", "The auto-generated question list. \"Confirm the model # for the displays at College Park\", \"Does the West Campus circuit upgrade need to wait on landlord approval?\" — etc."],
      ["Gaps", "Required scope categories not yet covered. Each gap links back to the atoms that suggest it."],
      ["Facts by category", "Pricing / schedule / stakeholders / sites / risks — grouped digest of the atoms."],
      ["SA focus", "What the assigned SA should pick up first: highest-risk site, biggest pricing question, blocking dependency."],
    ].map(([k, v]) => new TableRow({
      children: [
        cell(k, { width: 2800, bold: true, size: 22, fill: PAPER }),
        cell(v, { width: 6560, size: 20, fill: SURF }),
      ],
    })),
  }),

  pageBreak(),
  h3("The three Re- buttons"),
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [2400, 6960],
    rows: [
      ["Re-parse", "Only re-runs parser-os. Use when files changed but the deal context didn't. ~7s on small deals, ~10 min on big ones."],
      ["Re-brief", "Skip parser-os, regenerate the LLM brief from the existing envelope. ~90s. Use when the brief looks off but you trust the atoms."],
      ["Re-render SOW", "Skip parser + brief; just re-render the SOW Word doc from the latest atoms. Sub-second once envelope exists."],
    ].map(([k, v]) => new TableRow({ children: [
      cell(k, { width: 2400, mono: true, bold: true, size: 20, fill: INK, color: "FFFFFF" }),
      cell(v, { width: 6960, size: 20, fill: SURF }),
    ]})),
  }),
  pageBreak(),
];

// ═════════════════════════════════════════════════════════════════
//  SECTION 5 — SowSmith & SOW Versions
// ═════════════════════════════════════════════════════════════════
const section5 = [
  new Paragraph({ children: [new Bookmark({ id: "sec5", children: [new TextRun("")] })] }),
  new Paragraph({ spacing: { before: 0, after: 0 },
    children: [new TextRun({ text: "SECTION 5", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 60 })],
  }),
  h1("SowSmith & SOW Versions"),
  body("Two tabs that close the loop. SowSmith builds the SOW; SOW Versions tracks drafts."),

  h2("SowSmith tab", INK),
  body("Preview pane on the left, action buttons on the right."),
  num("Click the active SOW draft to preview it inline (full DOCX rendered in-browser)."),
  num("\"Re-render\" rebuilds the doc from the latest envelope. Use after Re-parse or Re-brief."),
  num("\"Promote to next version\" snapshots the current SOW into SOW Versions and bumps the version number."),
  num("\"Export DOCX\" downloads the Word file."),
  num("\"Open in Google Drive\" syncs to the customer-facing drive folder."),

  h2("SOW Versions tab", INK),
  body("Side-by-side compare of any two versions, with rolled-back highlights."),
  bullet("Each version has: version number · created at · who promoted it · changed atom count vs prior."),
  bullet("Click \"Diff vs v1\" to see what changed."),
  bullet("Click \"Restore\" to make any old version the active draft."),

  calloutBox(
    "SowSmith vs OrbitBrief vs Deal Artifacts — the data flow",
    [
      new Paragraph({ children: [new TextRun({
        text: "Deal Artifacts → atoms.  OrbitBrief → digested brief.  SowSmith → rendered SOW.",
        font: "Inter", size: 20, color: INK,
      })]}),
      new Paragraph({ spacing: { before: 80 }, children: [new TextRun({
        text: "All three read from the SAME envelope.json. Edit upstream (Re-parse), and SowSmith picks it up automatically on Re-render. Never directly edit a SOW — change the source.",
        font: "Inter", size: 20, color: INK, italics: true,
      })]}),
    ],
    GREEN,
    "E8F0EA",
  ),

  pageBreak(),
];

// ═════════════════════════════════════════════════════════════════
//  SECTION 6 — bug reports
// ═════════════════════════════════════════════════════════════════
const section6 = [
  new Paragraph({ children: [new Bookmark({ id: "sec6", children: [new TextRun("")] })] }),
  new Paragraph({ spacing: { before: 0, after: 0 },
    children: [new TextRun({ text: "SECTION 6", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 60 })],
  }),
  h1("How to report a bug"),
  body("The dev team treats every bug report as a chance to improve the parser. Good reports get fixed same-day. Bad reports become \"could you screenshot it?\" delays. Use this template."),

  h2("The bug report template", INK),

  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [3000, 6360],
    rows: [
      ["What you were doing", "1 sentence. \"Clicked Re-parse on OPTBOT after re-uploading the BOM.\""],
      ["What you expected", "1 sentence. \"Pipeline should rerun and show 9 files parsed.\""],
      ["What you saw instead", "1 sentence. \"Pipeline stuck at 0/14 stages for 3 minutes.\""],
      ["URL", "Copy from browser. Includes deal id + step."],
      ["Deal ID (8-char prefix)", "Find on the page: \"deal 841ea7e0 …\"."],
      ["Compile ID (12-char prefix)", "Find on the page: \"compile 83e8ae1a-24b\"."],
      ["Time it happened", "Approximate. \"~3:15 PM ET\". Worker logs are stamped — devs need a window."],
      ["Screenshot", "Required if the bug is visual. Include the full page chrome."],
      ["Console errors (if any)", "Open DevTools (F12) → Console tab → screenshot any red text."],
    ].map(([k, v]) => new TableRow({
      children: [
        cell(k, { width: 3000, bold: true, size: 20, fill: PAPER }),
        cell(v, { width: 6360, size: 20, fill: SURF }),
      ],
    })),
  }),

  h3("Where to send it"),
  bullet("Slack #parser-os-feedback channel — quickest response."),
  bullet("GitHub issue at github.com/Purtera-IT/parser-os — for tracked work."),
  bullet("Or: paste the template into the deal's Discussion tab. Devs subscribe to deal-level mentions."),

  h2("What gets fixed fastest", INK),
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [2400, 6960],
    rows: [
      ["★★★ SAME DAY", "Crashes · 404s · stuck-at-queued for >5 min · phantom entities · obviously-wrong extractions"],
      ["★★ THIS WEEK", "Missing atoms / wrong atom type · entity dupes · slow compiles · UI text clips"],
      ["★ NEXT SPRINT", "Cosmetic / wording / 0-value tile labels · new feature requests"],
    ].map(([k, v]) => new TableRow({ children: [
      cell(k, { width: 2400, mono: true, bold: true, size: 18, fill: PAPER, color: ACCENT }),
      cell(v, { width: 6960, size: 20, fill: SURF }),
    ]})),
  }),

  h3("Self-service recovery (before you file a bug)"),
  num("Refresh the page (Ctrl+Shift+R / Cmd+Shift+R)."),
  num("Click Re-parse once more — transient errors auto-retry."),
  num("Open another deal — confirms whether it's deal-specific or system-wide."),
  num("Wait 90 seconds — first request after idle can take 30+ sec (cold start)."),

  calloutBox(
    "Things that look like bugs but aren't",
    [
      new Paragraph({ children: [new TextRun({ text: "• EDGES: 0 + \"single-file scope\"  →  Only one file means no inter-file relations. Not broken.", font: "Inter", size: 20, color: INK })]}),
      new Paragraph({ children: [new TextRun({ text: "• ENTITIES: 0 + \"thin input\"  →  Source files have no devices/sites/orgs to extract. Not broken.", font: "Inter", size: 20, color: INK })]}),
      new Paragraph({ children: [new TextRun({ text: "• Tailscale OFFLINE in smoke test  →  Stale heartbeat. If compiles succeed, Ollama is fine.", font: "Inter", size: 20, color: INK })]}),
      new Paragraph({ children: [new TextRun({ text: "• Compile takes 8+ minutes on a real RFP  →  LLM classify stage is slow; that's normal.", font: "Inter", size: 20, color: INK })]}),
      new Paragraph({ children: [new TextRun({ text: "• \"This deal is not available in PurPulse\"  →  Deal is archived in HubSpot. Restore from HubSpot side.", font: "Inter", size: 20, color: INK })]}),
    ],
    INK3,
    PAPER,
  ),
  pageBreak(),
];

// ═════════════════════════════════════════════════════════════════
//  REFERENCE / appendix
// ═════════════════════════════════════════════════════════════════
const appendix = [
  new Paragraph({ spacing: { before: 0, after: 0 },
    children: [new TextRun({ text: "APPENDIX", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 60 })],
  }),
  h1("Quick reference"),

  h3("The 38 atom types — alphabetical"),
  body("Click any pill to filter the inspector. Most-common first:"),
  new Paragraph({ children: [new TextRun({
    text: "raw_table_row · scope_item · site_allocation · vendor_line_item · task · dependency · quantity · milestone_phase · bom_line · lead_time_constraint · stakeholder · risk · site_access_window · acceptance_criterion · requirement · service_line · approval_authority · payment_term · integration_checkpoint · physical_site · mitigation · deal_metadata · constraint · commercial_total · site_attribute · site_budget · site_access_restriction · exclusion · cutover_step · decision · metadata_requirement · blackout_date_range · pricing_assumption · compliance_classification · open_question · site_room_mix · customer_instruction · assumption · electrical_acceptance_test",
    font: "JetBrains Mono", size: 16, color: INK,
  })], spacing: { after: 200 } }),

  h3("Pipeline stages (parser-os) — what each does"),
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [2000, 7360],
    rows: [
      ["discover", "Open each file, identify type, route to the right parser."],
      ["parse", "Per-file: PDF page extraction, DOCX table walking, XLSX cell scanning. Produces raw atoms."],
      ["adjudic", "When two parsers see the same value, pick the higher-confidence one."],
      ["replay", "Re-verify atoms by re-fetching the source span — guards against hallucination."],
      ["conf", "Apply confidence floors. Below-threshold atoms get pruned or flagged."],
      ["prose", "Split long prose blocks into sentence-level atoms."],
      ["enrich", "LLM entity extraction — device · date · money · site · stakeholder names from text."],
      ["classify", "LLM atom-type classification — turn prose atoms into scope_item / risk / etc."],
      ["collapse", "Dedupe near-identical atoms across files."],
      ["dedup", "Semantic dedup — same fact stated differently."],
      ["calib", "Confidence recalibration after enrichment + classification."],
      ["resolve", "Entity resolution — \"OPTBOT HQ\" + \"Atlanta HQ\" + \"1200 Peachtree\" → ATL-HQ-01."],
      ["graph", "Build atom-entity-atom edges for cross-referencing."],
      ["packet", "Bundle related atoms into scope packets ready for SOW rendering."],
    ].map(([s, d], i) => new TableRow({
      children: [
        cell(s, { width: 2000, mono: true, bold: true, size: 18, fill: PAPER, color: ACCENT }),
        cell(d, { width: 7360, size: 19, fill: i % 2 === 0 ? SURF : PAPER }),
      ],
    })),
  }),

  pageBreak(),
  h3("Known-good test deals (for sanity-checking)"),
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [2400, 1600, 5360],
    rows: [
      ["DEAL", "EXPECTED", "WHEN TO USE"],
      ["OPTBOT / 841ea7e0", "433 / 155", "Real RFP — multi-site, multi-file. The reference deal."],
      ["Yonah Sapir / 1bf0c10e", "9 / 1", "Thin input — single notes PDF. Use to test thin-input handling."],
      ["AP swap qty 101 / 02557291", "307 / 72", "Medium RFP — DOCX + XLSX. Smoke test."],
    ].map((r, i) => new TableRow({
      children: r.map((c, ci) => cell(c, {
        width: [2400, 1600, 5360][ci],
        size: i === 0 ? 16 : 19,
        mono: ci <= 1,
        bold: i === 0,
        fill: i === 0 ? INK : SURF,
        color: i === 0 ? "FFFFFF" : INK,
      })),
    })),
  }),

  new Paragraph({ spacing: { before: 600 } }),
  rule(),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 240 },
    children: [new TextRun({ text: "PurPulse · OrbitBrief PM Field Guide v57.11", font: "Inter", size: 18, color: INK3 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Questions, bugs, requests → Slack #parser-os-feedback", font: "Inter", size: 18, color: INK3 })],
  }),
];

// ═════════════════════════════════════════════════════════════════
//  BUILD
// ═════════════════════════════════════════════════════════════════
const doc = new Document({
  creator: "Claude · PurPulse Eng",
  title: "OrbitBrief PM Field Guide",
  description: "How to read a parsed deal, work the OrbitBrief, and ship a clean SOW.",
  styles: {
    default: { document: { run: { font: "Inter", size: 22, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 44, bold: true, font: "Fraunces", color: INK },
        paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Inter", color: ACCENT },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Inter", color: INK },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 540, hanging: 360 } }, run: { font: "Inter", size: 22, color: INK } } },
        ],
      },
      { reference: "numbers",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 540, hanging: 360 } }, run: { font: "Inter", size: 22, color: INK, bold: true } } },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      headers: {
        default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "OrbitBrief · PM Field Guide", font: "JetBrains Mono", size: 16, color: INK3, characterSpacing: 30 })],
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE, space: 6 } },
        })] }),
      },
      footers: {
        default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "page ", font: "JetBrains Mono", size: 16, color: INK3 }),
            new TextRun({ children: [PageNumber.CURRENT], font: "JetBrains Mono", size: 16, color: INK3 }),
            new TextRun({ text: " of ", font: "JetBrains Mono", size: 16, color: INK3 }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], font: "JetBrains Mono", size: 16, color: INK3 }),
          ],
        })] }),
      },
      children: [
        ...cover,
        ...tocSection,
        ...section1,
        ...section2,
        ...section3,
        ...section4,
        ...section5,
        ...section6,
        ...appendix,
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  const out = "OrbitBrief_PM_Field_Guide.docx";
  fs.writeFileSync(out, buffer);
  console.log("✓ wrote " + out + " (" + buffer.length + " bytes)");
});
