/*
 * OrbitBrief PM Field Guide — v2
 *
 * One real screenshot (pipeline timeline from the live OPTBOT page) +
 * professionally-mocked sections styled to the PurPulse design system.
 */

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Header, Footer, Table, TableRow, TableCell,
  AlignmentType, PageOrientation, LevelFormat, ExternalHyperlink, Bookmark,
  InternalHyperlink, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, ImageRun, TabStopType, TabStopPosition,
} = require("docx");

// ───────── PurPulse design tokens (matching the live UI) ─────────
const INK    = "1A1815";
const INK2   = "6B6557";
const INK3   = "9B9384";
const PAPER  = "FAF6EE";
const SURF   = "FFFFFF";
const SURF2  = "F5F1EA";
const ACCENT = "B8722C";
const ACCENT_BG = "F0E0CC";
const GREEN  = "2E7D5C";
const GREEN_BG = "DDEEE3";
const AMBER  = "B8722C";
const RED    = "B33A3A";
const RULE   = "E8E2D5";
const RULES  = "C8BFA8";

const CONTENT_W = 9360;

const border = (color = RULE, sz = 4) => ({ style: BorderStyle.SINGLE, size: sz, color });
const cellBorders = (c = RULE) => ({ top: border(c), bottom: border(c), left: border(c), right: border(c) });
const margins = (t = 100, b = 100, l = 140, r = 140) => ({ top: t, bottom: b, left: l, right: r });

function p(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, font: "Inter", size: 22, color: INK, ...opts })],
    spacing: { before: opts.before ?? 0, after: opts.after ?? 100 },
    alignment: opts.align,
  });
}
function lead(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: "Inter", size: 24, color: INK2, italics: false })],
    spacing: { after: 200 },
  });
}
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, color: INK, bold: true, font: "Fraunces", size: 52 })],
    spacing: { before: 400, after: 240 },
  });
}
function h2(text, color = ACCENT) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, color, bold: true, font: "Inter", size: 32 })],
    spacing: { before: 320, after: 160 },
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, color: INK, bold: true, font: "Inter", size: 26 })],
    spacing: { before: 280, after: 120 },
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
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, color: INK3, font: "Inter", size: 18, italics: true })],
    spacing: { after: 200 },
    alignment: AlignmentType.CENTER,
  });
}
function cell(content, opts = {}) {
  const { width, fill = SURF, color = INK, bold = false, size = 22,
    align = AlignmentType.LEFT, mono = false, mute = false,
    borderColor = RULE, padTop = 100, padBottom = 100 } = opts;
  return new TableCell({
    borders: cellBorders(borderColor),
    width: { size: width, type: WidthType.DXA },
    shading: { fill, type: ShadingType.CLEAR },
    margins: margins(padTop, padBottom, 160, 160),
    children: Array.isArray(content) ? content : [
      new Paragraph({
        children: [new TextRun({
          text: String(content), font: mono ? "JetBrains Mono" : "Inter",
          size, bold, color: mute ? INK3 : color,
        })],
        alignment: align,
      }),
    ],
  });
}
function statTile(label, value, sub, accent = INK, width) {
  return new TableCell({
    borders: cellBorders(RULE),
    width: { size: width, type: WidthType.DXA },
    shading: { fill: SURF, type: ShadingType.CLEAR },
    margins: margins(180, 200, 200, 200),
    children: [
      new Paragraph({
        spacing: { after: 40 },
        children: [new TextRun({ text: label.toUpperCase(), font: "JetBrains Mono", size: 16, color: INK3, bold: true, characterSpacing: 40 })],
      }),
      new Paragraph({
        spacing: { after: 40 },
        children: [new TextRun({ text: String(value), font: "Fraunces", size: 48, bold: true, color: accent })],
      }),
      new Paragraph({
        children: [new TextRun({ text: sub || "", font: "Inter", size: 16, color: INK3 })],
      }),
    ],
  });
}
function calloutBox(title, body, color = ACCENT, fill = ACCENT_BG) {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [new TableCell({
      borders: {
        top: { style: BorderStyle.SINGLE, size: 28, color },
        bottom: border(color, 6), left: border(color, 6), right: border(color, 6),
      },
      width: { size: CONTENT_W, type: WidthType.DXA },
      shading: { fill, type: ShadingType.CLEAR },
      margins: margins(220, 220, 260, 260),
      children: [
        new Paragraph({
          spacing: { after: 120 },
          children: [new TextRun({ text: title, font: "Inter", size: 24, bold: true, color })],
        }),
        ...(Array.isArray(body) ? body : [new Paragraph({
          children: [new TextRun({ text: body, font: "Inter", size: 22, color: INK })],
        })]),
      ],
    })] })],
  });
}
function pageBreak() { return new Paragraph({ children: [new PageBreak()] }); }

// === Mock the deal page header ===
function mockHeader() {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [3120, 3120, 3120],
    rows: [new TableRow({ children: [
      cell([
        new Paragraph({ children: [new TextRun({ text: "OPTBOT  ·  #000087", font: "JetBrains Mono", size: 18, color: INK3 })] }),
        new Paragraph({ spacing: { before: 40 }, children: [new TextRun({ text: "Atlanta Office Refresh", font: "Fraunces", size: 26, color: INK, bold: true })] }),
      ], { width: 3120, fill: PAPER, padTop: 160, padBottom: 160 }),
      cell([
        new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: " OPEN — AWAITING SCOPE ", font: "JetBrains Mono", size: 16, color: ACCENT, bold: true,
            shading: { fill: ACCENT_BG, type: ShadingType.CLEAR } })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80 },
          children: [new TextRun({ text: " SA UNASSIGNED ", font: "JetBrains Mono", size: 16, color: AMBER, bold: true,
            shading: { fill: ACCENT_BG, type: ShadingType.CLEAR } })] }),
      ], { width: 3120, fill: PAPER, padTop: 160, padBottom: 160 }),
      cell([
        new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Owner  Max Friedberger", font: "Inter", size: 20, color: INK })] }),
        new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { before: 40 }, children: [new TextRun({ text: "Close  May 31", font: "Inter", size: 18, color: INK3 })] }),
        new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { before: 80 },
          children: [new TextRun({ text: " SAVE QUOTE ", font: "JetBrains Mono", size: 18, color: SURF, bold: true,
            shading: { fill: INK, type: ShadingType.CLEAR } })] }),
      ], { width: 3120, fill: PAPER, padTop: 160, padBottom: 160 }),
    ]})],
  });
}

// === Mock the 7-step workflow strip ===
function mockWorkflow() {
  const cw = Math.floor(CONTENT_W / 7);
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: Array(7).fill(cw),
    rows: [new TableRow({ children: [
      cell([
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "1", font: "Fraunces", size: 30, bold: true, color: INK3 })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Overview", font: "Inter", size: 16, color: INK3 })] }),
      ], { width: cw, fill: SURF, padTop: 140, padBottom: 140 }),
      cell([
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "2", font: "Fraunces", size: 30, bold: true, color: SURF })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Scope sources", font: "Inter", size: 16, color: SURF, bold: true })] }),
      ], { width: cw, fill: ACCENT, padTop: 140, padBottom: 140 }),
      cell([
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "3", font: "Fraunces", size: 30, color: INK3 })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Project Needs", font: "Inter", size: 16, color: INK3 })] }),
      ], { width: cw, fill: SURF, padTop: 140, padBottom: 140 }),
      cell([
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "4", font: "Fraunces", size: 30, color: INK3 })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Sites & Ingest", font: "Inter", size: 16, color: INK3 })] }),
      ], { width: cw, fill: SURF, padTop: 140, padBottom: 140 }),
      cell([
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "5", font: "Fraunces", size: 30, color: INK3 })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Scope Review", font: "Inter", size: 16, color: INK3 })] }),
      ], { width: cw, fill: SURF, padTop: 140, padBottom: 140 }),
      cell([
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "6", font: "Fraunces", size: 30, color: INK3 })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Deal Kit", font: "Inter", size: 16, color: INK3 })] }),
      ], { width: cw, fill: SURF, padTop: 140, padBottom: 140 }),
      cell([
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "7", font: "Fraunces", size: 30, color: INK3 })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "SOW Ready", font: "Inter", size: 16, color: INK3 })] }),
      ], { width: cw, fill: SURF, padTop: 140, padBottom: 140 }),
    ]})],
  });
}

// === Mock the compile snapshot bar ===
function mockCompileBar() {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [new TableCell({
      borders: cellBorders(RULES),
      shading: { fill: SURF2, type: ShadingType.CLEAR },
      margins: margins(220, 220, 280, 280),
      width: { size: CONTENT_W, type: WidthType.DXA },
      children: [
        new Paragraph({ spacing: { after: 100 },
          children: [new TextRun({ text: "000087 — (Test) OPTBOT — Atlanta Office Refresh", font: "Fraunces", size: 28, bold: true, color: INK })] }),
        new Paragraph({ spacing: { after: 160 },
          children: [new TextRun({ text: "deal 841ea7e0   ·   compile 83e8ae1a-24b   ·   generated 5/29/2026, 2:20 PM",
            font: "JetBrains Mono", size: 18, color: INK3 })] }),
        new Paragraph({ children: [
          new TextRun({ text: "  ✓  Compile complete  ", font: "JetBrains Mono", size: 20, color: GREEN, bold: true,
            shading: { fill: GREEN_BG, type: ShadingType.CLEAR } }),
          new TextRun({ text: "                    ", font: "Inter", size: 20 }),
          new TextRun({ text: "   RE-PARSE   ", font: "JetBrains Mono", size: 20, color: SURF, bold: true,
            shading: { fill: INK, type: ShadingType.CLEAR } }),
        ]}),
      ],
    })]})],
  });
}

// === Mock the stat tile strip ===
function mockStatTiles() {
  const cw = Math.floor(CONTENT_W / 7);
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: Array(7).fill(cw),
    rows: [new TableRow({ children: [
      statTile("Files", 9, "5 pdf · 2 docx · 2 xlsx", INK, cw),
      statTile("Pages", 23, "total pages", INK, cw),
      statTile("Atoms", 433, "241 typed · 192 prose", GREEN, cw),
      statTile("Entities", 155, "device · money · stake.", INK, cw),
      statTile("Edges", 182, "127 cross-file", INK, cw),
      statTile("Packets", 31, "scope bundles", INK, cw),
      statTile("Parse", "OK", "no degraded", GREEN, cw),
    ]})],
  });
}

// === Mock the file table ===
function mockFileTable() {
  const rows = [
    ["FILE", "ATOMS", "STATUS", "TOP TYPES"],
    ["04_hardware_bill_of_materials.xlsx", "120", "ok", "30 site_allocation · 16 raw_table_row"],
    ["02_statement_of_work.docx", "84", "ok", "30 raw_table_row · 19 scope_item"],
    ["03_site_surveys_and_requirements.docx", "76", "ok", "37 raw_table_row · 22 scope_item"],
    ["05_project_schedule_and_cutover_plan.xlsx", "63", "ok", "15 task · 15 dependency"],
    ["09_commercial_pricing_..._final.pdf", "34", "ok", "16 scope_item · 4 payment_term"],
    ["01_deal_overview_executive_brief.pdf", "21", "ok", "11 scope_item · 6 stakeholder"],
    ["08_site_roster_and_facilities_auth.pdf", "17", "ok", "5 physical_site · 3 stakeholder"],
    ["07_contracting_procurement_packet.pdf", "10", "ok", "6 scope_item"],
    ["06_security_it_integration_notes.pdf", "8", "ok", "4 scope_item · 2 dependency"],
  ];
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [4800, 1000, 1100, 2460],
    rows: rows.map((r, i) => new TableRow({
      children: r.map((c, ci) => cell(c, {
        width: [4800, 1000, 1100, 2460][ci],
        size: i === 0 ? 14 : 18,
        bold: i === 0,
        mono: i > 0 && ci === 1,
        align: ci === 1 || ci === 2 ? AlignmentType.CENTER : AlignmentType.LEFT,
        fill: i === 0 ? INK : (i % 2 === 0 ? SURF : SURF2),
        color: i === 0 ? SURF : INK,
      })),
    })),
  });
}

// === Mock the sites readiness panel ===
function mockSitesTable() {
  const rows = [
    ["SITE ID", "NAME · ADDRESS", "CONF", "STATUS"],
    ["ATL-HQ-01", "OPTBOT Atlanta HQ · 1200 Peachtree St NE", "0.90", "high"],
    ["ATL-WEST-02", "OPTBOT West Campus · 3100 Interstate N Pkwy", "0.85", "high"],
    ["ATL-AIR-03", "OPTBOT Airport Logistics · 6000 N Terminal Pkwy", "0.90", "high"],
    ["ATL-047-04", "OPTBOT Brady Training · 047 Brady Ave NW", "0.90", "high"],
    ["ATL-CP-05", "OPTBOT College Park S · 1850 Sullivan Rd", "0.90", "high"],
  ];
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [1800, 4560, 1500, 1500],
    rows: rows.map((r, i) => new TableRow({
      children: r.map((c, ci) => cell(c, {
        width: [1800, 4560, 1500, 1500][ci],
        size: i === 0 ? 14 : 19,
        bold: i === 0 || ci === 0,
        mono: i === 0 || ci === 0 || ci === 2,
        align: i === 0 ? AlignmentType.LEFT : (ci >= 2 ? AlignmentType.CENTER : AlignmentType.LEFT),
        fill: i === 0 ? INK : (i % 2 === 0 ? SURF : SURF2),
        color: i === 0 ? SURF : (ci === 3 && i > 0 ? GREEN : INK),
      })),
    })),
  });
}

// === Insert the REAL pipeline screenshot ===
function realPipelineImage() {
  const imgPath = path.join(__dirname, "screenshots/optbot_pipeline.png");
  if (!fs.existsSync(imgPath)) return p("(pipeline screenshot missing)");
  const data = fs.readFileSync(imgPath);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 80 },
    children: [new ImageRun({
      type: "png",
      data,
      transformation: { width: 600, height: 50 },  // 1566:130 ratio = 12.05
      altText: {
        title: "Pipeline timeline",
        description: "Real screenshot from OPTBOT artifacts page showing the 14-stage parser-os pipeline with durations.",
        name: "PipelineScreenshot",
      },
    })],
  });
}

// ═══════════════════════════════════════════════════════════════════
// THE DOCUMENT
// ═══════════════════════════════════════════════════════════════════
const docChildren = [];

// ── COVER ──────────────────────────────────────────────────────
docChildren.push(
  new Paragraph({ spacing: { before: 1800 }, children: [new TextRun("")] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "PURPULSE · ORBITBRIEF", font: "JetBrains Mono", size: 22, color: ACCENT, bold: true, characterSpacing: 80 })],
    spacing: { after: 280 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "The PM Field Guide", font: "Fraunces", size: 96, color: INK, bold: true })],
    spacing: { after: 240 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "How to read a parsed deal, work the OrbitBrief,", font: "Fraunces", size: 28, color: INK2, italics: true })],
    spacing: { after: 60 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "and ship a clean SOW.", font: "Fraunces", size: 28, color: INK2, italics: true })],
    spacing: { after: 500 },
  }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [new TextRun({ text: "WALKTHROUGH DEAL", font: "JetBrains Mono", size: 18, color: INK3, characterSpacing: 80 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 280 },
    children: [new TextRun({ text: "OPTBOT  ·  Atlanta Office Refresh", font: "Fraunces", size: 40, color: ACCENT, bold: true })] }),
  mockStatTiles(),
  new Paragraph({ spacing: { before: 1200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "v57.13  ·  current as of 1 June 2026", font: "JetBrains Mono", size: 16, color: INK3 })] }),
  pageBreak(),
);

// ── TOC ────────────────────────────────────────────────────────
docChildren.push(
  h1("What's in this guide"),
  lead("Six sections. Skim, or read end-to-end in 12 minutes. Every screen the PM will see, explained, with OPTBOT as the running example."),
);
const tocRows = [
  ["1", "Finding deals", "The quoting list · filters · jumping into a deal"],
  ["2", "The deal page layout", "Header bar · 7-step workflow strip · 6 content tabs"],
  ["3", "Deal Artifacts — the parser", "Where 9 files become 433 typed atoms and 5 canonical sites"],
  ["4", "OrbitBrief — the brief", "What the LLM makes of those atoms"],
  ["5", "SowSmith & SOW Versions", "Render the final Word doc, track revisions"],
  ["6", "How to report a bug", "Template, severity ladder, and what's not actually a bug"],
];
docChildren.push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: [800, 2400, 6160],
  rows: tocRows.map(([n, title, desc]) => new TableRow({ children: [
    cell(n, { width: 800, fill: ACCENT, color: SURF, bold: true, align: AlignmentType.CENTER, size: 30, padTop: 180, padBottom: 180 }),
    cell(title, { width: 2400, bold: true, size: 22, fill: SURF }),
    cell(desc, { width: 6160, size: 20, mute: true, fill: SURF }),
  ]})),
}));
docChildren.push(pageBreak());

// ── SECTION 1 ─────────────────────────────────────────────────
docChildren.push(
  new Paragraph({ children: [new TextRun({ text: "SECTION 1", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 80 })] }),
  h1("Finding deals"),
  lead("Every deal lives under the Quoting tab in the top nav. The list is filterable by stage, owner, SA assignment, account, and pipeline."),
  h3("Top nav (always visible)"),
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [new TableCell({
      borders: cellBorders(RULES),
      shading: { fill: INK, type: ShadingType.CLEAR },
      margins: margins(220, 220, 280, 280),
      width: { size: CONTENT_W, type: WidthType.DXA },
      children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
        new TextRun({ text: "PurPulse   ", font: "Fraunces", size: 26, color: SURF, bold: true }),
        new TextRun({ text: "│   Dashboard   │   Accounts   │   ", font: "Inter", size: 22, color: "C9C2B1" }),
        new TextRun({ text: "Quoting", font: "Inter", size: 22, color: ACCENT, bold: true }),
        new TextRun({ text: "   │   Projects   │   Planning   │   Execution   │   Closeout", font: "Inter", size: 22, color: "C9C2B1" }),
      ]})],
    })]})],
  }),
  caption("Click Quoting → land on the full deal list. ⌘K opens global search to jump by name or HubSpot number."),
  h3("The quoting list"),
  p("Each card shows the HubSpot stage, days open, amount, and a colored bar indicating SA assignment status — orange means unassigned, neutral means assigned."),
  bullet("Filter by stage to see only \"Submitted for Quoting\" or \"Open – Awaiting Scope\"."),
  bullet("Filter by owner to see only your deals."),
  bullet("Filter by SA to see deals waiting on SA pickup."),
  bullet("Click any card → opens the deal page (defaults to Project Overview)."),
  bullet("Or paste the URL directly: …/pm/quoting/{dealId}?step=artifacts"),
  calloutBox("Pro tip — the URL is the source of truth", [
    new Paragraph({ children: [new TextRun({ text: "Every tab change updates the URL. Bookmark a tab, share a link, hit browser-back — all of it works.",
      font: "Inter", size: 22, color: INK })] }),
    new Paragraph({ spacing: { before: 120 }, children: [new TextRun({
      text: "  …/pm/quoting/841ea7e0-…/?step=artifacts", font: "JetBrains Mono", size: 20, color: ACCENT })] }),
  ]),
  pageBreak(),
);

// ── SECTION 2 ─────────────────────────────────────────────────
docChildren.push(
  new Paragraph({ children: [new TextRun({ text: "SECTION 2", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 80 })] }),
  h1("The deal page layout"),
  lead("Three persistent rails — header bar, workflow strip, content tabs. Everything else swaps based on which tab you're in."),
  h3("1 │ Header bar"),
  mockHeader(),
  caption("Deal name on the left · stage + SA pills in the middle · owner + close + SAVE on the right."),
  h3("2 │ Workflow strip (7 steps)"),
  mockWorkflow(),
  caption("Active step is highlighted in burnt-orange. You can jump anywhere — the URL updates with ?step=…"),
  h3("3 │ Content tabs"),
);
const tabRows = [
  ["Project Overview", "HubSpot deal metadata · stage timing · owner · SA · contacts · checklists"],
  ["Deal Artifacts", "★ Where the parser output lives. Files → 433 atoms → 155 entities → 5 sites"],
  ["OrbitBrief", "The LLM's brief: PM handoff, gaps, customer questions, SA focus"],
  ["Deal Kit", "BOM, pricing, margins, scenarios"],
  ["SowSmith", "Render the final SOW Word doc from the envelope"],
  ["SOW Versions", "Compare and promote SOW drafts"],
];
docChildren.push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: [2400, 6960],
  rows: tabRows.map(([title, desc]) => new TableRow({ children: [
    cell(title, { width: 2400, bold: true, size: 22, fill: title === "Deal Artifacts" ? ACCENT : SURF, color: title === "Deal Artifacts" ? SURF : INK }),
    cell(desc, { width: 6960, size: 20, fill: SURF }),
  ]})),
}));
docChildren.push(pageBreak());

// ── SECTION 3 ─────────────────────────────────────────────────
docChildren.push(
  new Paragraph({ children: [new TextRun({ text: "SECTION 3", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 80 })] }),
  h1("Deal Artifacts — the parser"),
  lead("Drop files in, click Re-parse, get structured atoms back. Everything downstream reads from here."),

  h3("OPTBOT at a glance"),
  mockStatTiles(),
  new Paragraph({ spacing: { after: 200 } }),

  h3("Compile status banner"),
  mockCompileBar(),
  caption("Single line — deal id, compile id, when generated, the green ✓ badge, and the RE-PARSE button right there."),

  pageBreak(),
  h2("The 14-stage parser pipeline", INK),
  p("Below the status banner, the live pipeline timeline shows every stage's duration. Each tile is one parser-os step. Real screenshot from OPTBOT below:"),
  realPipelineImage(),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
    children: [new TextRun({ text: "↑ actual UI · OPTBOT · 14 stages · 3 min 49 sec total · 5/29/2026", font: "JetBrains Mono", size: 16, color: INK3 })] }),
  caption("Two LLM-heavy stages (enrich 1m45s · classify 1m49s) dominate the wall time. Everything else runs in milliseconds."),

  pageBreak(),
  h3("The 7 numbers — what each one means"),
);
const statExpl = [
  ["FILES", "9", "Source documents the parser consumed (PDFs, DOCX, XLSX)."],
  ["PAGES", "23", "Total pages across all files — rough volume signal."],
  ["ATOMS", "433", "Smallest unit of meaning. \"241 typed · 192 prose\": typed = classified, prose = raw. Higher typed-ratio = richer extraction."],
  ["ENTITIES", "155", "Named things — devices, dates, money, sites, people. Hover hints them: 26 device · 18 money · 16 milestone · 11 stakeholder…"],
  ["EDGES", "182", "Relations between atoms and entities. 127 cross-file means parser tied facts across different files (BOM site A = schedule site A)."],
  ["PACKETS", "31", "Bundled scope units — \"this site + these atoms + this BOM line\" — ready for the SOW to render."],
  ["PARSE", "OK", "If anything failed, you'd see N issue in amber/red here."],
];
docChildren.push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: [2000, 1400, 5960],
  rows: statExpl.map(([label, val, desc]) => new TableRow({ children: [
    cell(label, { width: 2000, mono: true, bold: true, size: 18, fill: PAPER, color: INK3 }),
    cell(val, { width: 1400, mono: true, bold: true, size: 28, fill: PAPER, color: ACCENT, align: AlignmentType.CENTER }),
    cell(desc, { width: 5960, size: 20, fill: SURF }),
  ]})),
}));
docChildren.push(pageBreak());

docChildren.push(
  h3("Site readiness — 5 canonical sites for OPTBOT"),
  p("The parser reconciles the same site mentioned in different files (\"OPTBOT HQ\", \"Atlanta HQ\", \"1200 Peachtree\") into one canonical record with confidence:"),
  mockSitesTable(),
  caption("\"conf 0.90\" = 90% confident. Below 0.70 → flagged for SA review."),

  pageBreak(),
  h3("The 9 source files (OPTBOT)"),
  mockFileTable(),
  caption("Sorted by atom yield. Click any row → right pane shows that file's atoms with text, atom_type, confidence, entity_keys, source page."),
  pageBreak(),
);

// ── SECTION 4 ─────────────────────────────────────────────────
docChildren.push(
  new Paragraph({ children: [new TextRun({ text: "SECTION 4", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 80 })] }),
  h1("OrbitBrief — the LLM brief"),
  lead("Same source atoms, packaged for PM consumption: \"what's the deal · what's missing · what to ask · what should the SA focus on.\""),
);
const briefRows = [
  ["PM status", "Red / amber / green. Derived from gap-coverage, risk count, scope completeness. OPTBOT today: RED (missing CFO signoff packet)."],
  ["One-line summary", "Single headline: \"5-site Atlanta office refresh, ~$1.5M, May 31 close, awaiting CFO approval for procurement.\""],
  ["Domains", "Scope buckets this deal spans (audio_visual, network, low_voltage). OPTBOT: audio_visual + procurement."],
  ["Sites", "Same 5 canonical sites from Deal Artifacts, with per-site readiness scores."],
  ["Customer questions", "Auto-generated. \"Confirm the model # for the displays at College Park\" · \"Does West Campus need landlord approval first?\""],
  ["Gaps", "Required scope categories not yet covered. Each gap links back to the atoms that triggered it."],
  ["Facts by category", "Pricing / schedule / stakeholders / sites / risks — grouped digest of the atoms."],
  ["SA focus", "What the assigned SA should pick up first: highest-risk site, biggest pricing question, blocking dependency."],
];
docChildren.push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: [2800, 6560],
  rows: briefRows.map(([k, v]) => new TableRow({ children: [
    cell(k, { width: 2800, bold: true, size: 22, fill: PAPER }),
    cell(v, { width: 6560, size: 20, fill: SURF }),
  ]})),
}));
docChildren.push(pageBreak());

docChildren.push(
  h3("The three Re- buttons"),
);
const reRows = [
  ["Re-parse", "Only re-runs parser-os. Use when files changed but context didn't. ~7s on small deals, ~10 min on big ones."],
  ["Re-brief", "Skip parser-os, regenerate the LLM brief from existing envelope. ~90s. Use when the brief looks off."],
  ["Re-render SOW", "Skip parser + brief; re-render the SOW Word doc from the latest atoms. Sub-second once envelope exists."],
];
docChildren.push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: [2400, 6960],
  rows: reRows.map(([k, v]) => new TableRow({ children: [
    cell(k, { width: 2400, mono: true, bold: true, size: 22, fill: INK, color: SURF }),
    cell(v, { width: 6960, size: 20, fill: SURF }),
  ]})),
}));
docChildren.push(pageBreak());

// ── SECTION 5 ─────────────────────────────────────────────────
docChildren.push(
  new Paragraph({ children: [new TextRun({ text: "SECTION 5", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 80 })] }),
  h1("SowSmith & SOW Versions"),
  lead("Close the loop. SowSmith builds the SOW; SOW Versions tracks drafts."),
  h3("SowSmith"),
  num("Click the active SOW draft to preview it inline (full DOCX rendered in-browser)."),
  num("\"Re-render\" rebuilds the doc from the latest envelope. Use after Re-parse or Re-brief."),
  num("\"Promote to next version\" snapshots the current SOW into SOW Versions and bumps the version number."),
  num("\"Export DOCX\" downloads the Word file."),
  num("\"Open in Google Drive\" syncs to the customer-facing drive folder."),
  h3("SOW Versions"),
  p("Side-by-side compare of any two versions, with rollback highlights:"),
  bullet("Each version has: version number · created at · who promoted it · changed atom count vs prior."),
  bullet("Click \"Diff vs v1\" to see what changed."),
  bullet("Click \"Restore\" to make any old version the active draft."),
  calloutBox("How the three flow together", [
    new Paragraph({ children: [new TextRun({ text: "Deal Artifacts → atoms.  OrbitBrief → brief.  SowSmith → SOW.",
      font: "Inter", size: 22, color: INK, bold: true })] }),
    new Paragraph({ spacing: { before: 120 }, children: [new TextRun({
      text: "All three read from the SAME envelope.json. Edit upstream (Re-parse), and SowSmith picks it up on Re-render. Never edit a SOW directly — change the source.",
      font: "Inter", size: 22, color: INK, italics: true })] }),
  ], GREEN, GREEN_BG),
  pageBreak(),
);

// ── SECTION 6 ─────────────────────────────────────────────────
docChildren.push(
  new Paragraph({ children: [new TextRun({ text: "SECTION 6", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 80 })] }),
  h1("How to report a bug"),
  lead("Good reports get fixed same-day. Bad reports become \"could you screenshot it?\" delays. Use this template."),
  h3("The bug report template"),
);
const bugRows = [
  ["What you were doing", "One sentence. \"Clicked Re-parse on OPTBOT after re-uploading the BOM.\""],
  ["What you expected", "One sentence. \"Pipeline should rerun, show 9 files parsed.\""],
  ["What you saw instead", "One sentence. \"Pipeline stuck at 0/14 stages for 3 minutes.\""],
  ["URL", "Copy from browser. Includes deal id + step."],
  ["Deal ID", "Find it on the page: \"deal 841ea7e0 …\""],
  ["Compile ID", "Find it on the page: \"compile 83e8ae1a-24b …\""],
  ["Time it happened", "Approximate. \"~3:15 PM ET\". Worker logs are stamped — devs need a window."],
  ["Screenshot", "Required if visual. Include the full page chrome."],
  ["Console errors", "Open DevTools (F12) → Console → screenshot any red text."],
];
docChildren.push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: [3000, 6360],
  rows: bugRows.map(([k, v]) => new TableRow({ children: [
    cell(k, { width: 3000, bold: true, size: 20, fill: PAPER }),
    cell(v, { width: 6360, size: 20, fill: SURF }),
  ]})),
}));

docChildren.push(
  h3("Where to send it"),
  bullet("Slack #parser-os-feedback channel — quickest response."),
  bullet("GitHub issue at github.com/Purtera-IT/parser-os — for tracked work."),
  bullet("Or paste the template into the deal's Discussion tab — devs subscribe to deal-level mentions."),
  h3("Severity ladder — what gets fixed fastest"),
);
const sevRows = [
  ["★★★  SAME DAY", "Crashes · 404s · stuck-at-queued for >5 min · phantom entities · obviously-wrong extractions"],
  ["★★  THIS WEEK", "Missing atoms / wrong atom type · entity dupes · slow compiles · UI text clips"],
  ["★  NEXT SPRINT", "Cosmetic · wording · 0-value tile labels · new feature requests"],
];
docChildren.push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: [2400, 6960],
  rows: sevRows.map(([k, v]) => new TableRow({ children: [
    cell(k, { width: 2400, mono: true, bold: true, size: 18, fill: PAPER, color: ACCENT }),
    cell(v, { width: 6960, size: 20, fill: SURF }),
  ]})),
}));

docChildren.push(
  h3("Self-service recovery (try before filing)"),
  num("Refresh the page (Ctrl+Shift+R / Cmd+Shift+R)."),
  num("Click Re-parse once more — transient errors auto-retry."),
  num("Open another deal — confirms whether it's deal-specific or system-wide."),
  num("Wait 90 seconds — first request after idle can take 30+ sec (cold start)."),
  calloutBox("Things that look like bugs but aren't", [
    new Paragraph({ children: [new TextRun({ text: "•  EDGES: 0 + \"single-file scope\"   →  one file → no inter-file relations. Not broken.",
      font: "Inter", size: 22, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "•  ENTITIES: 0 + \"thin input\"   →  source has no devices/sites/orgs to extract. Not broken.",
      font: "Inter", size: 22, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "•  Compile takes 8+ min on a real RFP   →  LLM classify is slow; normal.",
      font: "Inter", size: 22, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "•  \"This deal is not available in PurPulse\"   →  archived in HubSpot. Restore from HubSpot.",
      font: "Inter", size: 22, color: INK })] }),
  ], INK3, PAPER),
  pageBreak(),
);

// ── APPENDIX ─────────────────────────────────────────────────
docChildren.push(
  new Paragraph({ children: [new TextRun({ text: "APPENDIX", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 80 })] }),
  h1("Quick reference"),
  h3("The 38 atom types"),
  p("Most-common first. Click any pill in the UI to filter the inspector:"),
  new Paragraph({
    children: [new TextRun({
      text: "raw_table_row · scope_item · site_allocation · vendor_line_item · task · dependency · quantity · milestone_phase · bom_line · lead_time_constraint · stakeholder · risk · site_access_window · acceptance_criterion · requirement · service_line · approval_authority · payment_term · integration_checkpoint · physical_site · mitigation · deal_metadata · constraint · commercial_total · site_attribute · site_budget · site_access_restriction · exclusion · cutover_step · decision · metadata_requirement · blackout_date_range · pricing_assumption · compliance_classification · open_question · site_room_mix · customer_instruction · assumption · electrical_acceptance_test",
      font: "JetBrains Mono", size: 16, color: INK,
    })],
    spacing: { after: 240 },
  }),
  h3("Pipeline stages explained"),
);
const stageRows = [
  ["discover", "Open each file, identify type, route to the right parser."],
  ["parse", "Per-file: PDF page extraction, DOCX table walking, XLSX cell scanning. Raw atoms."],
  ["adjudic", "When two parsers see the same value, pick the higher-confidence one."],
  ["replay", "Re-verify atoms by re-fetching the source span — guards against hallucination."],
  ["conf", "Apply confidence floors. Below-threshold atoms get pruned or flagged."],
  ["prose", "Split long prose blocks into sentence-level atoms."],
  ["enrich", "LLM entity extraction — device · date · money · site · stakeholder from text."],
  ["classify", "LLM atom-type classification — turn prose atoms into scope_item / risk / etc."],
  ["collapse", "Dedupe near-identical atoms across files."],
  ["dedup", "Semantic dedup — same fact stated differently."],
  ["calib", "Confidence recalibration after enrichment + classification."],
  ["resolve", "Entity resolution — \"OPTBOT HQ\" + \"Atlanta HQ\" + \"1200 Peachtree\" → ATL-HQ-01."],
  ["graph", "Build atom-entity-atom edges for cross-referencing."],
  ["packet", "Bundle related atoms into scope packets ready for SOW rendering."],
];
docChildren.push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: [2000, 7360],
  rows: stageRows.map(([s, d], i) => new TableRow({ children: [
    cell(s, { width: 2000, mono: true, bold: true, size: 18, fill: PAPER, color: ACCENT }),
    cell(d, { width: 7360, size: 19, fill: i % 2 === 0 ? SURF : SURF2 }),
  ]})),
}));

docChildren.push(
  pageBreak(),
  h3("Known-good test deals"),
);
const knownDeals = [
  ["DEAL", "EXPECTED", "USE FOR"],
  ["OPTBOT  ·  841ea7e0", "433 / 155", "Real RFP — multi-site, multi-file. Reference deal."],
  ["Yonah Sapir  ·  1bf0c10e", "9 / 1", "Thin input — single notes PDF. Tests thin-input handling."],
  ["AP swap qty 101  ·  02557291", "307 / 72", "Medium RFP — DOCX + XLSX. Smoke test."],
];
docChildren.push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: [2800, 1800, 4760],
  rows: knownDeals.map((r, i) => new TableRow({ children: r.map((c, ci) => cell(c, {
    width: [2800, 1800, 4760][ci], size: i === 0 ? 14 : 19, mono: ci <= 1,
    bold: i === 0, fill: i === 0 ? INK : SURF, color: i === 0 ? SURF : INK,
  }))})),
}));

docChildren.push(
  new Paragraph({ spacing: { before: 600 } }),
  new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: RULE, space: 1 } },
    children: [new TextRun("")],
  }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240 },
    children: [new TextRun({ text: "PurPulse · OrbitBrief PM Field Guide  ·  v57.13", font: "Inter", size: 18, color: INK3 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Questions, bugs, requests → Slack #parser-os-feedback", font: "Inter", size: 18, color: INK3 })] }),
);

// ═══════════════════════════════════════════════════════════════════
// BUILD
// ═══════════════════════════════════════════════════════════════════
const doc = new Document({
  creator: "Claude · PurPulse Eng",
  title: "OrbitBrief PM Field Guide",
  description: "How to read a parsed deal, work the OrbitBrief, ship a clean SOW.",
  styles: {
    default: { document: { run: { font: "Inter", size: 22, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 52, bold: true, font: "Fraunces", color: INK },
        paragraph: { spacing: { before: 400, after: 240 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Inter", color: ACCENT },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Inter", color: INK },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 560, hanging: 360 } }, run: { font: "Inter", size: 22, color: INK } } }] },
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 560, hanging: 360 } }, run: { font: "Inter", size: 22, color: INK, bold: true } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: "OrbitBrief  ·  PM Field Guide", font: "JetBrains Mono", size: 16, color: INK3, characterSpacing: 30 })],
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE, space: 6 } },
      })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({ text: "page ", font: "JetBrains Mono", size: 16, color: INK3 }),
          new TextRun({ children: [PageNumber.CURRENT], font: "JetBrains Mono", size: 16, color: INK3 }),
          new TextRun({ text: " · PurPulse OrbitBrief", font: "JetBrains Mono", size: 16, color: INK3 }),
        ],
      })] }),
    },
    children: docChildren,
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  const out = "OrbitBrief_PM_Field_Guide.docx";
  fs.writeFileSync(out, buffer);
  console.log("✓ wrote " + out + " (" + buffer.length + " bytes)");
});
