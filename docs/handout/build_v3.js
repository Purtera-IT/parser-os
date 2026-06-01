/*
 * OrbitBrief PM Field Guide — v3
 *
 * Adds:
 *  • Section 4 — The 4 parser models (what each does, what to look for)
 *  • Section 5 — PM grading rubric (how to evaluate atoms / entities / sites / edges / packets)
 *  • Expanded Section 8 — Reporting a bug from inside the UI (FAB → menu → modal form)
 *
 * One real screenshot (pipeline timeline from the live OPTBOT page) +
 * high-fidelity UI mocks styled to the PurPulse design system.
 */

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Header, Footer, Table, TableRow, TableCell,
  AlignmentType, PageOrientation, LevelFormat, ExternalHyperlink, Bookmark,
  InternalHyperlink, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, ImageRun, TabStopType, TabStopPosition,
} = require("docx");

// ───────── PurPulse design tokens ─────────
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
const AMBER_BG = "F4E4C8";
const RED    = "B33A3A";
const RED_BG = "F4D8D8";
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
function h4(text) {
  return new Paragraph({
    children: [new TextRun({ text, color: ACCENT, bold: true, font: "JetBrains Mono", size: 20, characterSpacing: 40 })],
    spacing: { before: 240, after: 100 },
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
    borderColor = RULE, padTop = 100, padBottom = 100, italics = false } = opts;
  return new TableCell({
    borders: cellBorders(borderColor),
    width: { size: width, type: WidthType.DXA },
    shading: { fill, type: ShadingType.CLEAR },
    margins: margins(padTop, padBottom, 160, 160),
    children: Array.isArray(content) ? content : [
      new Paragraph({
        children: [new TextRun({
          text: String(content), font: mono ? "JetBrains Mono" : "Inter",
          size, bold, color: mute ? INK3 : color, italics,
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

// ═══════════════════════════════════════════════════════════════════
// MOCK UI BUILDERS
// ═══════════════════════════════════════════════════════════════════
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

// === NEW v3: parser model card ===
function parserCard(opts) {
  const { name, label, version, color, role, files, atomCount, mainTypes, lookFor, redFlags } = opts;
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [new TableCell({
      borders: {
        top: { style: BorderStyle.SINGLE, size: 24, color },
        bottom: border(RULE, 4), left: border(RULE, 4), right: border(RULE, 4),
      },
      width: { size: CONTENT_W, type: WidthType.DXA },
      shading: { fill: SURF, type: ShadingType.CLEAR },
      margins: margins(240, 240, 280, 280),
      children: [
        new Paragraph({
          spacing: { after: 60 },
          children: [
            new TextRun({ text: label.toUpperCase(), font: "JetBrains Mono", size: 18, color, bold: true, characterSpacing: 60 }),
            new TextRun({ text: "   " + version, font: "JetBrains Mono", size: 16, color: INK3 }),
          ],
        }),
        new Paragraph({
          spacing: { after: 100 },
          children: [new TextRun({ text: name, font: "Fraunces", size: 32, color: INK, bold: true })],
        }),
        new Paragraph({
          spacing: { after: 160 },
          children: [new TextRun({ text: role, font: "Inter", size: 22, color: INK2, italics: true })],
        }),
        new Paragraph({
          spacing: { before: 80, after: 60 },
          children: [new TextRun({ text: "WHEN IT RUNS", font: "JetBrains Mono", size: 14, color: INK3, bold: true, characterSpacing: 40 })],
        }),
        new Paragraph({
          spacing: { after: 140 },
          children: [new TextRun({ text: files, font: "Inter", size: 20, color: INK })],
        }),
        new Paragraph({
          spacing: { after: 60 },
          children: [new TextRun({ text: "OPTBOT YIELD", font: "JetBrains Mono", size: 14, color: INK3, bold: true, characterSpacing: 40 })],
        }),
        new Paragraph({
          spacing: { after: 140 },
          children: [
            new TextRun({ text: atomCount, font: "Fraunces", size: 30, color, bold: true }),
            new TextRun({ text: "   " + mainTypes, font: "Inter", size: 18, color: INK2 }),
          ],
        }),
        new Paragraph({
          spacing: { after: 60 },
          children: [new TextRun({ text: "WHAT TO LOOK FOR", font: "JetBrains Mono", size: 14, color: GREEN, bold: true, characterSpacing: 40 })],
        }),
        ...lookFor.map(t => new Paragraph({
          spacing: { after: 50 },
          children: [
            new TextRun({ text: "  ✓  ", font: "JetBrains Mono", size: 18, color: GREEN, bold: true }),
            new TextRun({ text: t, font: "Inter", size: 20, color: INK }),
          ],
        })),
        new Paragraph({
          spacing: { before: 140, after: 60 },
          children: [new TextRun({ text: "RED FLAGS", font: "JetBrains Mono", size: 14, color: RED, bold: true, characterSpacing: 40 })],
        }),
        ...redFlags.map(t => new Paragraph({
          spacing: { after: 50 },
          children: [
            new TextRun({ text: "  ⚠  ", font: "JetBrains Mono", size: 18, color: RED, bold: true }),
            new TextRun({ text: t, font: "Inter", size: 20, color: INK }),
          ],
        })),
      ],
    })] })],
  });
}

// === NEW v3: Help-and-Feedback FAB mock ===
function mockHelpFAB() {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [new TableCell({
      borders: cellBorders(RULES),
      shading: { fill: SURF2, type: ShadingType.CLEAR },
      margins: margins(420, 100, 200, 200),
      width: { size: CONTENT_W, type: WidthType.DXA },
      children: [
        new Paragraph({
          spacing: { after: 60 },
          children: [new TextRun({ text: "(deal page content)", font: "Inter", size: 18, color: INK3, italics: true })],
        }),
        new Paragraph({
          spacing: { after: 380 },
          children: [new TextRun({ text: "(scrolls)", font: "Inter", size: 18, color: INK3, italics: true })],
        }),
        new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [
            new TextRun({ text: "  ?  ", font: "JetBrains Mono", size: 36, color: SURF, bold: true,
              shading: { fill: ACCENT, type: ShadingType.CLEAR } }),
          ],
        }),
        new Paragraph({
          alignment: AlignmentType.RIGHT,
          spacing: { before: 40 },
          children: [new TextRun({ text: "Help and feedback", font: "Inter", size: 16, color: INK3, italics: true })],
        }),
      ],
    })] })],
  });
}

// === NEW v3: Help popover menu mock ===
function mockHelpMenu() {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [3000, 6360],
    rows: [new TableRow({ children: [
      cell("", { width: 3000, fill: SURF2, borderColor: SURF2 }),
      new TableCell({
        borders: cellBorders(RULES),
        width: { size: 6360, type: WidthType.DXA },
        shading: { fill: SURF, type: ShadingType.CLEAR },
        margins: margins(200, 200, 260, 260),
        children: [
          new Paragraph({
            spacing: { after: 60 },
            children: [new TextRun({ text: "HELP & FEEDBACK", font: "JetBrains Mono", size: 16, color: INK3, bold: true, characterSpacing: 50 })],
          }),
          new Paragraph({
            spacing: { after: 180 },
            children: [
              new TextRun({ text: "Context: ", font: "Inter", size: 18, color: INK3 }),
              new TextRun({ text: "pm · quoting.detail · OPTBOT", font: "JetBrains Mono", size: 18, color: INK }),
            ],
          }),
          new Paragraph({
            spacing: { after: 80 },
            children: [
              new TextRun({ text: " ℹ ", font: "JetBrains Mono", size: 22, color: INK, bold: true,
                shading: { fill: SURF2, type: ShadingType.CLEAR } }),
              new TextRun({ text: "    ", font: "Inter", size: 22 }),
              new TextRun({ text: "Help on this page", font: "Inter", size: 22, color: INK, bold: true }),
            ],
          }),
          new Paragraph({
            spacing: { after: 180 },
            children: [new TextRun({ text: "        Tour the active tab. What each panel is for.", font: "Inter", size: 19, color: INK3 })],
          }),
          new Paragraph({
            spacing: { after: 80 },
            children: [
              new TextRun({ text: " ⚠ ", font: "JetBrains Mono", size: 22, color: ACCENT, bold: true,
                shading: { fill: ACCENT_BG, type: ShadingType.CLEAR } }),
              new TextRun({ text: "    ", font: "Inter", size: 22 }),
              new TextRun({ text: "Report a bug", font: "Inter", size: 22, color: INK, bold: true }),
            ],
          }),
          new Paragraph({
            spacing: { after: 80 },
            children: [new TextRun({ text: "        Opens a form. Page context attaches automatically.", font: "Inter", size: 19, color: INK3 })],
          }),
        ],
      }),
    ]})],
  });
}

// === NEW v3: Bug report modal form mock ===
function mockBugForm() {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [new TableCell({
      borders: cellBorders(RULES),
      width: { size: CONTENT_W, type: WidthType.DXA },
      shading: { fill: SURF, type: ShadingType.CLEAR },
      margins: margins(280, 280, 320, 320),
      children: [
        new Paragraph({
          spacing: { after: 60 },
          children: [new TextRun({ text: "REPORT A BUG", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 60 })],
        }),
        new Paragraph({
          spacing: { after: 100 },
          children: [
            new TextRun({ text: "Reporting from Deal ", font: "Fraunces", size: 26, color: INK, bold: true }),
            new TextRun({ text: "quoting · pm.quoting.detail", font: "JetBrains Mono", size: 22, color: INK2 }),
          ],
        }),
        new Paragraph({
          spacing: { after: 240 },
          children: [new TextRun({ text: "Share what went wrong. Devs see this within minutes.", font: "Inter", size: 20, color: INK3, italics: true })],
        }),

        // CATEGORY row
        new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "CATEGORY", font: "JetBrains Mono", size: 14, color: INK3, bold: true, characterSpacing: 40 })] }),
        new Paragraph({
          spacing: { after: 200 },
          children: [
            new TextRun({ text: "  Bug  ", font: "JetBrains Mono", size: 20, color: SURF, bold: true,
              shading: { fill: ACCENT, type: ShadingType.CLEAR } }),
            new TextRun({ text: "     ", font: "Inter", size: 20 }),
            new TextRun({ text: "  Feature Request  ", font: "JetBrains Mono", size: 20, color: INK,
              shading: { fill: SURF2, type: ShadingType.CLEAR } }),
          ],
        }),

        // SEVERITY row
        new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "SEVERITY (OPTIONAL)", font: "JetBrains Mono", size: 14, color: INK3, bold: true, characterSpacing: 40 })] }),
        new Paragraph({
          spacing: { after: 240 },
          children: [
            new TextRun({ text: "  Critical  ", font: "JetBrains Mono", size: 18, color: RED, bold: true,
              shading: { fill: RED_BG, type: ShadingType.CLEAR } }),
            new TextRun({ text: "   ", font: "Inter", size: 18 }),
            new TextRun({ text: "  Major  ", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true,
              shading: { fill: AMBER_BG, type: ShadingType.CLEAR } }),
            new TextRun({ text: "   ", font: "Inter", size: 18 }),
            new TextRun({ text: "  Minor  ", font: "JetBrains Mono", size: 18, color: GREEN, bold: true,
              shading: { fill: GREEN_BG, type: ShadingType.CLEAR } }),
            new TextRun({ text: "   ", font: "Inter", size: 18 }),
            new TextRun({ text: "  Not specified  ", font: "JetBrains Mono", size: 18, color: INK3,
              shading: { fill: SURF2, type: ShadingType.CLEAR } }),
          ],
        }),

        // SUMMARY field
        new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "SUMMARY", font: "JetBrains Mono", size: 14, color: INK3, bold: true, characterSpacing: 40 })] }),
        new Paragraph({
          spacing: { after: 240 },
          children: [
            new TextRun({ text: "  Wrong atom_type on BOM line — labeled “scope_item”, should be “bom_line”                                  ",
              font: "Inter", size: 20, color: INK, italics: true,
              shading: { fill: PAPER, type: ShadingType.CLEAR } }),
          ],
        }),

        // WHAT HAPPENED textarea
        new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "WHAT HAPPENED?", font: "JetBrains Mono", size: 14, color: INK3, bold: true, characterSpacing: 40 })] }),
        new Paragraph({
          spacing: { after: 80 },
          children: [
            new TextRun({ text: "  In OPTBOT · 04_hardware_bill_of_materials.xlsx, row 14:                                                    ",
              font: "Inter", size: 20, color: INK,
              shading: { fill: PAPER, type: ShadingType.CLEAR } }),
          ],
        }),
        new Paragraph({
          spacing: { after: 80 },
          children: [
            new TextRun({ text: "  “LG 65UH5N-E commercial display” shows as atom_type=scope_item, conf 0.91.                          ",
              font: "Inter", size: 20, color: INK,
              shading: { fill: PAPER, type: ShadingType.CLEAR } }),
          ],
        }),
        new Paragraph({
          spacing: { after: 240 },
          children: [
            new TextRun({ text: "  Expected bom_line (quantity 5, vendor LG, unit price $1,250).                                                ",
              font: "Inter", size: 20, color: INK,
              shading: { fill: PAPER, type: ShadingType.CLEAR } }),
          ],
        }),

        // Auto-attach notice
        new Paragraph({
          spacing: { after: 240 },
          children: [
            new TextRun({ text: " ℹ ", font: "JetBrains Mono", size: 18, color: INK3, bold: true,
              shading: { fill: SURF2, type: ShadingType.CLEAR } }),
            new TextRun({ text: "  Page context (route, workspace, entity ids) is attached automatically. No passwords or customer PII, please.",
              font: "Inter", size: 18, color: INK3, italics: true }),
          ],
        }),

        // Buttons
        new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [
            new TextRun({ text: "  Cancel  ", font: "JetBrains Mono", size: 20, color: INK,
              shading: { fill: SURF2, type: ShadingType.CLEAR } }),
            new TextRun({ text: "    ", font: "Inter", size: 20 }),
            new TextRun({ text: "   Submit report   ", font: "JetBrains Mono", size: 20, color: SURF, bold: true,
              shading: { fill: ACCENT, type: ShadingType.CLEAR } }),
          ],
        }),
      ],
    })]})],
  });
}

// === NEW v3: grading scorecard tile ===
function gradeTile(letter, label, body, color, width) {
  return new TableCell({
    borders: cellBorders(RULE),
    width: { size: width, type: WidthType.DXA },
    shading: { fill: SURF, type: ShadingType.CLEAR },
    margins: margins(220, 220, 240, 240),
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 60 },
        children: [new TextRun({ text: letter, font: "Fraunces", size: 60, bold: true, color })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 100 },
        children: [new TextRun({ text: label.toUpperCase(), font: "JetBrains Mono", size: 16, color, bold: true, characterSpacing: 40 })],
      }),
      new Paragraph({
        children: [new TextRun({ text: body, font: "Inter", size: 18, color: INK })],
      }),
    ],
  });
}

// Real screenshot
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
      transformation: { width: 600, height: 50 },
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
    children: [new TextRun({ text: "How to read a parsed deal, grade the output,", font: "Fraunces", size: 28, color: INK2, italics: true })],
    spacing: { after: 60 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "and file a useful bug.", font: "Fraunces", size: 28, color: INK2, italics: true })],
    spacing: { after: 500 },
  }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [new TextRun({ text: "WALKTHROUGH DEAL", font: "JetBrains Mono", size: 18, color: INK3, characterSpacing: 80 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 280 },
    children: [new TextRun({ text: "OPTBOT  ·  Atlanta Office Refresh", font: "Fraunces", size: 40, color: ACCENT, bold: true })] }),
  mockStatTiles(),
  new Paragraph({ spacing: { before: 1200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "v57.14  ·  current as of 1 June 2026", font: "JetBrains Mono", size: 16, color: INK3 })] }),
  pageBreak(),
);

// ── TOC ────────────────────────────────────────────────────────
docChildren.push(
  h1("What's in this guide"),
  lead("Eight sections. Skim, or read end-to-end in 18 minutes. Every screen the PM will see, plus the parser models running underneath, plus how to grade output and report what's wrong."),
);
const tocRows = [
  ["1", "Finding deals", "The quoting list · filters · jumping into a deal"],
  ["2", "The deal page layout", "Header bar · 7-step workflow strip · 6 content tabs"],
  ["3", "Deal Artifacts — the parser pipeline", "Where 9 files become 433 typed atoms and 5 canonical sites"],
  ["4", "The 4 parser models", "orbitbrief_pdf_v3 · docx_parser_v1 · xlsx_parser_v2_1 · quote_parser_v1_4_1"],
  ["5", "PM grading rubric", "How to grade atoms, entities, sites, edges, packets — 5-letter scorecard"],
  ["6", "OrbitBrief — the LLM brief", "What the LLM makes of those atoms, what to check"],
  ["7", "SowSmith & SOW Versions", "Render the final Word doc, track revisions"],
  ["8", "Reporting a bug — from inside the UI", "The Help button · popover · modal form · severity ladder"],
];
docChildren.push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: [800, 2800, 5760],
  rows: tocRows.map(([n, title, desc]) => new TableRow({ children: [
    cell(n, { width: 800, fill: ACCENT, color: SURF, bold: true, align: AlignmentType.CENTER, size: 30, padTop: 180, padBottom: 180 }),
    cell(title, { width: 2800, bold: true, size: 22, fill: SURF }),
    cell(desc, { width: 5760, size: 20, mute: true, fill: SURF }),
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
  bullet("Filter by stage to see only \"Submitted for Quoting\" or \"Open — Awaiting Scope\"."),
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
  h1("Deal Artifacts — the parser pipeline"),
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

// ═══════════════════════════════════════════════════════════════════
// ── SECTION 4 — THE 4 PARSER MODELS (NEW IN v3) ─────────────────
// ═══════════════════════════════════════════════════════════════════
docChildren.push(
  new Paragraph({ children: [new TextRun({ text: "SECTION 4", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 80 })] }),
  h1("The 4 parser models"),
  lead("During the parse stage, parser-os routes each file to one of four specialized models. Each model has a job, an output signature, and a failure mode. Knowing which model produced which atoms is the difference between “file a bug” and “kick it back to the customer.”"),

  h3("Routing — which model sees which file"),
);
const routeRows = [
  ["FILE EXTENSION", "PRIMARY MODEL", "FALLBACK"],
  [".pdf (text-extractable)", "orbitbrief_pdf_v3", "—"],
  [".pdf (scanned / quotes)", "quote_parser_v1_4_1", "orbitbrief_pdf_v3"],
  [".docx", "docx_parser_v1", "—"],
  [".doc (legacy)", "docx_parser_v1", "(soffice convert first)"],
  [".xlsx / .xlsm", "xlsx_parser_v2_1", "—"],
  [".xls (legacy)", "xlsx_parser_v2_1", "(soffice convert first)"],
];
docChildren.push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: [3000, 3500, 2860],
  rows: routeRows.map((r, i) => new TableRow({ children: r.map((c, ci) => cell(c, {
    width: [3000, 3500, 2860][ci],
    size: i === 0 ? 14 : 19,
    mono: i > 0,
    bold: i === 0 || ci === 1,
    fill: i === 0 ? INK : (i % 2 === 0 ? SURF : SURF2),
    color: i === 0 ? SURF : INK,
  }))})),
}));

docChildren.push(
  caption("Discovery stage picks the model from file extension + magic bytes. A scanned-PDF quote routes to quote_parser, not the generic PDF model."),
  pageBreak(),
);

// MODEL 1 — orbitbrief_pdf_v3
docChildren.push(
  h2("Model 1 — orbitbrief_pdf_v3", INK),
  parserCard({
    name: "PDF text-and-layout parser",
    label: "orbitbrief_pdf_v3",
    version: "v3.2",
    color: ACCENT,
    role: "Pulls prose + tables out of text-extractable PDFs. Walks each page block-by-block, preserves heading hierarchy, captures table cells as raw_table_row, captures stand-alone sentences as raw prose atoms (later classified by the LLM enrich/classify stages).",
    files: "Notes PDFs · executive briefs · contract addenda · security packets · site rosters. Anything that's a PDF and isn't a printed quote/BOM scan.",
    atomCount: "90 atoms",
    mainTypes: "in OPTBOT across 5 PDF files — mostly scope_item, stakeholder, payment_term",
    lookFor: [
      "Headings preserved as bold-prefixed prose atoms (“Section 3: …”).",
      "Multi-column page layout split correctly (no left-column-then-right-column smash).",
      "Tables come through as raw_table_row, not as squashed prose.",
      "Page numbers visible in the right-pane atom inspector (“page 4”).",
      "Atom confidence ≥ 0.75 on body text; ≥ 0.85 on titles.",
    ],
    redFlags: [
      "Same paragraph appearing 2–3× in a row → page-header/footer leak (file a bug).",
      "All atoms tagged “prose” with no typed atoms after classify → LLM stage degraded.",
      "Tables flattened into one long string → layout-detection regression.",
      "Diacritics or smart quotes become mojibake (Â·) → encoding bug.",
      "Single-page PDF yields 0 atoms → PDF was actually scanned, route to quote_parser instead.",
    ],
  }),
  pageBreak(),
);

// MODEL 2 — docx_parser_v1
docChildren.push(
  h2("Model 2 — docx_parser_v1", INK),
  parserCard({
    name: "Word document parser",
    label: "docx_parser_v1",
    version: "v1.6",
    color: GREEN,
    role: "Walks the DOCX XML directly. Tables become raw_table_row (one atom per row). Body paragraphs become prose atoms. Headings get pStyle preserved so the LLM later knows what section they came from. Bullet/numbered lists are flattened into one atom per list item.",
    files: "Statements of work, site survey docs, customer questionnaires, scope addenda. Most DOCX from PurTera customers.",
    atomCount: "160 atoms",
    mainTypes: "in OPTBOT across 2 DOCX files — 67 raw_table_row, 41 scope_item, 18 site_allocation",
    lookFor: [
      "Each table row = exactly one raw_table_row atom (no row merging).",
      "Cell text inside a row preserved with “·” or “|” separators.",
      "Heading text appears in entity_keys (e.g. “section:site-survey”).",
      "Numbered lists keep their numbers in the atom text (“1. Provide rack space…”).",
      "Hyperlinked text retains visible label, not the URL.",
    ],
    redFlags: [
      "Multi-row merged-cell tables explode into many empty raw_table_row atoms.",
      "Embedded XLSX or PDF inside DOCX is silently dropped (filed as bug, severity Major).",
      "Track-changes content appears with the markup chars visible (“[del]…[ins]…”).",
      "Comments from the source DOCX leak into atoms as if they were body text.",
      "Footer/header text repeats on every page-equivalent block.",
    ],
  }),
  pageBreak(),
);

// MODEL 3 — xlsx_parser_v2_1
docChildren.push(
  h2("Model 3 — xlsx_parser_v2_1", INK),
  parserCard({
    name: "Excel workbook parser",
    label: "xlsx_parser_v2_1",
    version: "v2.1",
    color: ACCENT,
    role: "Reads every visible sheet, scans for header rows, then turns each data row into a raw_table_row atom. Detects merged-header BOMs (“Site → Item → Qty → Unit Price”) and tags them site_allocation. Date columns become milestone atoms. Money columns become commercial_total or vendor_line_item.",
    files: "BOMs, pricing schedules, cutover plans, site rosters with quantities.",
    atomCount: "183 atoms",
    mainTypes: "in OPTBOT across 2 XLSX files — 53 site_allocation, 31 raw_table_row, 15 task, 15 dependency",
    lookFor: [
      "Sheet name appears in entity_keys (“sheet:BOM”, “sheet:Schedule”).",
      "Site column on a BOM correctly resolves to canonical site_id (ATL-HQ-01, not “ATL HQ”).",
      "Quantity, unit price, line total all extracted as separate numeric fields.",
      "Date columns parsed as ISO dates (2026-06-15), not strings.",
      "Hidden sheets skipped unless they contain real data.",
    ],
    redFlags: [
      "Header row missed → every data row tagged with column letter (“A1: …, B1: …”) instead of column name.",
      "Merged cells across rows cause duplicate atoms with same content.",
      "Currency symbol stripped from money entity (“1250” instead of “$1,250”).",
      "Formula cells (=SUM(…)) appear with the formula text, not the value.",
      "Schedule with task + dependency columns missing the dependency edges.",
    ],
  }),
  pageBreak(),
);

// MODEL 4 — quote_parser_v1_4_1
docChildren.push(
  h2("Model 4 — quote_parser_v1_4_1", INK),
  parserCard({
    name: "Vendor-quote / BOM PDF parser",
    label: "quote_parser_v1_4_1",
    version: "v1.4.1",
    color: GREEN,
    role: "Specialized for printed/scanned vendor quotes and BOM PDFs. OCRs scanned tables, recovers part-number / qty / unit-price / extended-price columns, normalizes vendor names. Outputs bom_line and vendor_line_item atoms with structured numeric fields.",
    files: "Vendor quote PDFs (Synnex, TD Synnex, Ingram), printed BOMs, commercial pricing PDFs that were originally Excel.",
    atomCount: "34 atoms",
    mainTypes: "in OPTBOT on the commercial-pricing PDF — 16 scope_item, 4 payment_term, plus extracted bom_line in entity_keys",
    lookFor: [
      "Part number (“65UH5N-E”, “CRESTRON-DM-MD-4K”) preserved exactly, no dashes lost.",
      "Vendor name normalized (“TD Synnex” not “TDS” or “Td-Synnex”).",
      "Quantity, unit price, extended price all present as separate fields.",
      "Payment terms picked up as own atoms (“Net 30 from invoice”).",
      "Each line item carries page+line provenance in entity_keys.",
    ],
    redFlags: [
      "OCR confidence < 0.65 on numeric fields → don't trust the totals.",
      "Quantity off by 10× → decimal point misread, always file a bug.",
      "Vendor name fragmented into two atoms (“TD” + “Synnex”).",
      "Multi-page quote treated as separate quotes → totals will be wrong.",
      "Currency mixed (USD line items tagged EUR) → file as Critical, blocks pricing.",
    ],
  }),
  pageBreak(),
);

docChildren.push(
  calloutBox("Which model produced which atom?", [
    new Paragraph({ children: [new TextRun({
      text: "In the right-pane atom inspector, every atom has a `producer` field. Click any atom to see:",
      font: "Inter", size: 22, color: INK })] }),
    new Paragraph({ spacing: { before: 120 }, children: [new TextRun({
      text: "  producer: docx_parser_v1   file: 02_statement_of_work.docx   page: 3   conf: 0.88",
      font: "JetBrains Mono", size: 18, color: ACCENT })] }),
    new Paragraph({ spacing: { before: 120 }, children: [new TextRun({
      text: "When you file a bug, paste this line into the report. The dev jumps straight to the right code path.",
      font: "Inter", size: 22, color: INK, italics: true })] }),
  ], ACCENT, ACCENT_BG),
  pageBreak(),
);

// ═══════════════════════════════════════════════════════════════════
// ── SECTION 5 — PM GRADING RUBRIC (NEW IN v3) ────────────────────
// ═══════════════════════════════════════════════════════════════════
docChildren.push(
  new Paragraph({ children: [new TextRun({ text: "SECTION 5", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 80 })] }),
  h1("PM grading rubric"),
  lead("Every deal you grade makes the system smarter. Five dimensions, five letter grades. Spend 4 minutes per deal, file bugs on anything C or below, and the parser improves week over week."),

  h3("The 5-letter scorecard"),
);
const cw5 = Math.floor(CONTENT_W / 5);
docChildren.push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: Array(5).fill(cw5),
  rows: [new TableRow({ children: [
    gradeTile("A", "atoms", "Right type · right span · right confidence.", ACCENT, cw5),
    gradeTile("E", "entities", "Devices · sites · money · people resolved cleanly.", GREEN, cw5),
    gradeTile("S", "sites", "Same site across files → one canonical record.", ACCENT, cw5),
    gradeTile("X", "edges", "Cross-file links exist where they should.", GREEN, cw5),
    gradeTile("P", "packets", "Scope bundles ready for SOW — complete, not partial.", ACCENT, cw5),
  ]})],
}));
docChildren.push(caption("Grade each dimension A / B / C / D / F. Anything below B — file a bug. Use the in-UI Report a bug button (Section 8)."));

// A — Atoms grading
docChildren.push(
  h2("A — Grading the atoms", ACCENT),
  p("Click any file in the Deal Artifacts file list. Right pane shows every atom that file produced. Spot-check 5 random atoms per file."),
  h4("WHAT MAKES A GOOD ATOM"),
  bullet("atom_type is correct (bom_line for a BOM row, not generic scope_item)."),
  bullet("text quotes the source verbatim (paste into the source file — Ctrl-F should find it)."),
  bullet("confidence ≥ 0.75 for typed atoms, ≥ 0.65 for prose."),
  bullet("entity_keys include the right entities (site, vendor, device, person)."),
  bullet("page number / row number is accurate for navigation."),
  h4("LETTER GRADES"),
);
const aGrade = [
  ["A", "5/5 spot-checks correct — type, text, conf, entities all right."],
  ["B", "4/5 correct — one minor mistype or low-confidence-but-still-right."],
  ["C", "3/5 — noticeable wrong types or fuzzy spans. File a bug."],
  ["D", "2/5 — majority wrong. File a bug, Severity Major."],
  ["F", "0–1/5 — parser is broken on this file type. File a bug, Severity Critical."],
];
docChildren.push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: [800, 8560],
  rows: aGrade.map(([g, d], i) => new TableRow({ children: [
    cell(g, { width: 800, mono: true, bold: true, size: 28, fill: i === 0 ? GREEN_BG : (i === 4 ? RED_BG : PAPER), color: i === 0 ? GREEN : (i === 4 ? RED : ACCENT), align: AlignmentType.CENTER }),
    cell(d, { width: 8560, size: 20, fill: SURF }),
  ]})),
}));

docChildren.push(
  h4("EXAMPLE — OPTBOT bom row, line 14"),
  calloutBox("Good vs bad atom on the same source row", [
    new Paragraph({ children: [new TextRun({ text: "Source: row 14 of 04_hardware_bill_of_materials.xlsx", font: "JetBrains Mono", size: 18, color: INK3 })] }),
    new Paragraph({ spacing: { before: 100 }, children: [new TextRun({ text: "“LG 65UH5N-E commercial display · Qty 5 · $1,250 · ATL-HQ-01”", font: "Inter", size: 22, color: INK, italics: true })] }),
    new Paragraph({ spacing: { before: 200, after: 60 }, children: [new TextRun({ text: "GOOD (grade A):", font: "Inter", size: 20, color: GREEN, bold: true })] }),
    new Paragraph({ children: [new TextRun({ text: "  atom_type: bom_line  ·  conf 0.92", font: "JetBrains Mono", size: 18, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "  entity_keys: device:LG-65UH5N-E, site:ATL-HQ-01, money:1250.00, qty:5", font: "JetBrains Mono", size: 18, color: INK })] }),
    new Paragraph({ spacing: { before: 200, after: 60 }, children: [new TextRun({ text: "BAD (grade D, file bug):", font: "Inter", size: 20, color: RED, bold: true })] }),
    new Paragraph({ children: [new TextRun({ text: "  atom_type: scope_item  ·  conf 0.61", font: "JetBrains Mono", size: 18, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "  entity_keys: (empty)", font: "JetBrains Mono", size: 18, color: INK })] }),
    new Paragraph({ spacing: { before: 100 }, children: [new TextRun({ text: "Why bad: lost the qty, lost the price, lost the site, generic atom_type. SOW will price this wrong.",
      font: "Inter", size: 20, color: INK, italics: true })] }),
  ], ACCENT, ACCENT_BG),
  pageBreak(),
);

// E — Entities grading
docChildren.push(
  h2("E — Grading the entities", GREEN),
  p("Click the ENTITIES tile at the top of Deal Artifacts. Filter by kind: device, money, site, milestone, stakeholder, org. Spot-check 3 per kind."),
  h4("WHAT MAKES A GOOD ENTITY"),
  bullet("Canonical form: “LG 65UH5N-E” not “the LG TV”."),
  bullet("Mention count > 1 for entities that appear in multiple files (deduped properly)."),
  bullet("Money entities have currency code and parsed numeric value."),
  bullet("Date entities are ISO format (2026-06-15) not freeform (“mid-June”)."),
  bullet("Stakeholder entities have role attached (“Maria Chen, IT Director”)."),
  h4("RED FLAGS"),
  bullet("Phantom entities — a device:storage entity in a deal that has no storage scope. Always file as Critical."),
  bullet("Duplicate entities for the same thing (“LG 65UH5N” and “LG-65UH5N-E” both exist). File as Major."),
  bullet("Negated entities pulled positive (“NO firewall change required” → device:firewall). File as Major."),
  bullet("All money in one currency mixed (USD and EUR mingled with no currency tag). File as Critical."),
  pageBreak(),
);

// S — Sites grading
docChildren.push(
  h2("S — Grading the sites", ACCENT),
  p("Sites are the most important entity — they're the unit of pricing, scheduling, and SOW sectioning. Open the Sites & Ingest workflow step or hover the SITES count."),
  h4("WHAT MAKES A GOOD SITE RECORD"),
  bullet("Single canonical record — “OPTBOT HQ”, “Atlanta HQ”, “1200 Peachtree” all resolved to ATL-HQ-01."),
  bullet("Address parsed (street, city, state, zip in separate fields)."),
  bullet("Confidence ≥ 0.75 means parser is confident; below 0.70 flags for SA."),
  bullet("Site appears in the right files — a site mentioned in the BOM but not the schedule is suspicious."),
  bullet("Site count matches the deal's stated scope (“5-site refresh” → 5 site rows)."),
  h4("RED FLAGS"),
  bullet("Same physical site appears twice with different IDs → entity-resolve missed it. File as Major."),
  bullet("Site mentioned only in one file when it should appear in 3+ → cross-file linkage broken. File as Major."),
  bullet("Site address parsed wrong (state stuffed into city). File as Minor unless customer-facing."),
  bullet("“TBD” or “To be determined” shows up as a site → entity-resolve hallucinated. File as Major."),
  pageBreak(),
);

// X — Edges grading
docChildren.push(
  h2("X — Grading the edges", GREEN),
  p("Edges are the relationships parser-os builds: “this BOM line goes to this site”, “this task depends on that one”, “this stakeholder approves this scope.” Click the EDGES tile."),
  h4("WHAT MAKES GOOD EDGES"),
  bullet("Cross-file count is high. OPTBOT shows 127/182 cross-file — that's healthy (~70%)."),
  bullet("Site-allocation edges: every site has at least N BOM-line edges where N ≥ 5."),
  bullet("Schedule-dependency edges: every task with a precursor has a depends-on edge."),
  bullet("Stakeholder-approval edges: each approval_authority atom links to a person entity."),
  bullet("No orphan atoms — every typed atom should have at least one edge to something."),
  h4("RED FLAGS"),
  bullet("Cross-file edges = 0 on a multi-file deal → entity-resolve never linked anything. File as Critical."),
  bullet("Site allocation edges all point to one site even though BOM has 5 sites. File as Major."),
  bullet("Schedule tasks have no dependency edges even though source has “after …” wording. File as Major."),
  bullet("Single-file deals correctly show EDGES: 0 + “single-file scope” label — NOT a bug."),
  pageBreak(),
);

// P — Packets grading
docChildren.push(
  h2("P — Grading the packets", ACCENT),
  p("Packets are what SowSmith reads to render the SOW. A packet bundles together “one site + its atoms + its BOM lines + its schedule + its risks.” Click any packet to expand it."),
  h4("WHAT MAKES A GOOD PACKET"),
  bullet("Site-scoped: each packet has exactly one site_id."),
  bullet("BOM-complete: contains every BOM line that allocated to that site."),
  bullet("Schedule-complete: contains every task that mentions that site."),
  bullet("Risk-aware: includes site_access_window, site_access_restriction, blackout_date_range atoms if present."),
  bullet("Money rolls up: per-site total = sum of bom_line ext-prices for that site."),
  h4("RED FLAGS"),
  bullet("Packet count < site count → some sites have no scope bundled. File as Major."),
  bullet("Packet has BOM but no schedule → schedule never linked to sites. File as Major."),
  bullet("Packet total doesn't match Deal Kit pricing total. File as Critical."),
  bullet("Packet contains atoms from a different site (cross-contamination). File as Critical."),
  pageBreak(),
);

docChildren.push(
  h3("Grading workflow — 4 minutes per deal"),
);
const flowRows = [
  ["1", "Open Deal Artifacts", "30s", "Scan the 7-tile stat strip. Anything red?"],
  ["2", "Click the biggest file", "60s", "Spot-check 5 atoms in the right pane. Grade A."],
  ["3", "Click ENTITIES tile", "60s", "Filter by device, money, site. Grade E."],
  ["4", "Open the sites panel", "30s", "Confirm site count matches deal scope. Grade S."],
  ["5", "Click any packet", "30s", "Confirm it has BOM + schedule + risks. Grade P."],
  ["6", "File bugs", "30s", "Any C/D/F → click the Help button → Report a bug."],
];
docChildren.push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: [600, 2800, 800, 5160],
  rows: flowRows.map(([n, what, time, why], i) => new TableRow({ children: [
    cell(n, { width: 600, bold: true, size: 22, fill: ACCENT, color: SURF, align: AlignmentType.CENTER }),
    cell(what, { width: 2800, bold: true, size: 21, fill: SURF }),
    cell(time, { width: 800, mono: true, size: 18, fill: PAPER, color: INK3, align: AlignmentType.CENTER }),
    cell(why, { width: 5160, size: 19, fill: SURF }),
  ]})),
}));
docChildren.push(pageBreak());

// ── SECTION 6 — OrbitBrief (renumbered from 4) ────────────────
docChildren.push(
  new Paragraph({ children: [new TextRun({ text: "SECTION 6", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 80 })] }),
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

docChildren.push(
  h3("How to grade the brief"),
  bullet("PM status correct? — if you'd call it RED, brief says RED. If brief says GREEN on a deal missing pricing → bug."),
  bullet("Summary accurate? — dollar amount, site count, close date all right? If brief says “3-site” on a 5-site deal → bug."),
  bullet("Customer questions actionable? — each question should be answerable. Generic “clarify scope” → brief is weak, bug as Minor."),
  bullet("Gaps real? — a flagged gap that's actually covered in some file means LLM missed an atom. Bug as Major."),
  bullet("SA focus right? — if it says “start with West Campus” but the blocker is actually at HQ → bug."),

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

// ── SECTION 7 — SowSmith (renumbered) ─────────────────────────
docChildren.push(
  new Paragraph({ children: [new TextRun({ text: "SECTION 7", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 80 })] }),
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

// ═══════════════════════════════════════════════════════════════════
// ── SECTION 8 — REPORTING A BUG (EXPANDED, in-UI flow) ───────────
// ═══════════════════════════════════════════════════════════════════
docChildren.push(
  new Paragraph({ children: [new TextRun({ text: "SECTION 8", font: "JetBrains Mono", size: 18, color: ACCENT, bold: true, characterSpacing: 80 })] }),
  h1("Reporting a bug — from inside the UI"),
  lead("PurPulse ships an in-app bug-report tool. You never have to leave the deal page. Page context, route, workspace, deal id, compile id all attach automatically. Three clicks, 30 seconds, the dev sees it in Slack."),

  h3("Step 1 — Find the Help & feedback button"),
  p("Bottom-right corner of every PM page. Always visible. Burnt-orange circle with a “?” glyph:"),
  mockHelpFAB(),
  caption("The button stays pinned to the lower-right viewport. Doesn't move when you scroll."),
  pageBreak(),

  h3("Step 2 — Click it. Pick “Report a bug.”"),
  p("A popover slides in. It shows the page context up top (so devs know exactly which screen you were on) and two options:"),
  mockHelpMenu(),
  caption("“Help on this page” → a tour. “Report a bug” → the form on the next page."),
  pageBreak(),

  h3("Step 3 — Fill the form. Submit."),
  p("The form is intentionally tiny. The auto-attached context does the heavy lifting:"),
  mockBugForm(),
  caption("Real fields. Real auto-context notice. Real Submit button. This is the actual modal."),
  pageBreak(),

  h3("What each field is for"),
);
const fieldRows = [
  ["Category", "Bug = something is broken. Feature Request = something is missing. Pick Bug for grading-rubric issues."],
  ["Severity", "Critical / Major / Minor / Not specified. Match it to the grading rubric — see severity ladder below."],
  ["Summary", "One line. Goes straight into the GitHub issue title. Make it specific: “Wrong atom_type on BOM line 14 of OPTBOT”, not “parser is wrong.”"],
  ["What happened?", "Three to six lines. Source file + row/page + what you expected + what you saw. Paste the producer line from the atom inspector if it's an atom bug."],
  ["Auto-attached context", "Route (pm.quoting.detail), workspace, deal id, compile id, your user id, browser. Do NOT include passwords or customer PII — the form warns you."],
];
docChildren.push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: [2800, 6560],
  rows: fieldRows.map(([k, v]) => new TableRow({ children: [
    cell(k, { width: 2800, bold: true, size: 21, fill: PAPER }),
    cell(v, { width: 6560, size: 20, fill: SURF }),
  ]})),
}));
docChildren.push(pageBreak());

docChildren.push(
  h3("Severity ladder — what gets fixed when"),
);
const sevRows = [
  ["CRITICAL", "★★★ SAME DAY",
    "Crashes · stuck-at-queued > 5 min · phantom entities on a thin deal · wrong currency · packet total ≠ Deal Kit total · cross-site contamination."],
  ["MAJOR", "★★ THIS WEEK",
    "Wrong atom_type · wrong site resolution · missing cross-file edges on multi-file deal · OCR off by 10× · dup entities · brief says GREEN on missing-pricing deal."],
  ["MINOR", "★ NEXT SPRINT",
    "Cosmetic clipping · wording · 0-value tile labels · confidence slightly off · typo in extracted atom · address parsed slightly wrong but non-customer-facing."],
];
docChildren.push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: [1600, 1800, 5960],
  rows: sevRows.map(([sev, sla, what], i) => new TableRow({ children: [
    cell(sev, { width: 1600, mono: true, bold: true, size: 20, fill: i === 0 ? RED_BG : (i === 1 ? AMBER_BG : GREEN_BG), color: i === 0 ? RED : (i === 1 ? ACCENT : GREEN), align: AlignmentType.CENTER }),
    cell(sla, { width: 1800, mono: true, size: 18, fill: PAPER, color: INK3, align: AlignmentType.CENTER }),
    cell(what, { width: 5960, size: 19, fill: SURF }),
  ]})),
}));

docChildren.push(
  h3("Mapping grading rubric → severity"),
);
const mapRows = [
  ["Atoms grade D or F", "MAJOR or CRITICAL", "Parser broken on a file type → ship today/this week."],
  ["Entities phantom or negated", "CRITICAL", "Hallucination. Blocks scope. File immediately."],
  ["Sites duped or missing", "MAJOR", "Site is the unit of pricing. Wrong sites = wrong SOW."],
  ["Edges = 0 on multi-file", "CRITICAL", "Cross-file resolution dead → nothing rolls up."],
  ["Packet total ≠ Deal Kit total", "CRITICAL", "Money mismatch. Always Critical."],
  ["Brief says wrong status", "MAJOR", "PM trusts the brief. Wrong color → wrong action."],
  ["Cosmetic / wording / confidence", "MINOR", "Annoying but not blocking. Still file — it accumulates."],
];
docChildren.push(new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: [2800, 2000, 4560],
  rows: mapRows.map(([g, s, w], i) => new TableRow({ children: [
    cell(g, { width: 2800, size: 20, bold: true, fill: i % 2 === 0 ? SURF : SURF2 }),
    cell(s, { width: 2000, mono: true, bold: true, size: 18, fill: i % 2 === 0 ? SURF : SURF2,
      color: s.startsWith("CRITICAL") ? RED : (s.startsWith("MAJOR") ? ACCENT : GREEN), align: AlignmentType.CENTER }),
    cell(w, { width: 4560, size: 20, fill: i % 2 === 0 ? SURF : SURF2 }),
  ]})),
}));
docChildren.push(pageBreak());

docChildren.push(
  h3("Template you can paste — anatomy of a great report"),
  calloutBox("Paste this into “What happened?” — fill in the brackets", [
    new Paragraph({ children: [new TextRun({ text: "FILE:    [filename from Deal Artifacts list]", font: "JetBrains Mono", size: 20, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "ROW/PG:  [row 14 / page 3]", font: "JetBrains Mono", size: 20, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "PRODUCER:[paste from atom inspector — e.g. xlsx_parser_v2_1]", font: "JetBrains Mono", size: 20, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "EXPECTED:[what the atom/entity/site/edge should be]", font: "JetBrains Mono", size: 20, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "GOT:     [what the parser actually produced]", font: "JetBrains Mono", size: 20, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "IMPACT:  [what breaks downstream — SOW row missing, pricing wrong, etc]", font: "JetBrains Mono", size: 20, color: INK })] }),
    new Paragraph({ spacing: { before: 160 }, children: [new TextRun({
      text: "5 lines. The auto-attached context handles the rest. Dev jumps straight to the right model.",
      font: "Inter", size: 20, color: INK2, italics: true })] }),
  ], ACCENT, ACCENT_BG),

  h3("Self-service recovery (try before filing)"),
  num("Refresh the page (Ctrl+Shift+R / Cmd+Shift+R)."),
  num("Click Re-parse once more — transient errors auto-retry."),
  num("Open another deal — confirms whether it's deal-specific or system-wide."),
  num("Wait 90 seconds — first request after idle can take 30+ sec (cold start)."),

  calloutBox("Things that look like bugs but aren't", [
    new Paragraph({ children: [new TextRun({ text: "•  EDGES: 0 + “single-file scope”   →  one file → no inter-file relations. Not broken.",
      font: "Inter", size: 22, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "•  ENTITIES: 0 + “thin input”   →  source has no devices/sites/orgs to extract. Not broken.",
      font: "Inter", size: 22, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "•  Compile takes 8+ min on a real RFP   →  LLM classify is slow; normal.",
      font: "Inter", size: 22, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "•  “This deal is not available in PurPulse”   →  archived in HubSpot. Restore from HubSpot.",
      font: "Inter", size: 22, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "•  First request after lunch takes 30 sec   →  cold start, only happens after idle.",
      font: "Inter", size: 22, color: INK })] }),
  ], INK3, PAPER),
  pageBreak(),
);

// ── APPENDIX ──────────────────────────────────────────────────
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
  ["discover", "Open each file, identify type, route to the right parser model (Section 4)."],
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
  ["resolve", "Entity resolution — “OPTBOT HQ” + “Atlanta HQ” + “1200 Peachtree” → ATL-HQ-01."],
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
  h3("One-pager cheat sheet — grading + reporting"),
  calloutBox("The 30-second loop", [
    new Paragraph({ children: [new TextRun({ text: "1. Open deal. Scan stat strip. Anything red?", font: "Inter", size: 22, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "2. Click biggest file. Grade 5 atoms (A–F).", font: "Inter", size: 22, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "3. Click ENTITIES. Filter device + money. Sanity check.", font: "Inter", size: 22, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "4. Click sites. Count matches scope?", font: "Inter", size: 22, color: INK })] }),
    new Paragraph({ children: [new TextRun({ text: "5. Anything below B → burnt-orange ? → Report a bug → paste 5-line template.", font: "Inter", size: 22, color: INK })] }),
    new Paragraph({ spacing: { before: 160 }, children: [new TextRun({
      text: "Every bug filed = a regression test. Every regression test = a parser that gets stronger every week.",
      font: "Inter", size: 22, color: ACCENT, bold: true, italics: true })] }),
  ], GREEN, GREEN_BG),
);

docChildren.push(
  new Paragraph({ spacing: { before: 600 } }),
  new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: RULE, space: 1 } },
    children: [new TextRun("")],
  }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240 },
    children: [new TextRun({ text: "PurPulse · OrbitBrief PM Field Guide  ·  v57.14", font: "Inter", size: 18, color: INK3 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Bugs, requests, questions → burnt-orange ? in lower-right → Report a bug", font: "Inter", size: 18, color: INK3 })] }),
);

// ═══════════════════════════════════════════════════════════════════
// BUILD
// ═══════════════════════════════════════════════════════════════════
const doc = new Document({
  creator: "Claude · PurPulse Eng",
  title: "OrbitBrief PM Field Guide",
  description: "How to read a parsed deal, grade the output, and file a useful bug.",
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
