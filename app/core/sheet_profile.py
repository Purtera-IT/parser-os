"""A table is records. Atoms are statements. They are not the same thing.

Deal 010264 carries two inventory workbooks. One of them, on a laptop with no
network and no model of any kind, takes 136 seconds and produces **277,203
atoms**. The whole of 010180 -- a fully labelled deal -- is 249.

Every row of a 9,000-clock inventory became an atom, and once they exist every
later stage drags them along: entity enrichment, typed classification, the
packetizer. The deal cannot compile at all. It is not slow, it is impossible,
and no compile budget fixes it.

The category error is the point. A sentence somebody wrote is a CLAIM: it has
an author, an intent, and it can be agreed with or disputed, which is what
makes it labellable. A table row is a RECORD. Nobody asserted it; it was
exported. Shredding records into atoms produces a quarter of a million things
that look like claims, that nobody will ever review, and that drown the two
dozen facts on the sheet which actually decide the job.

So a sheet yields four things, and its rows are not among them:

    the TABLE      stored and queryable -- Atlas plans against it, RAG
                   answers from it, and a finding cites it. Nothing is lost.
    FINDINGS       the shape, and what each column's values say: three clock
                   models, 489 CRITICAL, 2,756 distinct cost centres.
    QUESTIONS      what the table forces somebody to decide. 1,317 of these
                   clocks are LEASED, which nobody wrote in an email and
                   which changes who owns the hardware being removed.
    CONFLICTS      a number here against a number in the prose. The SOW is
                   titled "9000 Clock Replacement"; the table says the 9000s
                   are 36% of the fleet.

None of this needs a Summary sheet, a keyword list or a customer name. The
distribution IS the summary, and every generator below is structural.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.column_roles import (
    ROLE_SAMPLE,
    Role,
    Supply,
    classify,
    contact_supply,
    site_supply,
)

#: At or below this many distinct values a column is a CATEGORY, and the
#: interesting thing about it is how the rows divide. Above it the column
#: identifies rows, and the interesting thing is how many there are.
#: Structural, not tuned: it separates "Status" from "Serial Number" on any
#: table in any domain.
CATEGORY_MAX_DISTINCT = 30

#: A category value held by this share or less of the rows is a minority worth
#: naming. 489 CRITICAL out of 4,928 is 10%: not an outlier to be hidden, a
#: population that needs different treatment.
MINORITY_SHARE = 0.45

#: Below this many rows a sheet is small enough that its rows may carry real
#: per-row meaning (a hand-written requirements matrix, a room schedule), so
#: it is left alone. Far above any authored table, well below any export.
BULK_ROWS = 200

#: Values that mean "this record is not in a good state", in the vocabularies
#: exports actually use. Matched whole-word and case-insensitively. This is a
#: judgement about ENGLISH, not about a customer: every inventory in every
#: domain marks its broken rows, and it marks them with these words.
_ADVERSE = re.compile(
    r"^(critical|error|fail(ed|ure)?|fault(y)?|down|offline|dead|missing|"
    r"unknown|not ?registered|unregistered|inactive|disabled|expired|"
    r"overdue|rma|decommission(ed)?|retired|unreachable|no ?comm\w*)$", re.I)

#: A column whose name says the row costs or is owed money.
_MONEY_NAME = re.compile(r"cost|price|amount|total|spend|value|rate|fee", re.I)

#: A column whose name says WHERE a row is. The count of distinct values is
#: the number of places a crew has to visit, which is what a rollout is
#: actually planned and priced against -- not the unit count.
_PLACE_NAME = re.compile(
    r"cost ?c(en|tr)|site|location|store|branch|facility|building|address|"
    r"region|district|premise", re.I)


#: A column whose name says the row's state rather than its price. "Cost
#: Center Status" contains the word cost and is neither a cost nor a place,
#: which is why the money and place rules below have to be told to stand
#: down before they read it as one.
_STATUS_NAME = re.compile(r"status|state|condition|flag|stage|phase", re.I)

#: A value that is a date or a time. An export stamps every row with when it
#: ran, so those columns hold few distinct values and look exactly like a
#: category -- but "476 clocks last polled at 02:19" is a fact about the
#: export, not about the job.
_TIMESTAMP = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"[A-Z][a-z]{2}\s+\d{1,2}\s+\d{4})[\sT]|^\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?$", re.I)

#: A name a spreadsheet tool made up because the cell was empty, or one this
#: module made up for the same reason. A finding about "Column17" tells a
#: reader nothing they can act on, because nobody knows what Column17 is.
_UNNAMED = re.compile(r"^(col|column|unnamed|field)[ _]?\d+$", re.I)

#: Blanks below this many rows are a typo somebody will fix, not a question
#: worth a PM's attention. Above it, a gap in a key column is a real hole in
#: the data the job is counted from.
MIN_GAP_ROWS = 25

#: A column filled on fewer than this share of rows is an annotation a few
#: people added, not a property of the population. Reporting "3 distinct
#: values" for a column with 3 values and 1,560 blanks reads as though it
#: divides the fleet three ways, which is the opposite of true.
MIN_COVERAGE = 0.10

#: How much of the smaller sheet's schema has to appear in the larger one
#: before they are views of the same population. Not 100%: whoever filtered
#: the sheet also pasted lookup columns beside it, so the filtered view had
#: 28 columns against the full export's 15.
SCHEMA_OVERLAP = 0.6

#: What a spreadsheet prints where a formula could not resolve. It is not
#: a value anybody entered and it is not a fact about the job -- reading it
#: as one produced the finding "Match Ser Num WFM: #N/A (634)", which says
#: only that a VLOOKUP missed. Same principle as withholding a reading from
#: bad OCR: the parser should not state what it cannot actually read.
_SPREADSHEET_ERROR = re.compile(
    r"^(#N/?A|#REF!?|#VALUE!?|#DIV/0!?|#NAME\??|#NULL!?|#NUM!?|#SPILL!?|"
    r"#CALC!?|#GETTING_DATA|N/?A|NULL|NONE|NAN)$", re.I)

#: How many distinct values are counted exactly before a column is simply
#: called "very many". It has to be far above any category AND far above
#: any address book: at 120 -- four times the category line -- a column of
#: 795 postcodes was called an identifier, which suppressed it, which took
#: the postcode out of the site address this whole module exists to build.
#: At 20,000 a column that overflows is an identifier by any reading, and
#: the exact count is still available everywhere it is needed.
DISTINCT_CAP = 20_000

#: Stands in for every value past the point where a column has clearly
#: stopped being a category. Counting a 115,000-row serial number column
#: value by value answers a question already settled.
_OVERFLOW = "<<overflow>>"


@dataclass
class Column:
    name: str
    distinct: int          # -1 == more than we bothered to count
    blanks: int
    top: list[tuple[str, int]]
    numeric_sum: float | None = None
    filled: int = 0
    #: Up to ROLE_SAMPLE of the column's values, kept so the role detector
    #: can see what the column actually holds. `top` is no substitute: it
    #: holds the eight COMMONEST values, and in an address column every
    #: value occurs once, so `top` is eight arbitrary streets.
    sample: list[str] = field(default_factory=list)

    @property
    def is_category(self) -> bool:
        return 0 < self.distinct <= CATEGORY_MAX_DISTINCT

    @property
    def is_identifier(self) -> bool:
        """One value per row, near enough: this column names rows."""
        return self.distinct < 0 or (
            self.filled > 0 and self.distinct >= self.filled * 0.9)


@dataclass
class SheetProfile:
    sheet: str
    rows: int
    columns: list[Column] = field(default_factory=list)
    headed: bool = True    # False when the top row is just more data

    def col(self, pattern: re.Pattern) -> Column | None:
        return next((c for c in self.columns if pattern.search(c.name or "")), None)


@dataclass
class Observation:
    """One thing the table says, or asks."""
    kind: str                 # "finding" | "question"
    key: str
    headline: str
    detail: str
    value: Any = None
    decides: str = ""


def profile_grid(
    sheet: str,
    rows: list[list[Any]],
    header_idx: int,
    *,
    headed: bool = True,
) -> SheetProfile:
    """One pass over the grid, positionally. Never materialises a row.

    The rows are kept once, by the artifact's structured document. Building
    a dict per row here to count values would mean holding a second copy of
    a 115,000-row export in memory to learn eleven numbers from it.
    """
    header = rows[header_idx] if (headed and 0 <= header_idx < len(rows)) else []
    names: list[str] = []
    used: dict[str, int] = {}
    start = header_idx if headed else max(header_idx, 0)
    width = max((len(r or []) for r in rows[start:start + 500]), default=0)
    for i in range(width):
        raw = str(header[i]).strip() if i < len(header) and header[i] is not None else ""
        name = raw or f"col_{i + 1}"
        if name in used:
            used[name] += 1
            name = f"{name}_{used[name]}"
        else:
            used[name] = 1
        names.append(name)

    counts: list[dict[str, int]] = [{} for _ in names]
    samples: list[list[str]] = [[] for _ in names]
    blanks = [0] * len(names)
    totals = [0.0] * len(names)
    numerics = [0] * len(names)
    n_rows = 0
    for r in rows[(header_idx + 1) if headed else max(header_idx, 0):]:
        r = r or []
        if not any(str(c or "").strip() for c in r):
            continue
        n_rows += 1
        for i in range(len(names)):
            v = r[i] if i < len(r) else None
            if v is None or str(v).strip() == "":
                blanks[i] += 1
                continue
            s_ = str(v).strip()[:60]
            if _SPREADSHEET_ERROR.match(s_):
                blanks[i] += 1      # a failed lookup is a gap, not a value
                continue
            if len(samples[i]) < ROLE_SAMPLE:
                samples[i].append(s_)
            c = counts[i]
            # A column that identifies rows has as many values as rows, and
            # counting them all buys nothing -- the answer is already "it is
            # not a category". Stop growing the map once it is past that line
            # and just keep counting, so a 115,000-row serial column costs a
            # bounded dict instead of 115,000 entries.
            if s_ in c:
                c[s_] += 1
            elif len(c) <= DISTINCT_CAP:
                c[s_] = 1
            else:
                c[_OVERFLOW] = c.get(_OVERFLOW, 0) + 1
            f = _as_number(s_)
            if f is not None:
                totals[i] += f
                numerics[i] += 1

    prof = SheetProfile(sheet=sheet, rows=n_rows, headed=headed)
    for i, name in enumerate(names):
        c = counts[i]
        overflowed = _OVERFLOW in c
        top = sorted(((k, v) for k, v in c.items() if k != _OVERFLOW),
                     key=lambda kv: -kv[1])[:8]
        filled = n_rows - blanks[i]
        prof.columns.append(Column(
            name=name,
            distinct=len(c) if not overflowed else -1,
            blanks=blanks[i],
            top=top,
            numeric_sum=totals[i] if numerics[i] and numerics[i] >= filled * 0.8 else None,
            filled=filled,
            sample=samples[i],
        ))
    return prof


def _usable(c: "Column") -> bool:
    """Can a reader act on a finding about this column at all?

    Two ways the answer is no. The column has no name -- a spreadsheet tool
    calls an empty header "Column17" and so does this module, and neither
    tells anybody what is in it. Or almost every row is blank: a column
    three people filled in is an annotation, and reporting it as "3 distinct
    values" reads as though it divides the fleet three ways.
    """
    if not (c.name or "").strip() or _UNNAMED.match(c.name.strip()):
        return False
    return c.filled > 0


def _is_money_column(c: "Column") -> bool:
    """A column of figures somebody is paying.

    The name has to say so AND the values have to bear it out. "Cost Center"
    is named like money and holds a 10-digit account code; summing it
    produced a $77 billion finding on a deal for 1,563 time clocks. So a
    money column must name money, must not name a place or a state, and must
    not be the column that identifies the rows.
    """
    if not _MONEY_NAME.search(c.name):
        return False
    if _PLACE_NAME.search(c.name) or _STATUS_NAME.search(c.name):
        return False
    return not c.is_identifier


def observe(prof: SheetProfile) -> list[Observation]:
    """What this table states, and what it asks. Structural throughout."""
    out: list[Observation] = []
    if prof.rows < 1:
        return out

    if not prof.headed:
        # Every rule below reads the column NAMES. On a sheet whose top row
        # is just more data, those names are values -- which is how a column
        # of postcodes came to be called "23231" and the PM was asked where
        # the 174 rows with no 23231 were. The honest reading of a headerless
        # export is its size and the fact that nobody can say what is in it.
        return [
            Observation(
                "finding", "table_shape",
                f"{prof.sheet}: {prof.rows:,} rows x {len(prof.columns)} columns, "
                f"no header row",
                "The sheet starts straight into data, so its columns are "
                "unnamed. The rows are stored and queryable either way; what "
                "cannot be done is say what any column means.",
                value=prof.rows),
            Observation(
                "question", "no_header",
                f"{prof.sheet} has {prof.rows:,} rows and no column headings - "
                f"what are its columns?",
                "A table nobody can read the columns of cannot be counted, "
                "filtered or reconciled against anything else in the deal. "
                "Either it was pasted in without its heading row, or it is a "
                "backing export that is not this job at all.",
                value=len(prof.columns),
                decides="Whether this sheet is deal content or somebody's "
                        "lookup table that came along with the workbook."),
        ]

    out.append(Observation(
        "finding", "table_shape",
        f"{prof.sheet}: {prof.rows:,} rows x {len(prof.columns)} columns",
        "The rows are stored and queryable; they are not atoms. A finding cites "
        "them, Atlas plans against them and a lookup answers from them.",
        value=prof.rows))

    for c in prof.columns:
        if not _usable(c):
            continue
        coverage = c.filled / max(prof.rows, 1)
        # A column of export timestamps holds few distinct values and is
        # shaped exactly like a category, but "476 clocks last polled at
        # 02:19" is a fact about when the report ran.
        stamped = bool(c.top) and sum(
            1 for v, _ in c.top if _TIMESTAMP.match(v)) >= max(1, len(c.top) // 2)

        if c.is_category and c.top and coverage >= MIN_COVERAGE and not stamped:
            over = (f"{c.filled:,} of {prof.rows:,} rows"
                    if c.blanks else f"{prof.rows:,} rows")
            if c.distinct == 1:
                # One value over every row is not a distribution, it is a
                # statement about the whole population -- which is often the
                # most useful thing the sheet says. "Device Type: InTouch
                # 9000" over all 1,563 rows is what makes this sheet the
                # 9000s, and reading it as "1 distinct value(s)" buries that.
                only = c.top[0][0]
                out.append(Observation(
                    "finding", f"constant:{c.name}",
                    f"Every row has {c.name} = {only}",
                    f"All {over} carry the same value, so this is a property of "
                    f"the whole table rather than a way it divides -- it says "
                    f"what this sheet IS.",
                    value=only))
            else:
                shown = ", ".join(f"{v} ({n:,})" for v, n in c.top[:5])
                out.append(Observation(
                    "finding", f"distribution:{c.name}",
                    f"{c.name}: {shown}",
                    f"{c.distinct} distinct value(s) over {over}. A column with "
                    f"few values divides the population, and how it divides is "
                    f"usually the job.",
                    value=dict(c.top)))

            # --- a minority that means "not in a good state" -------------
            for value, n in c.top:
                if not _ADVERSE.match(value.strip()):
                    continue
                share = n / max(c.filled, 1)
                if share > MINORITY_SHARE:
                    continue
                out.append(Observation(
                    "question", f"adverse:{c.name}:{value}",
                    f"{n:,} of {c.filled:,} rows are {value!r} - are they in scope?",
                    f"{c.name} marks {n:,} records ({share:.0%}) as {value!r}. A "
                    f"record in that state usually needs different handling from "
                    f"a healthy one, and often still needs a visit. Nothing in "
                    f"the deal's correspondence says whether these are included.",
                    value=n,
                    decides="Whether these units are in the count being priced, "
                            "and whether diagnosing them is our scope or the "
                            "customer's."))

        # --- a cost column holding a word instead of a figure -------------
        # An asset register that writes "Leased" where a price belongs is
        # stating an arrangement, not a price. It only means that if the
        # column really does hold figures elsewhere -- otherwise the column
        # was never about money and the name just contained the word.
        if _is_money_column(c) and c.top:
            words = [(v, n) for v, n in c.top if _as_number(v) is None]
            figures = [v for v, _ in c.top if _as_number(v) is not None]
            if words and figures:
                n = sum(n for _, n in words)
                named = ", ".join(v for v, _ in words[:3])
                out.append(Observation(
                    "question", f"ownership:{c.name}",
                    f"{n:,} rows carry {c.name} = {named!r} rather than a "
                    f"figure - who owns those?",
                    "A cost column holding a word instead of a number is saying "
                    "something about the arrangement, not the price. Ownership "
                    "decides who may dispose of the hardware being removed and "
                    "who is owed it back.",
                    value=n,
                    decides="Whether removed units are ours to scrap, or have to "
                            "be returned to whoever leased them."))

        # --- how many PLACES, which is what a rollout is priced on --------
        # Only the column that actually names places. "Cost Center Status"
        # contains the word centre and names a fiscal state, and reading it
        # as geography said a crew had eight places to reach.
        if (_PLACE_NAME.search(c.name) and not _STATUS_NAME.search(c.name)
                and c.distinct > 1 and not c.is_identifier
                and coverage >= MIN_COVERAGE):
            out.append(Observation(
                "finding", f"places:{c.name}",
                f"{c.distinct:,} distinct {c.name} across {c.filled:,} rows",
                "The number of places a crew has to reach. A rollout is planned "
                "and priced per visit, not per unit, so this is the figure that "
                "drives labour and travel.",
                value=c.distinct,
                decides="Truck rolls, travel and scheduling - usually a larger "
                        "cost than the hardware."))
            if c.blanks >= MIN_GAP_ROWS:
                out.append(Observation(
                    "question", f"unplaced:{c.name}",
                    f"{c.blanks:,} rows have no {c.name} - where are they?",
                    "A record with no location cannot be visited, counted into a "
                    "region, or scheduled.",
                    value=c.blanks,
                    decides="Whether these units exist on site, in a depot, or "
                            "not at all."))

        # --- the column that names the rows -------------------------------
        # An asset register's identifier column answers "how many things are
        # there", which stops matching "how many rows" once an export has
        # been appended to twice.
        if c.is_identifier and c.filled >= BULK_ROWS:
            if c.distinct > 0 and c.distinct < c.filled:
                dupes = c.filled - c.distinct
                out.append(Observation(
                    "question", f"duplicate_id:{c.name}",
                    f"{c.name} repeats: {c.filled:,} rows carry only "
                    f"{c.distinct:,} distinct values - is a unit listed twice?",
                    "An identifier that repeats means either the export was "
                    "appended to twice, or one physical unit has two records. "
                    f"The difference is {dupes:,} rows, and it lands straight on "
                    "the count being priced.",
                    value=dupes,
                    decides="The true unit count, and whether a site gets visited "
                            "twice for the same device."))
            if c.blanks >= MIN_GAP_ROWS:
                out.append(Observation(
                    "question", f"unidentified:{c.name}",
                    f"{c.blanks:,} of {prof.rows:,} rows have no {c.name} - are "
                    f"they real units?",
                    "A record with no identifier cannot be matched to a device on "
                    "site, tracked through a swap, or signed off.",
                    value=c.blanks,
                    decides="Whether these rows are units, blank filler, or "
                            "something the customer still has to fill in."))

        if c.numeric_sum and _is_money_column(c):
            out.append(Observation(
                "finding", f"total:{c.name}",
                f"{c.name} totals {c.numeric_sum:,.2f} over {c.filled:,} rows",
                "Summed from the column. Worth checking against any figure quoted "
                "in the deal's correspondence.",
                value=round(c.numeric_sum, 2)))
    return out


def subset_question(profiles: list[SheetProfile]) -> list[Observation]:
    """Two sheets of the same population, one much smaller: which is the job?

    An export beside a filtered view of itself is the commonest ambiguity a
    workbook carries, and no single sheet can see it. On this deal the
    in-scope list is 1,563 rows and the full fleet is 4,927 -- and the SOW
    is titled for the smaller one while the customer sent both.

    Requiring identical columns was too strict to catch that: the filtered
    sheet had 28 columns and the full one 15, because whoever filtered it
    also pasted lookups alongside. So the test is that most of the smaller
    sheet's columns appear in the larger one, which is what "a view of the
    same thing" means.
    """
    out: list[Observation] = []
    big = [p for p in profiles if p.headed and p.rows >= BULK_ROWS]
    for i, small in enumerate(big):
        cols_s = {c.name.strip().lower() for c in small.columns if c.name.strip()}
        if len(cols_s) < 4:
            continue
        for large in big:
            if large is small or large.rows < small.rows * 1.5:
                continue
            cols_l = {c.name.strip().lower() for c in large.columns if c.name.strip()}
            # Against the SMALLER schema, not the filtered sheet's. Whoever
            # filtered the 1,563 devices out of the 4,927 also pasted five
            # lookup columns beside them, so the filtered sheet has 28
            # columns to the export's 15 -- and all 15 of the export's are
            # among them. Dividing by 28 called that a 54% match and missed
            # the most valuable question on the deal.
            shared = len(cols_s & cols_l) / min(len(cols_s), len(cols_l))
            if shared < SCHEMA_OVERLAP:
                continue
            out.append(Observation(
                "question", f"subset_or_whole:{small.sheet}:{large.sheet}",
                f"{small.sheet} lists {small.rows:,} and {large.sheet} lists "
                f"{large.rows:,} of the same kind of record - which one is "
                f"the job?",
                f"The two sheets share {shared:.0%} of their columns, so one is "
                f"a view of the other's population. The deal has to say which "
                f"it is priced against; the difference is "
                f"{large.rows - small.rows:,} records.",
                value=[small.rows, large.rows],
                decides="The size of the job, and therefore the price."))
            break
    return out


def has_header(rows: list[list[Any]], header_idx: int, sample: int = 400) -> bool:
    """Is the top row a heading, or just the first record?

    A heading is not one of the values it heads. "Status" never appears in
    the Status column; "CONVENIENCE SOLUTIONS" appears in its column 43,976
    times, because it is not a heading -- it is the first of 114,956
    records, and the sheet was pasted in without its heading row.

    That is the whole test, and it holds for any table from any tool in any
    language. The obvious alternative -- compare the top row to the body on
    whether cells are numeric -- does not work: most columns are text over
    text, so a genuine header "agrees" with its body just as often as a data
    row does. It read Not Registered, a sheet with a perfectly good header,
    as headerless.
    """
    if header_idx < 0 or header_idx >= len(rows):
        return False
    head = rows[header_idx] or []
    body = [r or [] for r in rows[header_idx + 1: header_idx + 1 + sample]]
    body = [r for r in body if any(str(c or "").strip() for c in r)]
    if len(body) < 10:
        return True     # too little to judge; trust the row
    checked = in_body = 0
    for i, cell in enumerate(head):
        h = str(cell or "").strip()
        if not h:
            continue
        col = {str(r[i]).strip() for r in body
               if i < len(r) and str(r[i] or "").strip()}
        if len(col) < 2:
            continue    # a constant column tells us nothing either way
        checked += 1
        if h in col:
            in_body += 1
    if checked < 3:
        return True
    # One coincidence is a column that happens to contain its own name
    # ("REMOVE" over a column of REMOVE). Two or more is a data row.
    return in_body < 2


def _as_number(v: str) -> float | None:
    """Read a figure, in the formats a spreadsheet actually stores them in.

    An asset register exported from a European tenancy writes $1 520,91 --
    a non-breaking space for thousands and a comma for the decimal. Reading
    that as "not a number" is what made a cost column look like a word
    column and asked the PM who owned 1,346 clocks when the answer is 1,317.
    """
    t = str(v).strip().replace("\u00a0", " ").replace("$", "").replace("%", "")
    t = t.replace("(", "-").replace(")", "")
    if not t:
        return None
    # 1 520,91 / 1.520,91 -> 1520.91;  1,520.91 -> 1520.91
    if re.search(r",\d{1,2}$", t) and not re.search(r",\d{3}(\D|$)", t):
        t = t.replace(".", "").replace(" ", "").replace(",", ".")
    else:
        t = t.replace(",", "").replace(" ", "")
    try:
        return float(t)
    except (TypeError, ValueError):
        return None


def header_index(rows: list[list[Any]], scan: int = 8) -> int:
    """Which of the leading rows is the header.

    The most populated of the first few non-blank rows. A title, a date
    stamp or an export banner sits above the header and is narrower than it,
    which is the whole of the signal and holds for any export from any tool.
    """
    best, best_filled, seen = -1, -1, 0
    for idx, row in enumerate(rows):
        row = row or []
        filled = sum(1 for c in row if str(c or "").strip())
        if not filled:
            continue
        seen += 1
        if filled > best_filled:
            best_filled, best = filled, idx
        if seen >= scan:
            break
    return best


def is_bulk_export(rows: list[list[Any]], header_idx: int) -> bool:
    """Is this a machine export, or something a person wrote?

    Two structural conditions, no vocabulary: there are far more rows than
    anybody types by hand, and the rows agree on a width -- an export is
    rectangular, whereas a hand-built sheet grows sections, blank spacers
    and side notes of differing widths as somebody works on it.
    """
    if header_idx < 0:
        return False
    body = rows[header_idx + 1:]
    if len(body) < BULK_ROWS:
        return False
    width = sum(1 for c in (rows[header_idx] or []) if str(c or "").strip())
    if width < 2:
        return False
    sample, agree, seen = body[:500], 0, 0
    for r in sample:
        r = r or []
        filled = sum(1 for c in r if str(c or "").strip())
        if not filled:
            continue
        seen += 1
        if filled >= width * 0.5:
            agree += 1
    return seen > 0 and agree >= seen * 0.8


def supplies(
    prof: SheetProfile, rows: list[list[Any]], header_idx: int
) -> tuple[list[Supply], dict[str, Role]]:
    """What this sheet can be asked for, and by which columns.

    This is the half of a table that findings and questions cannot express.
    "1,340 distinct site addresses" is not a claim about the deal that
    somebody argues with, and it is not a gap somebody fills in: it is the
    sheet saying what it is able to answer. A SOW builder asking for sites,
    or a lookup answering "where is cost centre 0010058001", routes on this
    -- so nobody has to write a rule per workbook, which is the thing that
    does not scale.
    """
    if not prof.headed:
        return [], {}
    roles: dict[str, Role] = {}
    for c in prof.columns:
        if not c.name.strip() or _UNNAMED.match(c.name.strip()):
            continue
        r = classify(c.name, c.sample, is_identifier=c.is_identifier)
        if r is not None:
            roles[c.name] = r

    key = next((c.name for c in prof.columns
                if c.is_identifier and c.filled >= prof.rows * 0.5), None)
    names = [c.name for c in prof.columns]
    out: list[Supply] = []
    site = site_supply(names, roles, rows, header_idx, key_column=key)
    if site is not None:
        out.append(site)
    contact = contact_supply(
        roles,
        {c.name: c.distinct for c in prof.columns},
        prof.rows,
    )
    if contact is not None:
        out.append(contact)
    return out, roles


def supply_observations(sups: list[Supply], sheet: str) -> list[Observation]:
    """State what the sheet can supply, and ask what it cannot settle."""
    out: list[Observation] = []
    for sup in sups:
        if sup.what == "site_address":
            out.append(Observation(
                "finding", "supplies:site_address",
                f"{sup.distinct:,} distinct site addresses across "
                f"{sup.rows:,} rows",
                "Assembled from "
                + ", ".join(f"{k} = {v}" for k, v in sup.columns.items())
                + ". This is the sheet's answer to where the work is: a "
                "rollout is planned and priced per place, and the row count "
                "is a device count, not a site count.",
                value=sup.distinct,
                decides="The site list the SOW is written against, and the "
                        "number of visits the job is priced for."))
            for note in sup.notes:
                out.append(Observation(
                    "question", f"site_address_gap:{note[:24]}",
                    f"The address columns are incomplete - {note}. Can the "
                    f"customer supply the missing part?",
                    "An address that cannot be resolved to a door cannot be "
                    "scheduled, routed or signed off, and a crew sent to an "
                    "area is a wasted visit.",
                    value=sup.distinct,
                    decides="Whether the site list can be used as-is or has "
                            "to go back to the customer."))
        elif sup.what == "contact":
            out.append(Observation(
                "finding", "supplies:contact",
                f"The sheet carries {', '.join(sup.columns)} for its rows",
                "Somebody to reach per record, which is what site access and "
                "scheduling run on.",
                value=list(sup.columns)))
    return out
