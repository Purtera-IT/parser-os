"""Universal overlay rules.

Each module in this package encodes one *universal* rule that the overlay
must satisfy on every PDF.  Rules are small, named, well-documented
functions with no hidden behaviour: edit one place to change the rule
everywhere it applies.

The rules currently encoded:

- ``text_bisection``         — no orange grid edge cuts a PDF text word.
- ``title_overrides_rings``  — title-washed cells suppress cyan/green inside.
- ``colhdr_row_fill``        — cell beside >=2 cyan peers gets cyan too.
- ``colhdr_x_scope``         — narrow right-margin captions stay in their
                              own x-range (no cross-table bleed).
- ``orange_cyan_coexist``    — orange cell + cyan ring both render.
- ``header_isolation``       — non-bold colon-uppercase span is a title
                              only if it stands alone in its column.

Each rule lives in its own file so that finding/tuning a rule for a new
PDF is "open the file named after the symptom you saw and read the rule."
"""
from .text_bisection import edge_bisects_word, candidate_bisects_any_word
from .title_overrides_rings import (
    drawn_title_bboxes,
    centroid_in_drawn_title,
)
from .colhdr_row_fill import row_fill_missing_colhdrs
from .colhdr_x_scope import x_scope_for_textsec_caption
from .orange_cyan_coexist import orange_should_render_with_colhdr
from .header_isolation import filter_isolated_colon_titles
from .label_value_pairing import add_value_spans_for_colon_labels
from .thin_stroke_artifacts import reclassify_thin_stroke_artifacts
from .fact_statements import filter_fact_statements
from .underline_titles import collect_underlined_caps_titles
from .continuation_blocks import find_continuation_blocks
from .section_hierarchy import is_section_parent, find_parent_child_links
from .thin_header_row import find_thin_header_rows
from .abbreviation_tables import find_abbreviation_tables

__all__ = [
    "edge_bisects_word",
    "candidate_bisects_any_word",
    "drawn_title_bboxes",
    "centroid_in_drawn_title",
    "row_fill_missing_colhdrs",
    "x_scope_for_textsec_caption",
    "orange_should_render_with_colhdr",
    "filter_isolated_colon_titles",
    "add_value_spans_for_colon_labels",
    "reclassify_thin_stroke_artifacts",
    "filter_fact_statements",
    "collect_underlined_caps_titles",
    "find_continuation_blocks",
    "is_section_parent",
    "find_parent_child_links",
    "find_thin_header_rows",
    "find_abbreviation_tables",
]
