"""Tab to target, loss, and evaluation. Retained JSON is not a consumed target.

Statuses:
  consumed                         a student loss reads the saved value
  consumed_without_student_decoder the exact string is a teacher target only
  rationale                        privileged note text, not a class
  deferred                         no fixed target; preflight blocks it
  external_pending                 a separate learner, not this model
  blocked                          a release gate; unit tests do not clear it
"""
from __future__ import annotations

from .schema import NUMBER, RELATION, Schema
from .values import LITERAL_HEADS


READY_FOR_TRAINING = False


def release_blockers() -> list[str]:
    """Why a customer training run and shadow export stay refused."""
    if READY_FOR_TRAINING:
        raise RuntimeError("READY_FOR_TRAINING cannot be enabled in this tree")
    return [
        "ready_for_training_false",
        "frozen_corpus_not_audited",
        "rules_evaluation_not_separate",
        "pretrained_budget_not_certified",
        "student_normalized_string_head",
    ]


def _row(surface: str, target: str, loss: str, evaluation: str, status: str) -> dict[str, str]:
    return {"surface": surface, "target": target, "loss": loss,
            "evaluation": evaluation, "status": status}


def lesson_for(opp) -> str:
    """How this tab is taught, and which other tabs it is allowed to use."""
    if opp.field == "router" or opp.key == "jdg:router":
        return "not this model"
    if opp.key == "jdg:rule":
        return "the person's call on that line for the rule the card names; the parser's call is only an input"
    if opp.kind == RELATION:
        return "drawn link or no link; every other head reads only a link the model draws"
    if opp.size == "pair":
        return "the person's verdict on those two lines, and the link between them"
    if opp.size == "group":
        return "the person's verdict for that document, roster or sheet, using its lines"
    if opp.size == "deal":
        return "one answer for the whole deal, after each line has used its links"
    if opp.kind == NUMBER:
        return "the person's number wins; the paragraph fills a number they left blank"
    if opp.key.endswith(":reason"):
        return "the person's reason code on that same card; a sentence stays a note"
    return "the person's mark wins; the paragraph fills a question they left blank on that line"


def _opportunity_row(opp) -> dict[str, str]:
    if opp.field == "router" or opp.key == "jdg:router":
        return _row(opp.key, "none", "none", "preflight blocker", "deferred")
    if opp.key in {f"jdg:{head}" for head in LITERAL_HEADS}:
        return _row(opp.key, f"literal {opp.key}",
                    "brain.value_loss; literal_span_loss only for a unique in-line span",
                    "teacher token NLL; no student string decoder",
                    "consumed_without_student_decoder")
    if opp.kind == RELATION:
        loss, evaluation = "relation_loss", "labeled-edge exact match"
    elif opp.kind == NUMBER:
        loss, evaluation = "smooth-l1 on log1p", "labeled-number error; teacher match on a WHY"
    elif opp.size != "line" or opp.key in ("jdg:rule", "jdg:rule:reason"):
        # A rule card is a line card asked with its rule's question (data.judgment_query).
        loss, evaluation = "judgment_loss", "per-head exact match on labeled subjects"
    elif opp.multiple:
        loss, evaluation = "sigmoid BCE", "exact set match on labeled rows"
    else:
        loss, evaluation = "answer_loss", "per-head exact match on labeled rows"
    return _row(opp.key, opp.key, loss, evaluation + "; not learned accuracy", "consumed")


def coverage_table(schema: Schema | None = None) -> list[dict[str, str]]:
    """One row per schema opportunity, plus surfaces that are not opportunities."""
    if schema is None:
        from .schema import load_schema
        schema = load_schema()
    rows = []
    for opp in schema.opportunities:
        row = _opportunity_row(opp)
        row["lesson"] = lesson_for(opp)
        rows.append(row)
    seen = {row["surface"] for row in rows}
    for name in schema.note_fields:
        rows.append(_row(f"note:{name}", "field note", "rationale loss",
                         "not a class score", "rationale"))
    for head in sorted(LITERAL_HEADS):
        key = f"jdg:{head}"
        if key not in seen:
            rows.append(_row(key, f"literal {key}",
                             "brain.value_loss; literal_span_loss only for a unique in-line span",
                             "teacher token NLL; no student string decoder",
                             "consumed_without_student_decoder"))
    if "deal:declared_site_count" not in seen:
        rows.append(_row("deal:declared_site_count", "deal_numbers", "deal_number_loss",
                         "smooth-l1 on log1p(count)", "consumed"))
    if "jdg:rule" not in seen:
        rows.append(_row("jdg:rule", "separate Rules export", "not c3_loss",
                         "separate Rules holdout", "external_pending"))
    if "jdg:router" not in seen:
        rows.append(_row("jdg:router", "none", "none", "preflight blocker", "deferred"))
    rows.append(_row("router", "none", "none", "preflight blocker", "blocked"))
    for code in release_blockers():
        rows.append(_row(code, "release gate", "none",
                         "train() and export_shadow refuse", "blocked"))
    return rows


def main() -> None:
    for row in coverage_table():
        print("\t".join(row.get(k, "") for k in ("status", "surface", "lesson", "loss", "evaluation")))


if __name__ == "__main__":
    main()
