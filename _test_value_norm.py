"""NORM front tests: deterministic value normalization + the deal_financials
consumer fallback. No network/embedder. Run: python _test_value_norm.py
"""
from types import SimpleNamespace as NS

from app.core.entity_extraction import (
    normalize_atom_value,
    parse_money,
    parse_quantity_spans,
)


def _norm(text, keys, value=None):
    a = NS(raw_text=text, entity_keys=list(keys), value=value)
    normalize_atom_value(a)
    qk = [k for k in a.entity_keys if k.startswith("quantity:")]
    return a.value or {}, qk


def test_parse_money_and_back_compat():
    from app.core.entity_extraction import _emit_money_keys
    assert parse_money("budget $1.5M")[0]["amount"] == 1_500_000
    assert parse_money("budget $1.5M")[0]["currency"] == "USD"
    # back-compat: the money:<n> keys are byte-identical
    assert _emit_money_keys("not-to-exceed $1,847,250.00") == {"money:1847250"}
    print("  ok: parse_money + back-compat money keys")


def test_single_money_sets_amount():
    v, _ = _norm("Grand total not-to-exceed $1,847,250.00", [])
    assert v.get("amount") == 1_847_250 and v.get("currency") == "USD"
    print("  ok: single money -> value.amount + currency")


def test_qty_plus_device_emits_key():
    v, qk = _norm("Install 60 access points at ATL", ["device:access_point"])
    assert v.get("quantity") == 60
    assert qk == ["quantity:60"]  # the cross-doc-conflict unblock
    print("  ok: qty + device -> value.quantity + quantity:60 key")


def test_multi_money_stays_key_only():
    v, _ = _norm("Rates: $98/hr, $4,200 conduit, $1,200 panel", [])
    assert v.get("amount") is None  # multi-value -> no scalar (avoids corruption)
    print("  ok: multi-money stays key-only")


def test_parser_value_wins():
    v, _ = _norm("60 access points", ["device:access_point"], value={"quantity": 55})
    assert v.get("quantity") == 55  # setdefault: parser/table value wins
    print("  ok: parser-supplied quantity is not clobbered")


def test_unanchored_qty_no_key():
    _, qk = _norm("we reviewed 60 items", [])
    assert qk == []  # no device/part anchor -> no quantity: key (avoids noise)
    print("  ok: unanchored quantity emits no key")


def test_deal_financials_fallback():
    from app.core.orbitbrief_core import build_deal_financials
    atoms = [
        NS(atom_type=NS(value="scope_item"), value={"amount": 4200}, id="a1"),
        NS(atom_type=NS(value="commercial_total"), value={"amount": 1_847_250}, id="a2"),
        NS(atom_type=NS(value="scope_item"), value={"amount": 1200}, id="a3"),
    ]
    fin = build_deal_financials(atoms=atoms)
    assert fin["present"] is True and fin.get("derived") is True
    assert fin["totals"]["revenue"] == 1_847_250  # leads with the max amount
    print("  ok: deal_financials fallback lights up (derived, max amount)")


if __name__ == "__main__":
    test_parse_money_and_back_compat()
    test_single_money_sets_amount()
    test_qty_plus_device_emits_key()
    test_multi_money_stays_key_only()
    test_parser_value_wins()
    test_unanchored_qty_no_key()
    test_deal_financials_fallback()
    print("PASS _test_value_norm")
