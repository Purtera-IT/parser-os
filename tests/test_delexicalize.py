"""Tests for app.core.delexicalize — the generalization enforcer.

The load-bearing test is ``test_name_swap_invariant``: swapping the proper
nouns in the raw text must NOT change the masked text. That invariance is the
training-time guarantee that a head learns the role, not the identity.
"""

from __future__ import annotations

from app.core.delexicalize import (
    LIT_ADDR,
    LIT_EMAIL,
    LIT_MONEY,
    ROLE_CUSTOMER,
    ROLE_SELF_ORG,
    delexicalize,
)


def test_masks_self_org_and_customer_roles():
    text = "PurTera proposes to serve Yonah County under this agreement."
    role_map = {"PurTera": ROLE_SELF_ORG, "Yonah County": ROLE_CUSTOMER}
    res = delexicalize(text, role_map)
    assert "PurTera" not in res.masked
    assert "Yonah County" not in res.masked
    assert ROLE_SELF_ORG in res.masked
    assert ROLE_CUSTOMER in res.masked
    # Reversible audit trail records what was hidden.
    assert ("PurTera", ROLE_SELF_ORG) in res.substitutions


def test_name_swap_invariant():
    """The whole point: different names, same roles → identical masked text."""
    t1 = "PurTera's headquarters at 11720 Amber Park Drive is not a job site."
    t2 = "Acme Corp's headquarters at 400 Industrial Way is not a job site."
    m1 = delexicalize(t1, {"PurTera": ROLE_SELF_ORG}).masked
    m2 = delexicalize(t2, {"Acme Corp": ROLE_SELF_ORG}).masked
    assert m1 == m2, f"masked text leaked identity:\n  {m1!r}\n  {m2!r}"
    # And the rule-bearing structure survived.
    assert ROLE_SELF_ORG in m1
    assert LIT_ADDR in m1


def test_longest_surface_wins():
    text = "Acme Security Corp and Acme are the same firm."
    role_map = {"Acme": ROLE_CUSTOMER, "Acme Security Corp": ROLE_CUSTOMER}
    res = delexicalize(text, role_map)
    # "Acme Security Corp" masked as a unit, not left as "<CUSTOMER> Security Corp".
    assert "Security Corp" not in res.masked


def test_literal_masking_shapes():
    text = "Email ops@purtera.com, pay $12,500.00 to 100 Main Street."
    res = delexicalize(text, None)
    assert LIT_EMAIL in res.masked
    assert LIT_MONEY in res.masked
    assert LIT_ADDR in res.masked
    assert "purtera.com" not in res.masked


def test_unknown_placeholder_ignored():
    # A bad caller can't inject junk tokens as roles.
    res = delexicalize("Foo bar", {"Foo": "<NOT_A_ROLE>"})
    assert "Foo" in res.masked  # left untouched
    assert res.substitutions == [] or all(
        ph != "<NOT_A_ROLE>" for _, ph in res.substitutions
    )


def test_empty_text():
    res = delexicalize("", {"X": ROLE_CUSTOMER})
    assert res.masked == ""
    assert res.substitutions == []


def test_mask_literals_can_be_disabled():
    text = "Pay $500 now."
    res = delexicalize(text, None, mask_literals=False)
    assert "$500" in res.masked
