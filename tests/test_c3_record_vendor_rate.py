"""Asking a supplier for its rate is our cost side, never our quote (owner ruling, 010224)."""
from c3_record.sender import sends_quote, to_supplier_only


def test_asking_their_rate_is_not_our_quote():
    assert not sends_quote("What travel budget would you need to get the rate down to $95?")
    assert not sends_quote("Can you send your best rate for the PDU install?")
    assert sends_quote("The T&M rate we would use for something like this is $115/per hour")
    assert sends_quote("Please see the attached survey quote and install ROM.")


def test_recipients_all_at_a_supplier():
    to = [{"email": "nick@network-dynamics.com"}, {"email": "t@purtera-it.com"}]
    assert to_supplier_only(to, {"network-dynamics.com"})
    assert not to_supplier_only(to + [{"email": "a@computacenter.com"}], {"network-dynamics.com"})
    assert not to_supplier_only(to, set())
