"""A SOW never carries a credential (010264 printed a clock password and SSID)."""
from app.core.c3_sow import _scrub, _withhold


def test_values_after_a_credential_word_are_withheld() -> None:
    got = _withhold("Maintenance password: 8812ab, SSID: SodexoClock, gateway 10.2.3.4")
    assert "8812ab" not in got and "SodexoClock" not in got and "10.2.3.4" not in got


def test_ordinary_words_are_not_credentials() -> None:
    text = "Customer handles password reset; mount the PIN pad; the password is provided by the customer"
    assert _withhold(text) == text


def test_items_still_carrying_one_are_dropped() -> None:
    c = {"scope": [{"text": "Enter password 1234 on the clock"}, {"text": "Mount the clock"},
                   {"text": "Join SSID: [withheld]"}]}
    assert _scrub(c) == 2 and [x["text"] for x in c["scope"]] == ["Mount the clock"]
