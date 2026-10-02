"""Countries, US states, Canadian provinces and world regions, by name.

A rate sheet or a Gantt prices work per country ("Country: United States",
"Owner: Hong Kong"), and a name-shaped reader that sees two capitalised
words beside a role cue reads the country as a person: deal 010246's Gantt
came back with ``stakeholder:united_states``, and its duplicate rows folded
into that "person". A place is never a party to the deal, so a stakeholder
identity whose whole name is one of these is refused wherever it is minted.

Whole-name match only: "Georgia Smith" is a person, "Georgia" alone is not a
stakeholder anyone can act on, "Jordan Lee" is a person.
"""
from __future__ import annotations

import re

_COUNTRIES = """
afghanistan albania algeria andorra angola argentina armenia australia austria
azerbaijan bahamas bahrain bangladesh barbados belarus belgium belize benin
bhutan bolivia bosnia and herzegovina botswana brazil brunei bulgaria
burkina faso burundi cambodia cameroon canada cape verde chad chile china
colombia comoros congo costa rica croatia cuba cyprus czechia czech republic
denmark djibouti dominica dominican republic ecuador egypt el salvador
equatorial guinea eritrea estonia eswatini ethiopia fiji finland france gabon
gambia germany ghana greece grenada guatemala guinea guyana haiti honduras
hong kong hungary iceland india indonesia iran iraq ireland israel italy
ivory coast jamaica japan jordan kazakhstan kenya kiribati kosovo kuwait
kyrgyzstan laos latvia lebanon lesotho liberia libya liechtenstein lithuania
luxembourg macau madagascar malawi malaysia maldives mali malta mauritania
mauritius mexico micronesia moldova monaco mongolia montenegro morocco
mozambique myanmar namibia nepal netherlands new zealand nicaragua niger
nigeria north korea north macedonia norway oman pakistan palau panama
papua new guinea paraguay peru philippines poland portugal puerto rico qatar
romania russia rwanda saudi arabia senegal serbia seychelles sierra leone
singapore slovakia slovenia somalia south africa south korea south sudan spain
sri lanka sudan suriname sweden switzerland syria taiwan tajikistan tanzania
thailand togo tonga trinidad and tobago tunisia turkey turkiye turkmenistan
uganda ukraine united arab emirates uae united kingdom uk great britain
england scotland wales northern ireland united states usa us
united states of america uruguay uzbekistan vanuatu venezuela vietnam
viet nam yemen zambia zimbabwe korea
"""

_US_STATES = """
alabama alaska arizona arkansas california colorado connecticut delaware
district of columbia florida georgia hawaii idaho illinois indiana iowa kansas
kentucky louisiana maine maryland massachusetts michigan minnesota mississippi
missouri montana nebraska nevada new hampshire new jersey new mexico new york
north carolina north dakota ohio oklahoma oregon pennsylvania rhode island
south carolina south dakota tennessee texas utah vermont virginia washington
west virginia wisconsin wyoming
"""

_CA_PROVINCES = """
alberta british columbia manitoba new brunswick newfoundland nova scotia
ontario prince edward island quebec saskatchewan yukon nunavut
northwest territories
"""

_REGIONS = """
north america south america latin america central america europe asia africa
oceania middle east emea apac apj latam amer amers americas asia pacific
western europe eastern europe southeast asia
"""

# Multi-word names first so "united arab emirates" is read whole.
_MULTI = [
    "bosnia and herzegovina", "burkina faso", "cape verde", "costa rica", "czech republic",
    "dominican republic", "el salvador", "equatorial guinea", "hong kong", "ivory coast",
    "new zealand", "north korea", "north macedonia", "papua new guinea", "puerto rico",
    "saudi arabia", "sierra leone", "south africa", "south korea", "south sudan",
    "sri lanka", "trinidad and tobago", "united arab emirates", "united kingdom",
    "great britain", "northern ireland", "united states", "united states of america",
    "viet nam", "district of columbia", "new hampshire", "new jersey", "new mexico",
    "new york", "north carolina", "north dakota", "rhode island", "south carolina",
    "south dakota", "west virginia", "british columbia", "new brunswick", "nova scotia",
    "prince edward island", "northwest territories", "north america", "south america",
    "latin america", "central america", "middle east", "asia pacific", "western europe",
    "eastern europe", "southeast asia",
]


def _names() -> frozenset[str]:
    words: set[str] = set(_MULTI)
    multi_tokens = {t for m in _MULTI for t in m.split()}
    for block in (_COUNTRIES, _US_STATES, _CA_PROVINCES, _REGIONS):
        for tok in block.split():
            # single-word names; tokens that only occur inside a multi-word
            # name ("united", "new", "north") are not names on their own
            if tok in multi_tokens and tok not in {
                "georgia", "washington", "jordan", "korea", "guinea", "ireland",
                "africa", "america", "europe", "asia", "mexico", "sudan", "macedonia",
                "virginia", "carolina", "dakota", "columbia", "zealand", "york",
            }:
                continue
            words.add(tok)
    return frozenset(words)


PLACE_NAMES: frozenset[str] = _names()


def _norm(text: str) -> str:
    t = str(text or "").lower().replace("_", " ").replace("&", " and ")
    t = re.sub(r"[^a-z ]+", " ", t)
    t = re.sub(r"^the ", "", " ".join(t.split()))
    return t


def is_place_name(text_or_slug: str) -> bool:
    """True when the WHOLE name is a country, US state, Canadian province or
    world region ("United States", ``united_states``, "Hong Kong", "EMEA")."""
    t = _norm(text_or_slug)
    return bool(t) and t in PLACE_NAMES


__all__ = ["PLACE_NAMES", "is_place_name"]
