"""Site tier and remote_miles from official US place classifications.

A labeler used to guess ``location_tier`` ("small_town/rural" for Tupelo, MS).
This tool reads it instead from the OMB Core Based Statistical Area (CBSA)
delineation, the federal definition of a labor market: a county belongs to a
Metropolitan area, a Micropolitan area, or neither, by where its people
commute to work. That is exactly the question ``location_tier`` asks (is a
metro tech pool within reach of the site), so the mapping is one table:

    OMB class of the site's county            location_tier
    ----------------------------------------  -------------
    Metropolitan area, population >= 1M       major_metro
    Metropolitan area, population <  1M       mid_metro
    Micropolitan area                         small_town
    Outside every CBSA                        rural

``remote_miles`` is the straight-line distance from the site to the principal
city of the nearest metropolitan area of 1M or more (0 inside one), written
the way the registry asks: "103 miles from Memphis, TN".

Data: the ``uszipinfo`` package (MIT), which packs per-ZIP Census data
already joined to the OMB delineation: ZIP-to-county (Census ZCTA-county
relationship), county-to-CBSA and Metro/Micro type (OMB), ZCTA internal
points (Census Gazetteer) and population (ACS 5-year). A site resolves by its
ZIP when the address has one, otherwise by every standard ZIP whose primary
city is the site's city; a city spread over several CBSAs reports the split.
Metro population is the sum of its ZIPs' ACS population, so a metro within
5% of the 1M line is flagged ``near_threshold``.

Pure stdlib apart from loading the table (pandas + pyarrow, which
``uszipinfo`` already needs). Read-only.

Usage:
  pip install uszipinfo
  python tools/geo_tier.py "Tupelo, MS" "Hudson, WI" 10014
  python tools/geo_tier.py --json "Delphos, OH"
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

MAJOR_METRO_POP = 1_000_000
NEAR_THRESHOLD = 0.05
#: A city whose second CBSA holds at least this share of its people is a split.
SPLIT_SHARE = 0.20
EARTH_MILES = 3958.8

TIER_TABLE = (
    ("Metro", ">= 1M", "major_metro"),
    ("Metro", "< 1M", "mid_metro"),
    ("Micro", "any", "small_town"),
    (None, "any", "rural"),
)

SOURCE = ("OMB CBSA delineation + Census ZCTA-county, Gazetteer and ACS 5-year, "
          "via uszipinfo")


@dataclass
class Zip:
    zip: str
    city: str
    state: str
    county: Optional[str]
    county_fips: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    cbsa_code: Optional[str]
    cbsa_name: Optional[str]
    cbsa_type: Optional[str]  # "Metro" | "Micro" | None
    population: int
    standard: bool = True


@dataclass
class Result:
    query: str
    location_tier: Optional[str]
    remote_miles: Optional[int] = None
    remote_miles_text: Optional[str] = None
    cbsa_code: Optional[str] = None
    cbsa_name: Optional[str] = None
    cbsa_type: Optional[str] = None
    cbsa_population: Optional[int] = None
    county: Optional[str] = None
    county_fips: Optional[str] = None
    zips: list[str] = field(default_factory=list)
    split: list[dict] = field(default_factory=list)
    near_threshold: bool = False
    resolved_by: str = ""
    error: Optional[str] = None
    source: str = SOURCE


def tier_for(cbsa_type: Optional[str], cbsa_population: Optional[int]) -> str:
    if cbsa_type == "Metro":
        return "major_metro" if (cbsa_population or 0) >= MAJOR_METRO_POP else "mid_metro"
    if cbsa_type == "Micro":
        return "small_town"
    return "rural"


def miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_MILES * math.asin(math.sqrt(a))


def _centroid(zips: Iterable[Zip]) -> Optional[tuple[float, float]]:
    pts = [(z.lat, z.lon, max(z.population, 1)) for z in zips
           if z.lat is not None and z.lon is not None]
    if not pts:
        return None
    w = sum(p[2] for p in pts)
    return (sum(p[0] * p[2] for p in pts) / w, sum(p[1] * p[2] for p in pts) / w)


def principal_city(cbsa_name: str) -> tuple[str, str]:
    """First named city and state: "Memphis, TN-MS-AR" -> ("Memphis", "TN")."""
    cities, _, states = cbsa_name.partition(", ")
    return cities.split("-")[0].strip(), states.split("-")[0].strip()


class GeoTable:
    def __init__(self, rows: Iterable[Zip]):
        self.rows = [r for r in rows]
        self.by_zip = {r.zip: r for r in self.rows}
        self.by_city: dict[tuple[str, str], list[Zip]] = defaultdict(list)
        self.by_cbsa: dict[str, list[Zip]] = defaultdict(list)
        for r in self.rows:
            if r.standard:
                self.by_city[(r.city.casefold(), r.state)].append(r)
            if r.cbsa_code:
                self.by_cbsa[r.cbsa_code].append(r)
        self.cbsa_pop = {c: sum(z.population for z in zs) for c, zs in self.by_cbsa.items()}
        self._centers: dict[str, Optional[tuple[float, float]]] = {}

    def metro_center(self, cbsa_code: str) -> Optional[tuple[float, float]]:
        """Population centre of the metro's principal city (else the metro)."""
        if cbsa_code not in self._centers:
            zs = self.by_cbsa[cbsa_code]
            city, state = principal_city(zs[0].cbsa_name or "")
            core = [z for z in zs if z.city.casefold() == city.casefold() and z.state == state]
            self._centers[cbsa_code] = _centroid(core) or _centroid(zs)
        return self._centers[cbsa_code]

    def major_metros(self) -> list[str]:
        return [c for c, zs in self.by_cbsa.items()
                if zs[0].cbsa_type == "Metro" and self.cbsa_pop[c] >= MAJOR_METRO_POP]

    def nearest_major_metro(self, lat: float, lon: float) -> Optional[tuple[str, float]]:
        best = None
        for c in self.major_metros():
            ctr = self.metro_center(c)
            if ctr is None:
                continue
            d = miles(lat, lon, *ctr)
            if best is None or d < best[1]:
                best = (c, d)
        return best

    def _find(self, query: str) -> tuple[list[Zip], str]:
        q = query.strip()
        m = re.search(r"\b(\d{5})(?:-\d{4})?\s*$", q)
        if m and m.group(1) in self.by_zip:
            return [self.by_zip[m.group(1)]], "zip"
        m = re.match(r"^\s*(.+?)\s*,\s*([A-Za-z]{2})\s*$", q)
        if m:
            return list(self.by_city.get((m.group(1).casefold(), m.group(2).upper()), [])), "city"
        return [], ""

    def lookup(self, query: str) -> Result:
        zips, how = self._find(query)
        if not zips:
            return Result(query=query, location_tier=None,
                          error="not found: give 'City, ST' or a 5-digit ZIP")
        share: dict[Optional[str], int] = defaultdict(int)
        for z in zips:
            share[z.cbsa_code] += max(z.population, 0)
        total = sum(share.values()) or 1
        ranked = sorted(share.items(), key=lambda kv: -kv[1])
        code = ranked[0][0]
        main = [z for z in zips if z.cbsa_code == code]
        by_county: dict[Optional[str], int] = defaultdict(int)
        for z in main:
            by_county[z.county_fips] += max(z.population, 0)
        top_county = max(by_county, key=by_county.get)
        head = next(z for z in main if z.county_fips == top_county)
        pop = self.cbsa_pop.get(code) if code else None
        tier = tier_for(head.cbsa_type, pop)
        res = Result(
            query=query, location_tier=tier, cbsa_code=code,
            cbsa_name=head.cbsa_name, cbsa_type=head.cbsa_type, cbsa_population=pop,
            county=head.county, county_fips=head.county_fips,
            zips=sorted(z.zip for z in zips), resolved_by=how,
            near_threshold=bool(head.cbsa_type == "Metro" and pop
                                and abs(pop - MAJOR_METRO_POP) <= NEAR_THRESHOLD * MAJOR_METRO_POP),
        )
        if len(ranked) > 1 and ranked[1][1] / total >= SPLIT_SHARE:
            for c, p in ranked:
                z0 = next(z for z in zips if z.cbsa_code == c)
                res.split.append({"cbsa_code": c, "cbsa_name": z0.cbsa_name,
                                  "cbsa_type": z0.cbsa_type,
                                  "location_tier": tier_for(z0.cbsa_type, self.cbsa_pop.get(c) if c else None),
                                  "share": round(p / total, 2)})
        if tier == "major_metro":
            res.remote_miles = 0
            city, state = principal_city(head.cbsa_name or "")
            res.remote_miles_text = f"0 miles: inside the {city}, {state} metro"
        else:
            ctr = _centroid(zips)
            near = self.nearest_major_metro(*ctr) if ctr else None
            if near:
                res.remote_miles = round(near[1])
                city, state = principal_city(self.by_cbsa[near[0]][0].cbsa_name or "")
                res.remote_miles_text = f"{res.remote_miles} miles from {city}, {state}"
        return res


def load_uszipinfo() -> GeoTable:
    try:
        import uszipinfo  # noqa: F401
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(f"needs `pip install uszipinfo` ({exc})")
    from importlib.resources import files

    path = next(p for p in (files("uszipinfo") / "_data").iterdir() if p.name.endswith(".parquet"))
    df = pd.read_parquet(str(path))

    def s(v):
        return None if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA else str(v)

    def f(v):
        return None if s(v) is None else float(v)

    rows = [Zip(zip=str(r.zip), city=str(r.primary_city or ""), state=str(r.state or ""),
                county=s(r.county), county_fips=s(r.county_fips), lat=f(r.lat), lon=f(r.lon),
                cbsa_code=s(r.cbsa_code), cbsa_name=s(r.cbsa_name), cbsa_type=s(r.cbsa_type),
                population=int(r.population) if s(r.population) is not None else 0,
                standard=(r.zip_type == "Standard"))
            for r in df.itertuples(index=False)]
    return GeoTable(rows)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sites", nargs="+", help="'City, ST' or an address ending in a ZIP")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    table = load_uszipinfo()
    results = [table.lookup(q) for q in args.sites]
    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
        return 0
    for r in results:
        if r.error:
            print(f"{r.query}: {r.error}")
            continue
        extra = " (near the 1M line)" if r.near_threshold else ""
        print(f"{r.query}: {r.location_tier}{extra} | {r.cbsa_type or 'no CBSA'} "
              f"{r.cbsa_code or ''} {r.cbsa_name or ''} pop {r.cbsa_population or '-'} | "
              f"{r.county} {r.county_fips} | remote_miles {r.remote_miles_text}")
        for sp in r.split:
            print(f"    split {sp['share']:.0%} {sp['cbsa_name'] or 'no CBSA'} -> {sp['location_tier']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
