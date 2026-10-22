"""000123: a row counter typed twice is not the store id, and two stores that
share a name (a copy error in the sheet) are still two sites."""
from pathlib import Path

from app.core.orbitbrief_core import build_site_readiness
from app.parsers.registry import choose_parser


def test_the_unique_store_number_keys_rows_and_a_shared_name_merges_nothing(tmp_path: Path) -> None:
    rows = ["BevMo Store\tlocid\tLocationName\tAddress\tCity\tState\tZip Code",
            "1\t406\tOAK_Walnut-Creek_406\t2940 N. Main Street\tWalnut Creek\tCA\t94597",
            "2\t446\tSAN_Point-Loma_446\t3210 Rosecrans Place\tSan Diego\tCA\t92110",
            "3\t495\tSAN_Point-Loma_446\t200 S. Brand Boulevard\tGlendale\tCA\t91204",
            "4\t501\tSAN_Mission-Valley_501\t925a Camino De La Reina\tSan Diego\tCA\t92108",
            "4\t1456\tSAN_Mission-Valley-North_1456\t5644 Mission Center Road\tSan Diego\tCA\t92108"]
    path = tmp_path / "locations.txt"
    path.write_text("\r\n".join(rows) + "\r\n", encoding="utf-8")
    parser, _, _ = choose_parser(path)
    atoms = parser.parse(path)
    ids = [a.value.get("site_id") for a in atoms]
    assert len(ids) == 5 and len(set(ids)) == 5
    assert len(build_site_readiness(atoms=atoms, edges=[])["sites"]) == 5
