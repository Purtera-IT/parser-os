from app.core.site_facility_head import annotate_site_facility_labels, decide_site_facility_label


class _Atom:
    def __init__(self, value, raw_text="", entity_keys=None):
        self.atom_type = type("T", (), {"value": "physical_site"})()
        self.value = value
        self.raw_text = raw_text
        self.entity_keys = entity_keys or []
        self.review_flags = []


def test_site_facility_head_keeps_facility_when_no_locality() -> None:
    # No city/state parsed (e.g. a transcript-only site whose address note was never
    # ingested): the head must not invent a city and keeps the existing facility label.
    atom = _Atom(
        {
            "site_id": "GECKO-ROBOTICS-NEW-OFFICE-WORKSHOP",
            "name": "gecko robotics new office workshop",
            "facility_name": "gecko robotics new office workshop",
        }
    )
    decision = decide_site_facility_label(atom)
    assert decision.label == "keep_facility"
    assert decision.facility_name == "gecko robotics new office workshop"


def test_site_facility_head_uses_city_office_for_address_backed_site() -> None:
    atom = _Atom(
        {
            "site_id": "PITTSBURGH-PA-15212",
            "name": "100 S COMMONS STE 145, PITTSBURGH, PA 15212",
            "street_address": "100 S COMMONS STE 145",
            "city": "PITTSBURGH",
            "state": "PA",
            "zip": "15212",
            "aliases": ["gecko robotics pittsburgh office workshop"],
        }
    )
    # The CITY, verbatim — not "Pittsburgh Office".
    #
    # This test used to pin the composed form. Composing the word "Office"
    # invents a name that is in no document, which is what the base-health
    # `fabricated_names` invariant counts ("Zero, permanently"): 371 of 858
    # fabricated names in a 140-envelope sample, and for 367 of them the bare
    # city IS in the source text. `city_office` is still a verdict the HEAD may
    # predict once a PM teaches it — a name a person chose is a judgement; a
    # name this rule invented is a fabrication.
    decision = decide_site_facility_label(atom)
    assert decision.facility_name == "Pittsburgh"
    atoms, n = annotate_site_facility_labels([atom], project_id="deal-1")
    assert n == 1
    assert atoms[0].value["facility_name"] == "Pittsburgh"
    assert atoms[0].value["name"] == "Pittsburgh"
    assert atoms[0].value["display_name"] == "Pittsburgh"


class _QuoteTask:
    def __init__(self, text, site="site:pittsburgh_pa_15212"):
        self.atom_type = type("T", (), {"value": "task"})()
        self.value = {
            "text": text,
            "task_tier": "parent",
            "is_quote_line": True,
            "quote_context": {"delivery_model": "config_only"},
        }
        self.entity_keys = [site]
        self.raw_text = text


_CONFIG_LINES = [
    "Camera configuration / setup",
    "Badge reader configuration",
    "Okta integration",
    "UID Enterprise setup",
    "Knowledge transfer / walking him through the setup",
]


def _task_names(out):
    return sorted(a.value["text"] for a in out if getattr(getattr(a, "atom_type", None), "value", "") == "task")


def test_quote_lines_keep_their_own_words_without_a_trained_head() -> None:
    """No head: the rule menu must not rename or merge work. 000061 MBrany had
    twenty recap sentences renamed "Wireless site survey"."""
    from app.core.quote_line_head import CONFIG_UMBRELLA, consolidate_quote_line_tasks

    atoms = [_QuoteTask(t) for t in _CONFIG_LINES]
    out, _ = consolidate_quote_line_tasks(atoms, project_id="deal-1")
    assert _task_names(out) == sorted(_CONFIG_LINES)
    assert CONFIG_UMBRELLA not in _task_names(out)
    assert all(a.value["quote_line"]["source"] == "deterministic_fallback" for a in out)
    assert all(not a.value["quote_line"]["technician_skill"] for a in out)


def test_quote_line_head_merges_config_install_lines_when_a_head_decides(monkeypatch) -> None:
    from app.core import quote_line_head as qlh

    def _head(relation, text, candidates):
        if relation == qlh.TASK_TECHNICIAN_SKILL_RELATION:
            return "Network / Wireless L2", 0.9, "neural_head"
        if "knowledge transfer" in text.lower():
            return None, 0.0, "head_miss"
        return qlh.CONFIG_UMBRELLA, 0.9, "neural_head"

    monkeypatch.setattr(qlh, "_head_classify", _head)
    atoms = [_QuoteTask(t) for t in _CONFIG_LINES]
    out, changed = qlh.consolidate_quote_line_tasks(atoms, project_id="deal-1")
    names = _task_names(out)
    assert names == sorted([qlh.CONFIG_UMBRELLA, "Knowledge transfer / walking him through the setup"])
    config_task = next(a for a in out if a.value["text"] == qlh.CONFIG_UMBRELLA)
    assert config_task.value["technician_skill"] == "Network / Wireless L2"
    assert "Okta integration" in config_task.value["quote_line"]["original_text"]


def test_quote_line_head_collapses_ubiquiti_micro_tasks_when_a_head_decides(monkeypatch) -> None:
    from app.core import quote_line_head as qlh

    monkeypatch.setattr(qlh, "_head_classify", lambda relation, text, c: (
        ("Network / Wireless L2", 0.9, "neural_head") if relation == qlh.TASK_TECHNICIAN_SKILL_RELATION
        else (qlh.CONFIG_UMBRELLA, 0.9, "neural_head")))
    atoms = [_QuoteTask("Ubiquiti VLAN configuration / setup"), _QuoteTask("UDM Beast integration support")]
    out, _ = qlh.consolidate_quote_line_tasks(atoms, project_id="deal-1")
    assert _task_names(out) == [qlh.CONFIG_UMBRELLA]


def test_hardware_evidence_backfill_mints_bom_lines() -> None:
    from app.core.hardware_evidence_backfill import backfill_hardware_bom_lines

    class _Scope:
        def __init__(self, text):
            self.atom_type = type("T", (), {"value": "scope_item"})()
            self.raw_text = text
            self.text = text
            self.value = {"text": text}

    atoms = [
        _Scope(
            "Everything is already installed. Just the configuration part. "
            "4 E7 APs, 2 UDM Beast, 2 48 port switches and 2 NVR."
        )
    ]
    out, minted = backfill_hardware_bom_lines(atoms, project_id="deal-1")
    assert minted >= 3
    bom = {
        a.value["sku"]: a.value["quantity"]
        for a in out
        if getattr(getattr(a, "atom_type", None), "value", "") == "bom_line"
    }
    assert bom["UBNT-E7-AP"] == 4
    assert bom["UBNT-UDM-BEAST"] == 2
    assert bom["UBNT-SW-PRO"] == 2


def test_hardware_evidence_backfill_mints_udm_unvr_and_badge_reader() -> None:
    from app.core.hardware_evidence_backfill import backfill_hardware_bom_lines

    class _Scope:
        def __init__(self, text):
            self.atom_type = type("T", (), {"value": "scope_item"})()
            self.raw_text = text
            self.text = text
            self.value = {"text": text}

    atoms = [
        _Scope(
            "Installed gear: two UDM Beast, 2 UNVR, three badge readers, 1 G6 Pro doorbell."
        )
    ]
    out, minted = backfill_hardware_bom_lines(atoms, project_id="deal-1")
    assert minted >= 3
    bom = {
        a.value["sku"]: a.value["quantity"]
        for a in out
        if getattr(getattr(a, "atom_type", None), "value", "") == "bom_line"
    }
    assert bom.get("UBNT-UDM-BEAST") == 2
    assert bom.get("UBNT-UNVR") == 2
    assert bom.get("UBNT-BADGE-READER") == 3


def test_hardware_evidence_backfill_ignores_manifest_blob_urls() -> None:
    from app.core.hardware_evidence_backfill import backfill_hardware_bom_lines

    class _Scope:
        def __init__(self, text, *, locator=None):
            self.atom_type = type("T", (), {"value": "scope_item"})()
            self.raw_text = text
            self.text = text
            self.value = {"text": text}
            self.locator = locator or {}

    atoms = [
        _Scope(
            "artifacts[3].blob_url: https://example.blob.core.windows.net/deals/x/010058-4%20e7%20aps.%202%20udm.txt",
            locator={"kind": "json_value", "key_path": "artifacts[3].blob_url"},
        ),
        _Scope("4 e7 aps and two 48 port switches in the workshop."),
    ]
    out, minted = backfill_hardware_bom_lines(atoms, project_id="deal-1")
    bom = {
        a.value["sku"]: a.value["quantity"]
        for a in out
        if getattr(getattr(a, "atom_type", None), "value", "") == "bom_line"
    }
    assert "UBNT-AP-GENERIC" not in bom
    assert bom.get("UBNT-E7-AP") == 4
    assert bom.get("UBNT-SW-PRO") == 2
