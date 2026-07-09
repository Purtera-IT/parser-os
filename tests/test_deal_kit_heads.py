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
    decision = decide_site_facility_label(atom)
    assert decision.facility_name == "Pittsburgh Office"
    atoms, n = annotate_site_facility_labels([atom], project_id="deal-1")
    assert n == 1
    assert atoms[0].value["facility_name"] == "Pittsburgh Office"
    assert atoms[0].value["name"] == "Pittsburgh Office"
    assert atoms[0].value["display_name"] == "Pittsburgh Office"


def test_quote_line_head_merges_config_install_email_lines() -> None:
    from app.core.quote_line_head import CONFIG_UMBRELLA, consolidate_quote_line_tasks

    class _Task:
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

    atoms = [
        _Task("Camera configuration / setup"),
        _Task("Badge reader configuration"),
        _Task("Okta integration"),
        _Task("UID Enterprise setup"),
        _Task("Knowledge transfer / walking him through the setup"),
    ]
    out, changed = consolidate_quote_line_tasks(atoms, project_id="deal-1")
    task_atoms = [a for a in out if getattr(getattr(a, "atom_type", None), "value", "") == "task"]
    names = sorted(a.value["text"] for a in task_atoms)
    assert changed >= 4
    assert names == sorted([CONFIG_UMBRELLA, "Knowledge transfer / walking him through the setup"])
    assert len(task_atoms) == 2
    config_task = next(a for a in task_atoms if a.value["text"] == CONFIG_UMBRELLA)
    assert config_task.value["technician_skill"] == "Network / Wireless L2"
    originals = config_task.value["quote_line"]["original_text"]
    assert "Camera configuration" in originals
    assert "Okta integration" in originals
    assert "guided handoff" not in " ".join(names)


def test_quote_line_head_collapses_ubiquiti_micro_tasks() -> None:
    from app.core.quote_line_head import CONFIG_UMBRELLA, consolidate_quote_line_tasks

    class _Task:
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

    atoms = [
        _Task("Ubiquiti VLAN configuration / setup"),
        _Task("UDM Beast integration support"),
    ]
    out, _ = consolidate_quote_line_tasks(atoms, project_id="deal-1")
    task_atoms = [a for a in out if getattr(getattr(a, "atom_type", None), "value", "") == "task"]
    names = [a.value["text"] for a in task_atoms]
    assert names == [CONFIG_UMBRELLA]


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
