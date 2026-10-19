"""PurTera SOW standards, editable without a deploy (app/core/standards_store.py).

    GET  /sow/standards[?family=]            merged standards + the overlay's items
    POST /sow/standards/items                add an item (takes effect on the next compile)
    POST /sow/standards/items/{id}/retire    retire an added item
    POST /sow/standards/retire               retire a packaged item by family/section/text

SowSmith's "Add as template item" action posts here. Every item records who
added it, when, and the deal it came from.
"""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core import standards_store

router = APIRouter(prefix="/sow/standards", tags=["standards"])


class NewItem(BaseModel):
    family: str
    section: str
    text: str
    added_by: str
    when: str = ""
    unless: str = ""
    source_deal: str = ""
    note: str = ""


class Retire(BaseModel):
    by: str
    family: str = ""
    section: str = ""
    text: str = ""


@router.get("")
def get_standards(family: str = ""):
    merged = standards_store.load()
    overlay, _ = standards_store.read_overlay()
    items = overlay.get("items") or []
    if family:
        merged = {family: merged.get(family) or {}}
        items = [i for i in items if i.get("family") == family]
    families = sorted(k for k in standards_store.load() if not k.startswith("_"))
    return {"families": families, "sections": list(standards_store.SECTIONS), "standards": merged,
            "added": items}


@router.post("/items")
def add_item(body: NewItem):
    try:
        return standards_store.add_item(**body.model_dump())
    except (ValueError, re.error) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except standards_store.ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/items/{item_id}/retire")
def retire_added(item_id: str, body: Retire):
    try:
        return standards_store.retire_item(item_id, by=body.by)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such added item")
    except standards_store.ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/retire")
def retire_packaged(body: Retire):
    try:
        return standards_store.retire_item(family=body.family, section=body.section, text=body.text, by=body.by)
    except (ValueError, re.error) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except standards_store.ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
