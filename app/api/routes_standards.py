"""PurTera SOW standards, editable without a deploy (app/core/standards_store.py).

    GET  /standards                      merged standards (packaged + approved items), same shape as
                                         purtera_standards.json; "items" = every overlay item with its status
    POST /standards/items                {family, list, text, by, deal_id?, source_line?, when?, unless?, note?}
                                         -> a "proposed" item (never in a SOW until approved)
    POST /standards/items/{id}/approve   {by} -> approved: written into SOWs from the next compile
    POST /standards/items/{id}/retire    {by}
    POST /standards/retire               {by, family, list, text}: switch off a packaged item

SowSmith's "Add as PurTera standard" posts here; each item keeps who, when and the deal it came from.
"""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core import standards_store

router = APIRouter(prefix="/standards", tags=["standards"])


class NewItem(BaseModel):
    family: str
    list: str
    text: str
    by: str
    deal_id: str = ""
    source_line: str = ""
    when: str = ""
    unless: str = ""
    note: str = ""


class By(BaseModel):
    by: str


class RetirePackaged(BaseModel):
    by: str
    family: str
    list: str
    text: str


def _guard(fn):
    try:
        return fn()
    except KeyError:
        raise HTTPException(status_code=404, detail="no such item")
    except (ValueError, re.error) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except standards_store.ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("")
def get_standards():
    overlay, _ = standards_store.read_overlay()
    return {"standards": standards_store.load(), "lists": list(standards_store.SECTIONS),
            "items": overlay.get("items") or []}


@router.post("/items")
def add_item(b: NewItem):
    return _guard(lambda: standards_store.add_item(
        family=b.family, section=b.list, text=b.text, added_by=b.by, when=b.when, unless=b.unless,
        source_deal=b.deal_id, note=b.note, source_line=b.source_line, status="proposed"))


@router.post("/items/{item_id}/approve")
def approve(item_id: str, b: By):
    return _guard(lambda: standards_store.approve_item(item_id, by=b.by))


@router.post("/items/{item_id}/retire")
def retire_added(item_id: str, b: By):
    return _guard(lambda: standards_store.retire_item(item_id, by=b.by))


@router.post("/retire")
def retire_packaged(b: RetirePackaged):
    return _guard(lambda: standards_store.retire_item(family=b.family, section=b.list, text=b.text, by=b.by))
