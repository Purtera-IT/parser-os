from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ReviewQueueItem(BaseModel):
    item_id: str
    item_type: str
    target_id: str
    priority_score: float
    priority_reasons: list[str] = Field(default_factory=list)
    suggested_question: str
    family_or_type: str
    anchor_key: str | None = None
    risk_score: float | None = None
    ambiguity_score: float | None = None
    novelty_score: float | None = None
    created_at: str
    queue_tier: int = Field(default=50, ge=0, le=99)
    anchor_sort_key: int = Field(default=50, ge=0, le=99)


class ReviewQueueFile(BaseModel):
    items: list[ReviewQueueItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PacketReview(BaseModel):
    packet_id: str
    family: str
    anchor_key: str
    correct_packet: bool | None = None
    correct_governing_atom: bool | None = None
    correct_severity: bool | None = None
    should_be_status: str | None = None
    missing_evidence: str | None = None
    false_positive_reason: str | None = None
    reviewer_notes: str = ""
    reviewed_at: str


class PacketReviewFile(BaseModel):
    reviews: list[PacketReview] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
