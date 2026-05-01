from __future__ import annotations

import json
from pathlib import Path

from app.review.schemas import PacketReview, PacketReviewFile


def load_reviews(path: Path) -> dict[str, PacketReview]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        parsed = [PacketReview.model_validate(row) for row in payload]
    else:
        parsed_file = PacketReviewFile.model_validate(payload)
        parsed = parsed_file.reviews
    return {row.packet_id: row for row in parsed}


def save_reviews(path: Path, reviews: list[PacketReview], metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = PacketReviewFile(
        reviews=sorted(reviews, key=lambda review: review.packet_id),
        metadata=metadata or {},
    )
    path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def upsert_reviews(existing: dict[str, PacketReview], new_reviews: list[PacketReview]) -> list[PacketReview]:
    merged = dict(existing)
    for review in new_reviews:
        merged[review.packet_id] = review
    return sorted(merged.values(), key=lambda review: review.packet_id)
