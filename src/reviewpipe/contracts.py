"""Data contracts for the raw JSONL records.

A record either becomes a typed row or is rejected with a stable reason code. Rejected records go to
quarantine with their line number, so nothing disappears silently and every count reconciles:
read = accepted + quarantined.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa

CONTRACT_VERSION = "1"

# Amazon Reviews '23 starts in 1996; anything outside this window is a unit or parsing error.
MIN_TS_MS = int(datetime(1995, 1, 1, tzinfo=UTC).timestamp() * 1000)
MAX_TS_MS = int(datetime(2030, 1, 1, tzinfo=UTC).timestamp() * 1000)

REVIEW_SCHEMA = pa.schema(
    [
        ("user_id", pa.string()),
        ("asin", pa.string()),
        ("parent_asin", pa.string()),
        ("ts", pa.timestamp("ms", tz="UTC")),
        ("rating", pa.int8()),
        ("title", pa.string()),
        ("text", pa.string()),
        ("image_count", pa.int16()),
        ("helpful_vote", pa.int32()),
        ("verified_purchase", pa.bool_()),
    ]
)

META_SCHEMA = pa.schema(
    [
        ("parent_asin", pa.string()),
        ("title", pa.string()),
        ("store", pa.string()),
        ("price", pa.float64()),
        ("description_len", pa.int32()),
        ("feature_count", pa.int16()),
        ("image_count", pa.int16()),
        # snapshot values that include FUTURE reviews; kept for reporting, never used as model features
        ("average_rating", pa.float32()),
        ("rating_number", pa.int32()),
    ]
)


class Rejected(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _id(rec: dict[str, Any], key: str) -> str:
    v = rec.get(key)
    if not isinstance(v, str) or not v.strip():
        raise Rejected(f"missing_{key}")
    return v.strip()


def _int(v: Any, reason: str) -> int:
    # bool is an int subclass in Python; a JSON true is not a count
    if isinstance(v, bool) or not isinstance(v, int | float) or (isinstance(v, float) and not v.is_integer()):
        raise Rejected(reason)
    return int(v)


def review(rec: dict[str, Any]) -> dict[str, Any]:
    rating = rec.get("rating")
    if isinstance(rating, bool) or not isinstance(rating, int | float) or rating not in (1, 2, 3, 4, 5):
        raise Rejected("bad_rating")
    ts = _int(rec.get("timestamp"), "bad_timestamp")
    if not MIN_TS_MS <= ts < MAX_TS_MS:
        raise Rejected("timestamp_out_of_range")
    votes = _int(rec.get("helpful_vote", 0), "bad_helpful_vote")
    if votes < 0:
        raise Rejected("bad_helpful_vote")
    verified = rec.get("verified_purchase", False)
    if not isinstance(verified, bool):
        raise Rejected("bad_verified_purchase")
    text, title = rec.get("text") or "", rec.get("title") or ""
    if not isinstance(text, str) or not isinstance(title, str):
        raise Rejected("bad_text")
    images = rec.get("images") or []
    return {
        "user_id": _id(rec, "user_id"),
        "asin": _id(rec, "asin"),
        "parent_asin": _id(rec, "parent_asin"),
        "ts": ts,
        "rating": int(rating),
        "title": title,
        "text": text,
        "image_count": len(images) if isinstance(images, list) else 0,
        "helpful_vote": votes,
        "verified_purchase": verified,
    }


_PRICE = re.compile(r"^\$?\s*(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{1,2}))?$")


def parse_price(v: Any) -> float | None:
    """Price as dollars, or None when unknown. Ranges ('$5 - $9') and junk ('None', 'from $3') are unknown."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        return float(v) if math.isfinite(v) and v >= 0 else None
    m = _PRICE.match(str(v).strip())
    if not m:
        return None
    return float(m.group(1).replace(",", "") + "." + (m.group(2) or "0"))


def _text_len(v: Any) -> int:
    if isinstance(v, list):
        return sum(len(x) for x in v if isinstance(x, str))
    return len(v) if isinstance(v, str) else 0


def meta(rec: dict[str, Any]) -> dict[str, Any]:
    avg = rec.get("average_rating")
    if avg is not None and (isinstance(avg, bool) or not isinstance(avg, int | float) or not 0 <= avg <= 5):
        raise Rejected("bad_average_rating")
    n = rec.get("rating_number")
    n = None if n is None else _int(n, "bad_rating_number")
    features = rec.get("features") or []
    images = rec.get("images") or []
    store = rec.get("store")
    title = rec.get("title")
    return {
        "parent_asin": _id(rec, "parent_asin"),
        "title": title if isinstance(title, str) else None,
        "store": store if isinstance(store, str) else None,
        "price": parse_price(rec.get("price")),
        "description_len": _text_len(rec.get("description")),
        "feature_count": len(features) if isinstance(features, list) else 0,
        "image_count": len(images) if isinstance(images, list) else 0,
        "average_rating": None if avg is None else float(avg),
        "rating_number": n,
    }
