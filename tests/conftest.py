from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from reviewpipe.config import Config, DataConfig, IngestConfig, SegmentConfig, TrainConfig

DAY_MS = 86_400_000
T0 = 1_500_000_000_000  # 2017-07-14


def write_gz(path: Path, lines: list[str | dict]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for x in lines:
            f.write((x if isinstance(x, str) else json.dumps(x)) + "\n")
    return path


def review(user: str, item: str, day: int, rating: int = 5, votes: int = 0, text: str = "fine", **kw) -> dict:
    return {
        "rating": float(rating),
        "title": "t",
        "text": text,
        "images": [],
        "asin": item,
        "parent_asin": item,
        "user_id": user,
        "timestamp": T0 + day * DAY_MS,
        "helpful_vote": votes,
        "verified_purchase": True,
        **kw,
    }


def product(item: str, price=9.99, avg=4.0, n=10) -> dict:
    return {
        "parent_asin": item,
        "title": f"Product {item}",
        "price": price,
        "average_rating": avg,
        "rating_number": n,
        "features": ["a"],
        "description": ["desc"],
        "images": [],
        "store": "S",
    }


def make_config(tmp: Path, reviews: Path, meta: Path, **ingest) -> Config:
    return Config(
        data=DataConfig(category="test", reviews=reviews, meta=meta, out=tmp / "out"),
        ingest=IngestConfig(**ingest),
        train=TrainConfig(bootstrap=50),
        segment=SegmentConfig(k_min=2, k_max=4, silhouette_sample=2000, stability_seeds=3),
    )


@pytest.fixture
def synthetic(tmp_path: Path) -> Config:
    """3,000 reviews over 600 days, 40 products, 600 reviewers. Helpful votes depend on text length and
    on verification (a learnable signal), plus noise."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(3000):
        n_words = int(rng.integers(1, 200))
        verified = bool(rng.random() < 0.7)
        lam = 0.02 * n_words * (1.5 if verified else 0.5)
        rows.append(
            review(
                f"U{i % 600}",
                f"P{int(rng.integers(0, 40))}",
                day=int(i * 600 / 3000),
                rating=int(rng.integers(1, 6)),
                votes=int(rng.poisson(lam)),
                text="word " * n_words,
                verified_purchase=verified,
            )
        )
    reviews = write_gz(tmp_path / "reviews.jsonl.gz", rows)
    meta = write_gz(tmp_path / "meta.jsonl.gz", [product(f"P{j}", price=1.0 + j) for j in range(40)])
    return make_config(tmp_path, reviews, meta)
