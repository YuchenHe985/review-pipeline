"""Run configuration, loaded from TOML. Its hash is part of every output id."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    category: str
    reviews: Path
    meta: Path
    out: Path


@dataclass(frozen=True)
class IngestConfig:
    sample_rate: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class TrainConfig:
    test_share: float = 0.2
    valid_share: float = 0.1
    bootstrap: int = 1000
    seed: int = 42


@dataclass(frozen=True)
class SegmentConfig:
    min_reviews: int = 2
    k_min: int = 2
    k_max: int = 8
    silhouette_sample: int = 20000
    stability_seeds: int = 5
    seed: int = 42


@dataclass(frozen=True)
class Config:
    data: DataConfig
    ingest: IngestConfig = field(default_factory=IngestConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)

    def __post_init__(self) -> None:
        if not 0 < self.ingest.sample_rate <= 1:
            raise ValueError("ingest.sample_rate must be in (0, 1]")
        if not 0 < self.train.test_share < 1 or not 0 < self.train.valid_share < 1:
            raise ValueError("train shares must be in (0, 1)")
        if not 2 <= self.segment.k_min <= self.segment.k_max:
            raise ValueError("segment needs 2 <= k_min <= k_max")

    def digest(self, section: str) -> str:
        """Short hash of one section, so changing [train] does not invalidate the ingest output."""
        blob = json.dumps(asdict(getattr(self, section)), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load(path: str | Path) -> Config:
    path = Path(path)
    raw = tomllib.loads(path.read_text())
    base = path.parent
    d = raw["data"]
    data = DataConfig(
        category=d["category"],
        reviews=(base / d["reviews"]).resolve(),
        meta=(base / d["meta"]).resolve(),
        out=(base / d.get("out", "out")).resolve(),
    )
    return Config(
        data=data,
        ingest=IngestConfig(**raw.get("ingest", {})),
        train=TrainConfig(**raw.get("train", {})),
        segment=SegmentConfig(**raw.get("segment", {})),
    )
