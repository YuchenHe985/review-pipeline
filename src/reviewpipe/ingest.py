"""Stage 1: raw JSONL.gz -> validated, typed Parquet plus a quarantine of rejected lines.

Idempotent: the output id is a hash of the input files' content, the contract version and the
sampling settings. Running again on the same input does nothing and returns the existing output.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from . import contracts, storage
from .config import Config

log = logging.getLogger(__name__)
BATCH_ROWS = 50_000
QUARANTINE_SCHEMA = pa.schema([("line_no", pa.int64()), ("reason", pa.string()), ("raw", pa.string())])


def keep_reviewer(user_id: str, rate: float, seed: int) -> bool:
    """Deterministic sampling by reviewer: the same reviewer is in or out on every run and machine,
    and all of a kept reviewer's reviews stay together (per-reviewer features need that)."""
    if rate >= 1:
        return True
    h = hashlib.blake2b(f"{seed}:{user_id}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64 < rate


def _lines(path: Path) -> Iterator[tuple[int, str]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as f:
        yield from enumerate(f, start=1)


def _convert(
    path: Path,
    out: Path,
    schema: pa.Schema,
    validate: Callable[[dict[str, Any]], dict[str, Any]],
    keep: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    with (
        pq.ParquetWriter(out / f"{out.name}.parquet", schema, compression="zstd") as good_w,
        pq.ParquetWriter(out / "quarantine.parquet", QUARANTINE_SCHEMA, compression="zstd") as bad_w,
    ):
        for line_no, line in _lines(path):
            counts["read"] += 1
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise contracts.Rejected("not_an_object")
                row = validate(rec)
            except json.JSONDecodeError:
                reason = "malformed_json"
            except contracts.Rejected as e:
                reason = e.reason
            else:
                if keep(row):
                    rows.append(row)
                    counts["accepted"] += 1
                else:
                    counts["sampled_out"] += 1
                if len(rows) >= BATCH_ROWS:
                    good_w.write_table(pa.Table.from_pylist(rows, schema))
                    rows.clear()
                continue
            counts["quarantined"] += 1
            counts[f"quarantined.{reason}"] += 1
            bad.append({"line_no": line_no, "reason": reason, "raw": line[:2000]})
            if len(bad) >= BATCH_ROWS:
                bad_w.write_table(pa.Table.from_pylist(bad, QUARANTINE_SCHEMA))
                bad.clear()
        if rows:
            good_w.write_table(pa.Table.from_pylist(rows, schema))
        if bad:
            bad_w.write_table(pa.Table.from_pylist(bad, QUARANTINE_SCHEMA))
    assert counts["read"] == counts["accepted"] + counts["quarantined"] + counts["sampled_out"]
    return dict(counts)


def run(cfg: Config, max_quarantine_share: float = 0.01) -> Path:
    inputs = {"reviews": cfg.data.reviews, "meta": cfg.data.meta}
    hashes = {k: storage.file_sha256(p) for k, p in inputs.items()}
    batch_id = storage.digest(
        hashes["reviews"], hashes["meta"], contracts.CONTRACT_VERSION, cfg.digest("ingest")
    )
    final = cfg.data.out / "ingest" / batch_id
    if storage.read_manifest(final):
        log.info("ingest %s already done; skipping", batch_id)
        return final

    rate, seed = cfg.ingest.sample_rate, cfg.ingest.seed
    with storage.publish(final) as stage:
        (stage / "reviews").mkdir()
        (stage / "meta").mkdir()
        review_counts = _convert(
            inputs["reviews"],
            stage / "reviews",
            contracts.REVIEW_SCHEMA,
            contracts.review,
            lambda r: keep_reviewer(r["user_id"], rate, seed),
        )
        meta_counts = _convert(
            inputs["meta"], stage / "meta", contracts.META_SCHEMA, contracts.meta, lambda r: True
        )
        for name, c in (("reviews", review_counts), ("meta", meta_counts)):
            share = c.get("quarantined", 0) / max(c["read"], 1)
            if share > max_quarantine_share:
                # a contract that suddenly rejects many rows means the source changed: stop, do not train on it
                raise RuntimeError(
                    f"{name}: {share:.1%} of lines quarantined (limit {max_quarantine_share:.0%})"
                )
        storage.write_manifest(
            stage,
            {
                "stage": "ingest",
                "batch_id": batch_id,
                "category": cfg.data.category,
                "inputs": {k: {"path": str(p), "sha256": hashes[k]} for k, p in inputs.items()},
                "contract_version": contracts.CONTRACT_VERSION,
                "config": {"sample_rate": rate, "seed": seed},
                "counts": {"reviews": review_counts, "meta": meta_counts},
            },
        )
    log.info("ingest %s: reviews %s, meta %s", batch_id, review_counts, meta_counts)
    return final
