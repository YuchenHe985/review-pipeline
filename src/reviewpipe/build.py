"""Stage 2: typed Parquet -> feature tables, in DuckDB SQL.

`examples` has one row per review for the helpfulness model. Every feature is computed only from
what was known when the review was posted. Product history uses window frames that end strictly
before the review's timestamp. The product's catalogue `average_rating`/`rating_number` are a 2023
snapshot that includes future reviews, so they are deliberately left out (see LEAKY_COLUMNS).

`reviewers` has one row per reviewer for segmentation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import duckdb

from . import storage
from .config import Config

log = logging.getLogger(__name__)
BUILD_VERSION = "2"
LEAKY_COLUMNS = (
    "catalog_average_rating",
    "catalog_rating_number",
    "average_rating",
    "rating_number",
    "helpful_vote",
    "y",
)

FEATURES = [
    "rating",
    "text_len",
    "title_len",
    "image_count",
    "verified",
    "age_days",
    "product_prior_reviews",
    "product_prior_avg_rating",
    "rating_deviation",
    "reviewer_prior_reviews",
    "price",
    "description_len",
    "feature_count",
    "product_images",
]

_SQL = """
CREATE TEMP TABLE raw AS SELECT * FROM read_parquet($reviews);
CREATE TEMP TABLE meta AS
  SELECT * FROM read_parquet($meta)
  QUALIFY row_number() OVER (PARTITION BY parent_asin ORDER BY rating_number DESC NULLS LAST) = 1;

-- the same review (reviewer, item, time) can appear more than once in the source: keep one
CREATE TEMP TABLE r AS
  SELECT * FROM raw
  QUALIFY row_number() OVER (PARTITION BY user_id, asin, ts ORDER BY helpful_vote DESC, text) = 1;

CREATE TEMP TABLE snapshot AS SELECT max(ts) AS snap_at FROM r;

CREATE TEMP TABLE examples AS
WITH hist AS (
  SELECT r.*,
    -- RANGE ... 1 ms PRECEDING: reviews posted in the same millisecond do not see each other
    count(*) OVER prod AS product_prior_reviews,
    avg(rating) OVER prod AS product_prior_avg_rating,
    count(*) OVER usr AS reviewer_prior_reviews
  FROM r
  WINDOW
    prod AS (PARTITION BY parent_asin ORDER BY ts
             RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL 1 MILLISECOND PRECEDING),
    usr AS (PARTITION BY user_id ORDER BY ts
            RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL 1 MILLISECOND PRECEDING)
)
SELECT
  h.user_id, h.asin, h.parent_asin, h.ts,
  h.helpful_vote,
  ln(1 + h.helpful_vote) AS y,
  h.rating,
  length(h.text) AS text_len,
  length(h.title) AS title_len,
  h.image_count,
  h.verified_purchase::INT AS verified,
  date_diff('day', h.ts, (SELECT snap_at FROM snapshot)) AS age_days,
  h.product_prior_reviews,
  h.product_prior_avg_rating,
  h.rating - h.product_prior_avg_rating AS rating_deviation,
  h.reviewer_prior_reviews,
  m.price,
  m.description_len,
  m.feature_count,
  m.image_count AS product_images,
  -- kept only for the leakage diagnostic in train.py; never a model feature
  m.average_rating AS catalog_average_rating,
  m.rating_number AS catalog_rating_number,
  m.parent_asin IS NOT NULL AS has_meta
FROM hist h LEFT JOIN meta m USING (parent_asin);

CREATE TEMP TABLE reviewers AS
SELECT
  user_id,
  count(*) AS n_reviews,
  avg(rating) AS avg_rating,
  coalesce(stddev_samp(rating), 0) AS rating_std,
  avg(ln(1 + length(text))) AS avg_log_text_len,
  avg(ln(1 + helpful_vote)) AS avg_log_helpful,
  avg(verified_purchase::INT) AS verified_share,
  avg((image_count > 0)::INT) AS image_share
FROM r GROUP BY user_id;
"""


def run(cfg: Config, ingest_dir: Path) -> Path:
    src = storage.read_manifest(ingest_dir)
    if not src:
        raise FileNotFoundError(f"{ingest_dir} is not a finished ingest output")
    build_id = storage.digest(src["batch_id"], BUILD_VERSION)
    final = cfg.data.out / "build" / build_id
    if storage.read_manifest(final):
        log.info("build %s already done; skipping", build_id)
        return final
    with storage.publish(final) as stage:
        con = duckdb.connect()
        try:
            con.execute("SET TimeZone = 'UTC'")
            params = {
                "reviews": str(ingest_dir / "reviews" / "reviews.parquet"),
                "meta": str(ingest_dir / "meta" / "meta.parquet"),
            }
            for stmt in filter(
                str.strip, re.split(r";\s*\n", _SQL)
            ):  # statements end a line; comments may hold ";"
                used = {k: v for k, v in params.items() if f"${k}" in stmt}
                con.execute(stmt, used)
            counts = {
                "reviews_in": con.sql("SELECT count(*) FROM raw").fetchone()[0],
                "duplicates_removed": con.sql(
                    "SELECT (SELECT count(*) FROM raw) - count(*) FROM r"
                ).fetchone()[0],
                "examples": con.sql("SELECT count(*) FROM examples").fetchone()[0],
                "examples_without_meta": con.sql(
                    "SELECT count(*) FROM examples WHERE NOT has_meta"
                ).fetchone()[0],
                "reviewers": con.sql("SELECT count(*) FROM reviewers").fetchone()[0],
            }
            for table in ("examples", "reviewers"):
                con.execute(
                    f"COPY (SELECT * FROM {table} ORDER BY ALL) TO '{stage / table}.parquet' "
                    "(FORMAT parquet, COMPRESSION zstd)"
                )
        finally:
            con.close()
        storage.write_manifest(
            stage,
            {
                "stage": "build",
                "build_id": build_id,
                "ingest_batch_id": src["batch_id"],
                "build_version": BUILD_VERSION,
                "features": FEATURES,
                "excluded_as_leaky": list(LEAKY_COLUMNS),
                "counts": counts,
            },
        )
    log.info("build %s: %s", build_id, counts)
    return final
