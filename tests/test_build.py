import pandas as pd

from reviewpipe import build, ingest
from tests.conftest import make_config, product, review, write_gz


def _examples(tmp_path, lines, meta_lines):
    r = write_gz(tmp_path / "r.jsonl.gz", lines)
    m = write_gz(tmp_path / "m.jsonl.gz", meta_lines)
    cfg = make_config(tmp_path, r, m)
    out = build.run(cfg, ingest.run(cfg))
    return pd.read_parquet(out / "examples.parquet").sort_values(["ts", "user_id"]).reset_index(
        drop=True
    ), out


def test_product_history_only_uses_earlier_reviews(tmp_path):
    lines = [
        review("A", "P1", day=0, rating=5),
        review("B", "P1", day=1, rating=1),
        review("C", "P1", day=1, rating=3),  # same moment as B: neither sees the other
        review("D", "P1", day=2, rating=4),
    ]
    ex, _ = _examples(tmp_path, lines, [product("P1")])
    by = ex.set_index("user_id")
    assert by.loc["A", "product_prior_reviews"] == 0 and pd.isna(by.loc["A", "product_prior_avg_rating"])
    assert by.loc["B", "product_prior_reviews"] == 1 and by.loc["B", "product_prior_avg_rating"] == 5
    assert by.loc["C", "product_prior_reviews"] == 1
    assert by.loc["D", "product_prior_reviews"] == 3 and by.loc["D", "product_prior_avg_rating"] == 3
    assert by.loc["B", "rating_deviation"] == -4


def test_reviewer_history_and_age(tmp_path):
    lines = [review("A", "P1", day=0), review("A", "P2", day=5), review("B", "P1", day=10)]
    ex, _ = _examples(tmp_path, lines, [product("P1"), product("P2")])
    assert ex["reviewer_prior_reviews"].tolist() == [0, 1, 0]
    assert ex["age_days"].tolist() == [10, 5, 0]


def test_duplicates_are_removed_and_counted_and_missing_metadata_is_kept(tmp_path):
    lines = [review("A", "P1", day=0, votes=1), review("A", "P1", day=0, votes=4), review("B", "P9", day=1)]
    ex, out = _examples(tmp_path, lines, [product("P1")])
    counts = __import__("json").loads((out / "manifest.json").read_text())["counts"]
    assert counts["duplicates_removed"] == 1 and counts["examples_without_meta"] == 1
    assert ex.loc[ex.user_id == "A", "helpful_vote"].item() == 4  # the most complete copy wins
    assert pd.isna(ex.loc[ex.user_id == "B", "price"].item())


def test_no_feature_is_a_snapshot_or_target_column():
    assert not set(build.FEATURES) & set(build.LEAKY_COLUMNS)
