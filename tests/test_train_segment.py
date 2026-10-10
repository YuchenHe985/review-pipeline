import json

import numpy as np
import pandas as pd
import pytest

from reviewpipe import build, ingest, report, segment, train


@pytest.fixture
def built(synthetic):
    return synthetic, build.run(synthetic, ingest.run(synthetic))


def test_time_split_never_puts_a_timestamp_on_both_sides():
    df = pd.DataFrame({"ts": pd.to_datetime([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], unit="D")})
    a, b = train.time_split(df, 0.3)
    assert a["ts"].max() < b["ts"].min()
    assert len(a) + len(b) == len(df)


def test_model_beats_the_baselines_and_writes_a_complete_run(built):
    cfg, out = built
    run = train.run(cfg, out)
    m = json.loads((run / "metrics.json").read_text())
    assert m["metrics"]["xgboost"]["r2"] > m["metrics"]["train_mean"]["r2"]
    assert m["metrics"]["xgboost"]["spearman"] > 0.3
    assert m["metrics"]["xgboost"]["top_decile_lift"] > 1.2
    ci = m["bootstrap_r2"]["xgboost"]
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert {"random_split_r2", "time_split_with_snapshot_columns_r2"} <= set(m["leakage_diagnostics"])
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["environment"]["xgboost"] and (run / "model.ubj").stat().st_size > 0
    assert train.run(cfg, out) == run  # same inputs and settings: reused, not retrained


def test_a_leaky_feature_list_is_refused(built, monkeypatch):
    cfg, out = built
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["features"].append("catalog_average_rating")
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leaky"):
        train.run(cfg, out)


def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.normal(size=500)
    p = {"ridge": y + rng.normal(scale=0.5, size=500), "xgboost": y + rng.normal(scale=0.3, size=500)}
    ci = train.bootstrap(y, p, n=200, seed=0)
    assert ci["xgboost_minus_ridge"]["lo"] > 0  # the clearly better model is detected


def test_segmentation_picks_k_in_range_and_reports_stability(built):
    cfg, out = built
    run = segment.run(cfg, out)
    g = json.loads((run / "segments.json").read_text())
    assert cfg.segment.k_min <= g["chosen_k"] <= cfg.segment.k_max
    assert 0 <= g["stability_min_ari"] <= g["stability_mean_ari"] <= 1
    assert sum(p["reviewers"] for p in g["profiles"]) == g["reviewers"]


def test_report_is_generated_from_the_run_outputs(built, tmp_path):
    cfg, out = built
    ing = ingest.run(cfg)
    md = report.write(
        ing, out, train.run(cfg, out), segment.run(cfg, out), tmp_path / "REPORT.md"
    ).read_text()
    assert "| xgboost |" in md and "Reviewer segments" in md and "random 80/20 split" in md
