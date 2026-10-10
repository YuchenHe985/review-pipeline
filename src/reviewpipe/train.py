"""Stage 3: helpful-vote model with a time-based split, baselines and bootstrap intervals.

The question a marketplace asks: when a review is posted, can we tell whether readers will find it
helpful, so the review can be surfaced early? So the split is by time: the model trains on older
reviews and is scored on the newest ones, as it would be used in production. A random split would
let it learn from the future.
"""

from __future__ import annotations

import json
import logging
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import build, storage
from .config import Config

log = logging.getLogger(__name__)


def time_split(df: pd.DataFrame, test_share: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Everything before the cutoff trains, everything at or after it tests. No timestamp is on both sides."""
    cutoff = df["ts"].quantile(1 - test_share)
    train, test = df[df["ts"] < cutoff], df[df["ts"] >= cutoff]
    if train.empty or test.empty:
        raise ValueError("time split produced an empty side; not enough distinct timestamps")
    return train, test


def _scores(y: np.ndarray, p: np.ndarray, helpful: np.ndarray) -> dict[str, float]:
    order = np.argsort(-p, kind="stable")
    top = order[: max(1, len(p) // 10)]
    base = helpful.mean()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
        "spearman": float(spearmanr(y, p).statistic) if np.ptp(p) > 0 else 0.0,
        # share of reviews with >= 1 helpful vote among the top 10% the model would surface, vs the base rate
        "top_decile_helpful_share": float(helpful[top].mean()),
        "top_decile_lift": float(helpful[top].mean() / base) if base > 0 else float("nan"),
    }


def bootstrap(
    y: np.ndarray, preds: dict[str, np.ndarray], n: int, seed: int, metric: str = "r2"
) -> dict[str, dict[str, float]]:
    """Paired bootstrap over test reviews: 95% intervals for each model and for the gap between them."""
    rng = np.random.default_rng(seed)
    names = list(preds)
    draws: dict[str, list[float]] = {k: [] for k in names}
    fn = r2_score if metric == "r2" else (lambda a, b: -np.sqrt(mean_squared_error(a, b)))
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        for k in names:
            draws[k].append(fn(y[idx], preds[k][idx]))
    out = {k: _ci(np.array(v)) for k, v in draws.items()}
    if "xgboost" in draws and "ridge" in draws:
        out["xgboost_minus_ridge"] = _ci(np.array(draws["xgboost"]) - np.array(draws["ridge"]))
    return out


def _ci(a: np.ndarray) -> dict[str, float]:
    lo, hi = np.percentile(a, [2.5, 97.5])
    return {"mean": float(a.mean()), "lo": float(lo), "hi": float(hi)}


def _fixed_xgb(n_trees: int, seed: int) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=n_trees,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        tree_method="hist",
        random_state=seed,
        n_jobs=0,
    )


def leakage_diagnostics(
    df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, features: list[str], n_trees: int, t: Any
) -> dict[str, float]:
    """How much two common shortcuts would have overstated the result. Same model and tree count each time;
    only the evaluation design changes."""
    out: dict[str, float] = {}
    # 1. a random split: training sees reviews posted after the ones it is tested on
    shuffled = df.sample(frac=1.0, random_state=t.seed)
    cut = int(len(shuffled) * (1 - t.test_share))
    a, b = shuffled.iloc[:cut], shuffled.iloc[cut:]
    m = _fixed_xgb(n_trees, t.seed).fit(a[features].astype("float64"), a["y"])
    out["random_split_r2"] = float(r2_score(b["y"], m.predict(b[features].astype("float64"))))
    # 2. the catalogue's average rating and rating count: a 2023 snapshot that already counts future reviews
    leaky = features + ["catalog_average_rating", "catalog_rating_number"]
    m = _fixed_xgb(n_trees, t.seed).fit(train[leaky].astype("float64"), train["y"])
    out["time_split_with_snapshot_columns_r2"] = float(
        r2_score(test["y"], m.predict(test[leaky].astype("float64")))
    )
    return out


def run(cfg: Config, build_dir: Path) -> Path:
    src = storage.read_manifest(build_dir)
    if not src:
        raise FileNotFoundError(f"{build_dir} is not a finished build output")
    features = list(src["features"])
    leaked = sorted(set(features) & set(build.LEAKY_COLUMNS))
    if leaked:
        raise ValueError(f"leaky columns in the feature list: {leaked}")
    t = cfg.train
    run_id = storage.digest(src["build_id"], cfg.digest("train"), xgb.__version__)
    final = cfg.data.out / "runs" / f"train-{run_id}"
    if storage.read_manifest(final):
        log.info("train run %s already done; skipping", run_id)
        return final

    df = pd.read_parquet(build_dir / "examples.parquet")
    train, test = time_split(df, t.test_share)
    fit, valid = time_split(train, t.valid_share)
    X = {k: d[features].astype("float64") for k, d in (("fit", fit), ("valid", valid), ("test", test))}
    y_test = test["y"].to_numpy()
    helpful_test = (test["helpful_vote"] > 0).to_numpy()

    preds: dict[str, np.ndarray] = {}
    # 1. what you get without a model
    preds["train_mean"] = np.full(len(test), train["y"].mean())
    # 2. a linear model on the same features
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(pd.concat([X["fit"], X["valid"]]), pd.concat([fit["y"], valid["y"]]))
    preds["ridge"] = ridge.predict(X["test"])
    # 3. gradient-boosted trees; early stopping on the newest slice of the training period
    model = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        tree_method="hist",
        early_stopping_rounds=100,
        random_state=t.seed,
        n_jobs=0,
    )
    model.fit(X["fit"], fit["y"], eval_set=[(X["valid"], valid["y"])], verbose=False)
    preds["xgboost"] = model.predict(X["test"])

    metrics: dict[str, Any] = {k: _scores(y_test, p, helpful_test) for k, p in preds.items()}
    diagnostics = leakage_diagnostics(df, train, test, features, model.best_iteration + 1, t)
    importance = model.get_booster().get_score(importance_type="gain")
    total = sum(importance.values()) or 1.0
    result = {
        "split": {
            "fit": len(fit),
            "valid": len(valid),
            "test": len(test),
            "train_until": str(train["ts"].max()),
            "test_from": str(test["ts"].min()),
            "test_helpful_base_rate": float(helpful_test.mean()),
        },
        "metrics": metrics,
        "bootstrap_r2": bootstrap(y_test, preds, t.bootstrap, t.seed),
        "best_iteration": int(model.best_iteration),
        "leakage_diagnostics": diagnostics,
        "feature_gain_share": {
            k: round(v / total, 4) for k, v in sorted(importance.items(), key=lambda kv: -kv[1])
        },
    }
    with storage.publish(final) as stage:
        model.save_model(stage / "model.ubj")
        (stage / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        storage.write_manifest(
            stage,
            {
                "stage": "train",
                "run_id": run_id,
                "build_id": src["build_id"],
                "features": features,
                "config": t.__dict__,
                "environment": {
                    "python": platform.python_version(),
                    "xgboost": xgb.__version__,
                    "scikit-learn": sklearn.__version__,
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                },
            },
        )
    log.info("train %s: %s", run_id, json.dumps(metrics))
    return final
