"""Stage 4: reviewer segmentation with KMeans, with k chosen by silhouette and checked for stability.

The 2026 coursework version fixed k by eye from an elbow plot. Here k is chosen by silhouette on a
fixed sample, and the chosen clustering is refit with several seeds: a segmentation that changes
with the seed should not drive decisions (mean pairwise adjusted Rand index is reported).
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from . import storage
from .config import Config

log = logging.getLogger(__name__)
COLUMNS = [
    "avg_rating",
    "rating_std",
    "avg_log_text_len",
    "avg_log_helpful",
    "verified_share",
    "image_share",
]


def run(cfg: Config, build_dir: Path) -> Path:
    src = storage.read_manifest(build_dir)
    if not src:
        raise FileNotFoundError(f"{build_dir} is not a finished build output")
    s = cfg.segment
    run_id = storage.digest(src["build_id"], cfg.digest("segment"))
    final = cfg.data.out / "runs" / f"segment-{run_id}"
    if storage.read_manifest(final):
        return final

    df = pd.read_parquet(build_dir / "reviewers.parquet")
    df = df[df["n_reviews"] >= s.min_reviews].reset_index(drop=True)
    if len(df) < s.k_max * 10:
        raise ValueError(f"only {len(df)} reviewers with >= {s.min_reviews} reviews")
    X = StandardScaler().fit_transform(df[COLUMNS].to_numpy())
    rng = np.random.default_rng(s.seed)
    sample = rng.choice(len(X), size=min(s.silhouette_sample, len(X)), replace=False)

    scan = {}
    for k in range(s.k_min, s.k_max + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=s.seed).fit_predict(X)
        scan[k] = {
            "silhouette": float(silhouette_score(X[sample], labels[sample])),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
        }
    best_k = max(scan, key=lambda k: scan[k]["silhouette"])

    runs = [
        KMeans(n_clusters=best_k, n_init=10, random_state=s.seed + i).fit_predict(X)
        for i in range(s.stability_seeds)
    ]
    ari = [adjusted_rand_score(a, b) for a, b in itertools.combinations(runs, 2)]
    df["segment"] = runs[0]
    profile = df.groupby("segment")[COLUMNS + ["n_reviews"]].mean().round(3)
    profile["reviewers"] = df.groupby("segment").size()

    result = {
        "reviewers": len(df),
        "k_scan": scan,
        "chosen_k": best_k,
        "stability_mean_ari": float(np.mean(ari)) if ari else 1.0,
        "stability_min_ari": float(np.min(ari)) if ari else 1.0,
        "profiles": json.loads(profile.reset_index().to_json(orient="records")),
    }
    with storage.publish(final) as stage:
        df[["user_id", "segment"]].to_parquet(stage / "assignments.parquet", index=False)
        (stage / "segments.json").write_text(json.dumps(result, indent=2) + "\n")
        storage.write_manifest(
            stage,
            {"stage": "segment", "run_id": run_id, "build_id": src["build_id"], "config": s.__dict__},
        )
    log.info("segment %s: k=%d, mean ARI %.3f", run_id, best_k, result["stability_mean_ari"])
    return final
