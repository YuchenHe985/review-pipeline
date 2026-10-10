"""Writes REPORT.md from the manifests and metrics of one pipeline run, so numbers are never typed by hand."""

from __future__ import annotations

import json
from pathlib import Path

from . import storage


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write(
    ingest_dir: Path, build_dir: Path, train_dir: Path | None, segment_dir: Path | None, out: Path
) -> Path:
    ing, bld = storage.read_manifest(ingest_dir), storage.read_manifest(build_dir)
    rc, mc = ing["counts"]["reviews"], ing["counts"]["meta"]
    lines = [
        f"# Pipeline report: {ing['category']}",
        "",
        f"ingest `{ing['batch_id']}` -> build `{bld['build_id']}`"
        + (f" -> train `{storage.read_manifest(train_dir)['run_id']}`" if train_dir else ""),
        "",
        "## Data",
        "",
        "| | read | accepted | quarantined |",
        "|---|---:|---:|---:|",
        f"| reviews | {rc['read']:,} | {rc['accepted']:,} | {rc.get('quarantined', 0):,} |",
        f"| products | {mc['read']:,} | {mc['accepted']:,} | {mc.get('quarantined', 0):,} |",
        "",
    ]
    reasons = {k.split(".", 1)[1]: v for k, v in {**rc, **mc}.items() if k.startswith("quarantined.")}
    if reasons:
        lines += ["Quarantine reasons: " + ", ".join(f"`{k}` {v:,}" for k, v in sorted(reasons.items())), ""]
    b = bld["counts"]
    lines += [
        f"Duplicates removed: {b['duplicates_removed']:,}. Examples: {b['examples']:,} "
        f"({b['examples_without_meta']:,} without product metadata). Reviewers: {b['reviewers']:,}.",
        "",
    ]
    if train_dir:
        m = json.loads((train_dir / "metrics.json").read_text())
        s = m["split"]
        lines += [
            "## Helpful-vote model (target: log(1 + helpful votes))",
            "",
            f"Time split: train {s['fit'] + s['valid']:,} reviews up to {s['train_until'][:10]}, "
            f"test {s['test']:,} reviews from {s['test_from'][:10]}. "
            f"{_pct(s['test_helpful_base_rate'])} of test reviews have at least one helpful vote.",
            "",
            "| model | RMSE | MAE | R² | Spearman | top-10% helpful share | lift |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, r in m["metrics"].items():
            lines.append(
                f"| {name} | {r['rmse']:.4f} | {r['mae']:.4f} | {r['r2']:.4f} | {r['spearman']:.3f} | "
                f"{_pct(r['top_decile_helpful_share'])} | {r['top_decile_lift']:.2f}× |"
            )
        bs = m["bootstrap_r2"]
        lines += ["", "Bootstrap 95% intervals for R² on the test set:", ""]
        for name, ci in bs.items():
            lines.append(f"- {name}: {ci['mean']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]")
        d = m.get("leakage_diagnostics")
        if d:
            lines += [
                "",
                "What shortcuts would have reported for the same XGBoost model (R², for comparison only):",
                "",
                f"- random 80/20 split instead of a time split: {d['random_split_r2']:.4f}",
                f"- time split plus the catalogue's snapshot rating columns: {d['time_split_with_snapshot_columns_r2']:.4f}",
                f"- the honest setup above: {m['metrics']['xgboost']['r2']:.4f}",
            ]
        top = list(m["feature_gain_share"].items())[:5]
        lines += ["", "Largest feature gain shares: " + ", ".join(f"{k} {_pct(v)}" for k, v in top), ""]
    if segment_dir:
        g = json.loads((segment_dir / "segments.json").read_text())
        lines += [
            "## Reviewer segments",
            "",
            f"{g['reviewers']:,} reviewers. k = {g['chosen_k']} by silhouette "
            f"({g['k_scan'][str(g['chosen_k'])]['silhouette']:.3f}). "
            f"Stability across seeds: mean ARI {g['stability_mean_ari']:.3f}, min {g['stability_min_ari']:.3f}.",
            "",
            "| segment | reviewers | avg rating | rating sd | avg log length | avg log helpful | verified | with images |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for p in g["profiles"]:
            lines.append(
                f"| {p['segment']} | {p['reviewers']:,} | {p['avg_rating']:.2f} | {p['rating_std']:.2f} | "
                f"{p['avg_log_text_len']:.2f} | {p['avg_log_helpful']:.2f} | {_pct(p['verified_share'])} | "
                f"{_pct(p['image_share'])} |"
            )
        lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out
