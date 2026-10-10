"""Command line: `reviewpipe all --config config.toml` runs every stage; each stage can run alone."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import build, config, ingest, report, segment, storage, train


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="reviewpipe")
    p.add_argument("stage", choices=["ingest", "build", "train", "segment", "report", "all"])
    p.add_argument("--config", default="config.toml")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = config.load(args.config)
    removed = storage.sweep_staging(cfg.data.out)
    if removed:
        logging.info("removed %d staging folders left by an interrupted run", removed)

    ingest_dir = ingest.run(cfg)
    if args.stage == "ingest":
        return 0
    build_dir = build.run(cfg, ingest_dir)
    if args.stage == "build":
        return 0
    train_dir = train.run(cfg, build_dir) if args.stage in ("train", "report", "all") else None
    segment_dir = segment.run(cfg, build_dir) if args.stage in ("segment", "report", "all") else None
    if args.stage in ("report", "all"):
        out = report.write(ingest_dir, build_dir, train_dir, segment_dir, Path(cfg.data.out) / "REPORT.md")
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
