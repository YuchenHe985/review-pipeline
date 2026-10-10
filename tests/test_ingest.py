import pyarrow.parquet as pq
import pytest

from reviewpipe import ingest, storage
from tests.conftest import make_config, product, review, write_gz


@pytest.fixture
def small(tmp_path):
    lines = [review(f"U{i}", "P1", day=i) for i in range(200)]
    lines[10] = "{not json"
    lines[20] = '["an", "array"]'
    lines[30] = review("U30", "P1", day=1, rating=0)
    r = write_gz(tmp_path / "r.jsonl.gz", lines)
    m = write_gz(tmp_path / "m.jsonl.gz", [product("P1")])
    return tmp_path, r, m


def test_every_line_is_accounted_for_and_rejects_keep_their_line_number(small):
    tmp, r, m = small
    out = ingest.run(make_config(tmp, r, m), max_quarantine_share=0.05)
    c = storage.read_manifest(out)["counts"]["reviews"]
    assert c["read"] == 200 and c["accepted"] == 197 and c["quarantined"] == 3
    assert c["quarantined.malformed_json"] == 1 and c["quarantined.not_an_object"] == 1
    q = pq.read_table(out / "reviews" / "quarantine.parquet").to_pylist()
    assert {(x["line_no"], x["reason"]) for x in q} == {
        (11, "malformed_json"),
        (21, "not_an_object"),
        (31, "bad_rating"),
    }


def test_too_many_rejects_stop_the_run_and_publish_nothing(small):
    tmp, r, m = small
    cfg = make_config(tmp, r, m)
    with pytest.raises(RuntimeError, match="quarantined"):
        ingest.run(cfg, max_quarantine_share=0.01)
    assert not (cfg.data.out / "ingest").exists() or not any((cfg.data.out / "ingest").iterdir())


def test_rerunning_on_the_same_input_reuses_the_output(small, monkeypatch):
    tmp, r, m = small
    cfg = make_config(tmp, r, m)
    first = ingest.run(cfg, max_quarantine_share=0.05)
    monkeypatch.setattr(ingest, "_convert", lambda *a, **k: pytest.fail("should not re-read the input"))
    assert ingest.run(cfg, max_quarantine_share=0.05) == first


def test_changed_input_gets_a_new_output(small):
    tmp, r, m = small
    cfg = make_config(tmp, r, m)
    first = ingest.run(cfg, max_quarantine_share=0.05)
    write_gz(r, [review("Z", "P1", day=0)])
    assert ingest.run(cfg, max_quarantine_share=0.05) != first


def test_a_crash_mid_write_leaves_no_partial_output(small, monkeypatch):
    tmp, r, m = small
    cfg = make_config(tmp, r, m)
    real = ingest._convert
    calls = []

    def crash_on_meta(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("disk full")
        return real(*a, **k)

    monkeypatch.setattr(ingest, "_convert", crash_on_meta)
    with pytest.raises(OSError):
        ingest.run(cfg, max_quarantine_share=0.05)
    leftovers = list((cfg.data.out / "ingest").iterdir())
    assert leftovers == []


def test_sampling_is_by_reviewer_and_repeatable(tmp_path):
    lines = [review(f"U{i % 50}", f"P{i}", day=i) for i in range(500)]
    r = write_gz(tmp_path / "r.jsonl.gz", lines)
    m = write_gz(tmp_path / "m.jsonl.gz", [product("P1")])
    out = ingest.run(make_config(tmp_path, r, m, sample_rate=0.3, seed=7))
    users = pq.read_table(out / "reviews" / "reviews.parquet").column("user_id").to_pylist()
    kept = set(users)
    assert 5 < len(kept) < 30
    assert all(users.count(u) == 10 for u in kept)  # every review of a kept reviewer, none of the others
    assert kept == {f"U{i}" for i in range(50) if ingest.keep_reviewer(f"U{i}", 0.3, 7)}


def test_leftover_staging_folders_are_swept(tmp_path):
    (tmp_path / "ingest" / ".staging-x-1").mkdir(parents=True)
    assert storage.sweep_staging(tmp_path) == 1
    assert not any((tmp_path / "ingest").iterdir())
