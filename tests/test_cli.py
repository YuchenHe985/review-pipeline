import pytest

from reviewpipe import cli, config


def _toml(tmp_path, synthetic, extra=""):
    p = tmp_path / "run.toml"
    p.write_text(
        f"""
[data]
category = "test"
reviews = "{synthetic.data.reviews}"
meta = "{synthetic.data.meta}"
out = "cli-out"

[train]
bootstrap = 20

[segment]
k_max = 3
silhouette_sample = 1000
stability_seeds = 2
{extra}
"""
    )
    return p


def test_all_stages_run_from_a_config_file(tmp_path, synthetic, capsys):
    p = _toml(tmp_path, synthetic)
    assert cli.main(["all", "--config", str(p)]) == 0
    report = tmp_path / "cli-out" / "REPORT.md"
    assert capsys.readouterr().out.strip() == str(report)
    assert "Helpful-vote model" in report.read_text()
    # a second run reuses every stage's output
    runs = sorted((tmp_path / "cli-out" / "runs").iterdir())
    assert cli.main(["all", "--config", str(p)]) == 0
    assert sorted((tmp_path / "cli-out" / "runs").iterdir()) == runs


def test_relative_paths_resolve_against_the_config_file(tmp_path, synthetic):
    cfg = config.load(_toml(tmp_path, synthetic))
    assert cfg.data.out == (tmp_path / "cli-out").resolve()


@pytest.mark.parametrize(
    "extra",
    ["[ingest]\nsample_rate = 0", "[ingest]\nsample_rate = 1.5"],
)
def test_impossible_settings_are_refused(tmp_path, synthetic, extra):
    with pytest.raises(ValueError):
        config.load(_toml(tmp_path, synthetic, extra))
