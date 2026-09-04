import csv

from experiments.aggregate_splits import _read_test_rows, _summarise


def test_summarise_reports_spread_not_just_the_mean() -> None:
    stats = _summarise([0.5, 0.6, 0.7, 0.8, 0.9])
    assert stats["n"] == 5
    assert stats["mean"] == 0.7
    assert stats["min"] == 0.5 and stats["max"] == 0.9
    assert stats["std"] > 0
    assert stats["p10"] <= stats["mean"] <= stats["p90"]


def test_summarise_handles_a_single_split() -> None:
    stats = _summarise([0.42])
    assert stats["std"] == 0.0
    assert stats["mean"] == stats["min"] == stats["max"] == 0.42


def test_only_test_split_rows_are_aggregated(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    fields = ["split", "target", "uq_method", "model", "prefix",
              "auroc", "auprc", "prr_at_0_5", "brier", "nll", "ece"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({**dict.fromkeys(fields, ""), "split": "test",
                         "target": "combined", "uq_method": "perplexity",
                         "model": "bayes_state", "auroc": "0.8", "brier": "0.1"})
        writer.writerow({**dict.fromkeys(fields, ""), "split": "calibration",
                         "target": "combined", "uq_method": "perplexity",
                         "model": "bayes_state", "auroc": "0.99"})

    rows = _read_test_rows(path)
    assert list(rows) == [("combined", "perplexity", "bayes_state", "")]
    values = rows[("combined", "perplexity", "bayes_state", "")]
    assert values["auroc"] == 0.8
    assert values["nll"] is None  # empty cells stay missing rather than becoming 0
