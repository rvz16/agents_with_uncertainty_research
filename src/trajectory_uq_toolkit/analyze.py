from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .aggregation import aggregate
from .bayes import BinaryBayes, ContinuousBayes, CriticBayes, DoubleBinaryBayes
from .metrics import fit_platt, metric_values
from .schema import validate_episode


DEFAULT_HIGHER_IS_UNCERTAIN = {
    "perplexity": True,
    "mean_token_entropy": True,
    "num_tokens": True,
    "sum_logprob": False,
    "mean_token_logprob": False,
    "sequence_probability": False,
    "kl_to_uniform": False,
    "self_certainty": False,
    "verbalized_confidence": False,
    "tool_success_rate": False,
}
BINARY_MODES = ("quantile", "sep", "lr_pos", "lr_neg", "double")


def _read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = validate_episode(json.loads(line))
            if record["episode_id"] in seen:
                raise ValueError(f"duplicate episode_id at line {line_number}: {record['episode_id']}")
            seen.add(record["episode_id"])
            records.append(record)
    if len(records) < 4:
        raise ValueError("analysis requires at least four episodes")
    return records


def _split(records: list[dict[str, Any]], fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        records,
        key=lambda row: hashlib.sha256(f"{seed}:{row['episode_id']}".encode()).hexdigest(),
    )
    size = min(len(ordered) - 1, max(1, round(len(ordered) * fraction)))
    return ordered[:size], ordered[size:]


def _sequence(record: dict[str, Any], signal: str) -> list[float]:
    values = [generation["signals"].get(signal) for generation in record["generations"]]
    sequence = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not sequence and signal in record["features"]:
        sequence = [float(record["features"][signal])]
    return sequence


def _metric_row(signal: str, model: str, labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    return {"signal": signal, "model": model, **metric_values(labels, probabilities)}


def _common_episode_critics(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return []
    return sorted(set.intersection(*(set(record["critics"]) for record in records)))


def _common_step_critics(records: list[dict[str, Any]]) -> list[str]:
    rows = [generation["critics"] for record in records for generation in record["generations"]]
    return sorted(set.intersection(*(set(row) for row in rows))) if rows else []


def _parse_directions(raw: list[str]) -> dict[str, bool]:
    directions = dict(DEFAULT_HIGHER_IS_UNCERTAIN)
    for item in raw:
        try:
            name, direction = item.split("=", 1)
        except ValueError as exc:
            raise ValueError("--signal-direction must be NAME=uncertain|confident") from exc
        if direction not in {"uncertain", "confident"}:
            raise ValueError("signal direction must be uncertain or confident")
        directions[name] = direction == "uncertain"
    return directions


def _parse_critic_sets(raw: list[str], common: list[str]) -> dict[str, list[str]]:
    if raw:
        result = {}
        for item in raw:
            name, critics = item.split("=", 1)
            selected = [critic for critic in critics.split(",") if critic]
            missing = sorted(set(selected) - set(common))
            if missing:
                raise ValueError(f"critic set {name!r} has unavailable critics: {missing}")
            result[name] = selected
        return result
    result = {f"only_{name}": [name] for name in common}
    if common:
        result["all"] = common
    without_judge = [name for name in common if name != "llm_judge_pass"]
    if len(without_judge) != len(common) and without_judge:
        result["without_judge"] = without_judge
    return result


def analyze(
    *,
    episodes_path: Path,
    output_dir: Path,
    calibration_fraction: float,
    seed: int,
    tempered_lambda: float,
    signal_directions: dict[str, bool],
    critic_set_specs: list[str],
) -> list[dict[str, Any]]:
    records = _read_records(episodes_path)
    train, test = _split(records, calibration_fraction, seed)
    labels_train = [record["success"] for record in train]
    labels_test = [record["success"] for record in test]
    prior = min(1.0 - 1e-6, max(1e-6, sum(labels_train) / len(labels_train)))
    metric_rows = [_metric_row("all", "prior", labels_test, [prior] * len(test))]
    parameters: list[dict[str, Any]] = []

    all_records = train + test
    signal_names = sorted(
        {name for record in all_records for generation in record["generations"] for name in generation["signals"]}
        | {name for record in all_records for name in record["features"]}
    )

    common_episode = _common_episode_critics(all_records)
    critic_sets = _parse_critic_sets(critic_set_specs, common_episode)
    critic_models: dict[str, CriticBayes] = {}
    critic_test_predictions: dict[str, dict[str, float]] = {}
    for set_name, names in critic_sets.items():
        model = CriticBayes.fit(
            [record["critics"] for record in train], labels_train, names=names
        )
        critic_models[set_name] = model
        predictions = {record["episode_id"]: model.predict(record["critics"]) for record in test}
        critic_test_predictions[set_name] = predictions
        metric_rows.append(
            _metric_row(
                "all",
                f"critic:{set_name}",
                labels_test,
                [predictions[record["episode_id"]] for record in test],
            )
        )
        parameters.append(
            {
                "model": f"critic:{set_name}",
                "likelihoods": {
                    name: vars(likelihood) for name, likelihood in model.likelihoods.items()
                },
            }
        )

    common_step = _common_step_critics(all_records)
    if common_step:
        step_observations = [
            generation["critics"] for record in train for generation in record["generations"]
        ]
        step_labels = [
            record["success"] for record in train for _ in record["generations"]
        ]
        step_model = CriticBayes.fit(step_observations, step_labels, names=common_step)
        predictions = [
            step_model.predict_sequence([generation["critics"] for generation in record["generations"]])
            for record in test
        ]
        metric_rows.append(_metric_row("all", "stepwise_critics:all", labels_test, predictions))

    for signal in signal_names:
        higher_is_uncertain = signal_directions.get(signal, False)
        train_available = [record for record in train if _sequence(record, signal)]
        test_available = [record for record in test if _sequence(record, signal)]
        if len(train_available) < 2 or len(test_available) < 2:
            continue
        train_sequences = [_sequence(record, signal) for record in train_available]
        test_sequences = [_sequence(record, signal) for record in test_available]
        train_outcomes = [record["success"] for record in train_available]
        test_outcomes = [record["success"] for record in test_available]

        aggregation_names = aggregate(train_sequences[0], higher_is_uncertain=higher_is_uncertain)
        for aggregation_name in aggregation_names:
            train_values = [
                aggregate(sequence, higher_is_uncertain=higher_is_uncertain)[aggregation_name]
                for sequence in train_sequences
            ]
            test_values = [
                aggregate(sequence, higher_is_uncertain=higher_is_uncertain)[aggregation_name]
                for sequence in test_sequences
            ]
            platt = fit_platt(train_values, train_outcomes)
            metric_rows.append(
                _metric_row(signal, f"feature:{aggregation_name}", test_outcomes, platt.predict(test_values))
            )

        uq_models: dict[str, Any] = {}
        for mode in BINARY_MODES:
            model = (
                DoubleBinaryBayes.fit(
                    train_sequences, train_outcomes, higher_is_uncertain=higher_is_uncertain
                )
                if mode == "double"
                else BinaryBayes.fit(
                    train_sequences,
                    train_outcomes,
                    mode=mode,
                    higher_is_uncertain=higher_is_uncertain,
                )
            )
            uq_models[f"binary:{mode}"] = model
        uq_models["continuous"] = ContinuousBayes.fit(train_sequences, train_outcomes, lambda_=1.0)
        uq_models["continuous_tempered"] = ContinuousBayes.fit(
            train_sequences, train_outcomes, lambda_=tempered_lambda
        )

        for model_name, model in uq_models.items():
            probabilities = [model.predict(sequence) for sequence in test_sequences]
            metric_rows.append(_metric_row(signal, f"uq:{model_name}", test_outcomes, probabilities))
            for critic_set_name, critic_predictions in critic_test_predictions.items():
                fused = [
                    model.predict(sequence, start=critic_predictions[record["episode_id"]])
                    for record, sequence in zip(test_available, test_sequences)
                ]
                metric_rows.append(
                    _metric_row(
                        signal,
                        f"critic:{critic_set_name}+uq:{model_name}",
                        test_outcomes,
                        fused,
                    )
                )
        parameters.append(
            {
                "signal": signal,
                "higher_is_uncertain": higher_is_uncertain,
                "models": {
                    name: {
                        key: value
                        for key, value in vars(model).items()
                        if isinstance(value, (int, float, str, bool))
                    }
                    for name, model in uq_models.items()
                },
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["signal", "model", "n", "n_success", "auroc", "auprc", "prr_at_0_5", "brier", "nll", "ece"]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)
    (output_dir / "model_parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    summary = [
        "# Trajectory UQ analysis",
        "",
        f"- Episodes: {len(records)}",
        f"- Calibration/test: {len(train)}/{len(test)}",
        f"- Signals: {', '.join(signal_names) or 'none'}",
        f"- Episode critics: {', '.join(common_episode) or 'none'}",
        f"- Step critics: {', '.join(common_step) or 'none'}",
        f"- Metric rows: {len(metric_rows)}",
        "",
        "All thresholds, densities, critic likelihoods, and calibrators are fitted on calibration episodes only.",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return metric_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute trajectory UQ, Bayes, critic, and fused metrics")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tempered-lambda", type=float, default=0.25)
    parser.add_argument("--signal-direction", action="append", default=[])
    parser.add_argument("--critic-set", action="append", default=[])
    args = parser.parse_args()
    if not 0.0 < args.calibration_fraction < 1.0:
        parser.error("--calibration-fraction must be in (0, 1)")
    rows = analyze(
        episodes_path=args.episodes,
        output_dir=args.output_dir,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
        tempered_lambda=args.tempered_lambda,
        signal_directions=_parse_directions(args.signal_direction),
        critic_set_specs=args.critic_set,
    )
    print(f"metric_rows={len(rows)} output={args.output_dir}")


if __name__ == "__main__":
    main()
