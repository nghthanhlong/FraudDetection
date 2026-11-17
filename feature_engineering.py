"""Leakage-safe feature engineering for the fraud detection dataset."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

REQUIRED_COLUMNS = [
    "transaction_id",
    "amount_usd",
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
    "is_foreign_transaction",
    "hours_since_last_txn",
    "txn_count_last_24h",
]

CATEGORICAL_COLUMNS = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
]

INTERACTION_PAIRS = [
    ("merchant_category", "channel"),
    ("merchant_category", "auth_method"),
    ("channel", "auth_method"),
    ("device_type", "channel"),
]


def _normalise(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else "__missing__"


def _number(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value in {column!r}") from error


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _validate_input(fieldnames: Iterable[str], rows: list[dict[str, str]]) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if not rows:
        raise ValueError("The input CSV contains no data rows")


class FeatureBuilder:
    """Fit category metadata on training rows and transform other splits."""

    def __init__(self) -> None:
        self._frequencies: dict[str, dict[str, float]] = {}
        self._categories: dict[str, list[str]] = {}

    def fit(self, rows: list[dict[str, str]]) -> FeatureBuilder:
        if not rows:
            raise ValueError("Cannot fit features on empty rows")
        categorical = {
            column: [_normalise(row.get(column)) for row in rows]
            for column in CATEGORICAL_COLUMNS
        }
        self._frequencies = {
            column: {
                value: count / len(rows)
                for value, count in Counter(values).items()
            }
            for column, values in categorical.items()
        }
        for left, right in INTERACTION_PAIRS:
            name = f"{left}__{right}"
            values = [f"{categorical[left][i]}__{categorical[right][i]}" for i in range(len(rows))]
            self._frequencies[name] = {
                value: count / len(rows) for value, count in Counter(values).items()
            }
        self._categories = {
            column: sorted(values) for column, values in categorical.items()
        }
        for left, right in INTERACTION_PAIRS:
            name = f"{left}__{right}"
            self._categories[name] = sorted(self._frequencies[name])
        return self

    def transform(self, rows: list[dict[str, str]]) -> tuple[list[str], list[dict[str, object]]]:
        if not self._frequencies:
            raise ValueError("FeatureBuilder must be fitted before transform")
        if not rows:
            raise ValueError("Cannot transform empty rows")

        categorical = {
            column: [_normalise(row.get(column)) for row in rows]
            for column in CATEGORICAL_COLUMNS
        }
        interactions = {
            f"{left}__{right}": [
                f"{categorical[left][index]}__{categorical[right][index]}"
                for index in range(len(rows))
            ]
            for left, right in INTERACTION_PAIRS
        }
        output_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows):
            amount = max(_number(row, "amount_usd"), 0.0)
            hours = max(_number(row, "hours_since_last_txn"), 0.0)
            txn_count = max(_number(row, "txn_count_last_24h"), 0.0)
            output: dict[str, object] = {
                "transaction_id": row["transaction_id"],
                "amount_log1p": math.log1p(amount),
                "amount_fraction": amount % 1,
                "amount_rounded_5": int(amount % 5 == 0),
                "amount_rounded_10": int(amount % 10 == 0),
                "amount_is_integer": int(amount % 1 == 0),
                "hours_since_last_txn_log1p": math.log1p(hours),
                "txn_count_last_24h_log1p": math.log1p(txn_count),
                "txn_intensity_per_hour": txn_count / (hours + 1 / 24),
                "txn_intensity_per_day": txn_count / (hours / 24 + 1),
                "is_recent_txn": int(hours <= 1),
                "is_high_velocity": int(txn_count >= 5),
                "is_foreign_transaction": int(
                    _is_true(row["is_foreign_transaction"])
                ),
            }
            values = {**categorical, **interactions}
            for column, column_values in values.items():
                value = column_values[index]
                output[f"{column}_frequency"] = self._frequencies[column].get(value, 0.0)
                output[column] = value
                for category in self._categories[column]:
                    output[f"{column}__{category}"] = int(value == category)
            output_rows.append(output)
        return list(output_rows[0].keys()), output_rows

    def fit_transform(self, rows: list[dict[str, str]]) -> tuple[list[str], list[dict[str, object]]]:
        return self.fit(rows).transform(rows)


def build_features(rows: list[dict[str, str]]) -> tuple[list[str], list[dict[str, object]]]:
    """Backward-compatible helper that fits and transforms one complete dataset."""
    return FeatureBuilder().fit_transform(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("engineered_features.csv"))
    args = parser.parse_args()
    with args.input_csv.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)
        _validate_input(reader.fieldnames or [], rows)
    fieldnames, engineered = build_features(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(engineered)
    print(f"Created {args.output} with shape ({len(engineered)}, {len(fieldnames)})")


if __name__ == "__main__":
    main()
