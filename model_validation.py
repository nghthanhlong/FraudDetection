"""Cross-validated fraud model training and submission generation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline

RAW_CATEGORICAL_COLUMNS = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
]

RAW_NUMERIC_COLUMNS = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
]


def _add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    amount = result["amount_usd"].clip(lower=0)
    hours = result["hours_since_last_txn"].clip(lower=0)
    txn_count = result["txn_count_last_24h"].clip(lower=0)
    card_age = result["card_age_months"].clip(lower=0)
    balance = result["account_balance_usd"].clip(lower=0)
    prior_disputes = result["prior_disputes"].clip(lower=0)
    result["amount_log1p"] = np.log1p(amount)
    result["amount_fraction"] = amount.mod(1)
    result["amount_is_integer"] = amount.mod(1).eq(0).astype(np.int8)
    result["amount_rounded_5"] = amount.mod(5).eq(0).astype(np.int8)
    result["amount_rounded_10"] = amount.mod(10).eq(0).astype(np.int8)
    result["hours_since_last_txn_log1p"] = np.log1p(hours)
    result["txn_count_last_24h_log1p"] = np.log1p(txn_count)
    result["txn_intensity_per_hour"] = txn_count / (hours + 1 / 24)
    result["txn_intensity_per_day"] = txn_count / (hours / 24 + 1)
    result["is_recent_txn"] = hours.le(1).astype(np.int8)
    result["is_high_velocity"] = txn_count.ge(5).astype(np.int8)
    result["balance_amount_ratio"] = balance / (amount + 1)
    result["prior_disputes_per_card_age"] = prior_disputes / (card_age + 1)
    for column in RAW_CATEGORICAL_COLUMNS:
        result[column] = result[column].fillna("__missing__").astype("category")
    result["merchant_category__channel"] = (
        result["merchant_category"].astype(str) + "__" + result["channel"].astype(str)
    )
    result["merchant_category__auth_method"] = (
        result["merchant_category"].astype(str) + "__" + result["auth_method"].astype(str)
    )
    result["channel__auth_method"] = (
        result["channel"].astype(str) + "__" + result["auth_method"].astype(str)
    )
    result["device_type__channel"] = (
        result["device_type"].astype(str) + "__" + result["channel"].astype(str)
    )
    return result


def _column_transformer(
    numeric_columns: Iterable[str],
    encoding: str = "onehot",
) -> ColumnTransformer:
    numeric_columns = list(numeric_columns)
    categorical_columns = RAW_CATEGORICAL_COLUMNS + [
        "merchant_category__channel",
        "merchant_category__auth_method",
        "channel__auth_method",
        "device_type__channel",
    ]
    if encoding == "target":
        encoder = TargetEncoder(target_type="binary", smooth="auto", random_state=42)
    elif encoding == "frequency":
        encoder = None
        categorical_columns = []
    else:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        )
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(with_mean=False), numeric_columns),
            *([("categorical", encoder, categorical_columns)] if encoder else []),
        ],
        sparse_threshold=0.3,
    )


def _frequency_encode(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_encoded = train.copy()
    valid_encoded = valid.copy()
    for column in columns:
        frequencies = train_encoded[column].astype(str).value_counts(normalize=True)
        train_encoded[f"{column}_frequency"] = (
            train_encoded[column].astype(str).map(frequencies).fillna(0.0).astype(np.float32)
        )
        valid_encoded[f"{column}_frequency"] = (
            valid_encoded[column].astype(str).map(frequencies).fillna(0.0).astype(np.float32)
        )
    return train_encoded, valid_encoded


def _fit_fold(
    train: pd.DataFrame,
    target: pd.Series,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    random_state: int,
    encoding: str,
    imbalance: str,
) -> tuple[float, float]:
    x_train = train.iloc[train_index]
    x_valid = train.iloc[valid_index]
    y_train = target.iloc[train_index]
    y_valid = target.iloc[valid_index]
    numeric_columns = [
        column
        for column in RAW_NUMERIC_COLUMNS
        + ["amount_log1p", "amount_fraction", "amount_is_integer",
           "amount_rounded_5", "amount_rounded_10",
           "hours_since_last_txn_log1p", "txn_count_last_24h_log1p",
           "txn_intensity_per_hour", "txn_intensity_per_day",
           "is_recent_txn", "is_high_velocity", "balance_amount_ratio",
           "prior_disputes_per_card_age"]
        if column in train.columns
    ]
    if encoding == "frequency":
        frequency_columns = RAW_CATEGORICAL_COLUMNS + [
            "merchant_category__channel",
            "merchant_category__auth_method",
            "channel__auth_method",
            "device_type__channel",
        ]
        x_train, x_valid = _frequency_encode(x_train, x_valid, frequency_columns)
        numeric_columns += [f"{column}_frequency" for column in frequency_columns]

    model = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced" if imbalance == "class_weight" else None,
        max_iter=500,
        random_state=random_state,
    )
    steps = [
        ("features", _column_transformer(numeric_columns, encoding)),
    ]
    if imbalance == "smote":
        steps.append(("smote", SMOTE(
            sampling_strategy=0.1,
            k_neighbors=5,
            random_state=random_state,
        )))
    steps.append(("model", model))
    pipeline = ImbalancedPipeline(steps) if imbalance == "smote" else Pipeline(steps)
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_valid)[:, 1]
    return (
        roc_auc_score(y_valid, probabilities),
        average_precision_score(y_valid, probabilities),
    )


def cross_validate(
    train: pd.DataFrame,
    folds: int,
    random_state: int,
    workers: int,
    encoding: str = "onehot",
    imbalance: str = "class_weight",
) -> dict[str, float]:
    target = train["is_fraud"].astype(np.int8)
    features = _add_features(train.drop(columns=["is_fraud"]))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    splits = list(splitter.split(features, target))
    workers = max(1, min(workers, folds, os.cpu_count() or 1))
    scores = Parallel(n_jobs=workers, backend="threading", verbose=10)(
        delayed(_fit_fold)(
            features, target, train_index, valid_index, random_state, encoding, imbalance
        )
        for train_index, valid_index in splits
    )
    roc_scores = [score[0] for score in scores]
    pr_scores = [score[1] for score in scores]
    return {
        "roc_auc": float(np.mean(roc_scores)),
        "pr_auc": float(np.mean(pr_scores)),
    }


def train_submission(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output: Path,
    random_state: int,
    encoding: str,
    imbalance: str,
) -> None:
    target = train["is_fraud"].astype(np.int8)
    x_train = _add_features(train.drop(columns=["is_fraud"]))
    x_test = _add_features(test)
    numeric_columns = [
        column
        for column in RAW_NUMERIC_COLUMNS
        + ["amount_log1p", "amount_fraction", "amount_is_integer",
           "amount_rounded_5", "amount_rounded_10",
           "hours_since_last_txn_log1p", "txn_count_last_24h_log1p",
           "txn_intensity_per_hour", "txn_intensity_per_day",
           "is_recent_txn", "is_high_velocity", "balance_amount_ratio",
           "prior_disputes_per_card_age"]
        if column in x_train.columns
    ]
    if encoding == "frequency":
        frequency_columns = RAW_CATEGORICAL_COLUMNS + [
            "merchant_category__channel",
            "merchant_category__auth_method",
            "channel__auth_method",
            "device_type__channel",
        ]
        x_train, x_test = _frequency_encode(x_train, x_test, frequency_columns)
        numeric_columns += [f"{column}_frequency" for column in frequency_columns]

    model = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced" if imbalance == "class_weight" else None,
        max_iter=500,
        random_state=random_state,
    )
    steps = [("features", _column_transformer(numeric_columns, encoding))]
    if imbalance == "smote":
        steps.append(("smote", SMOTE(
            sampling_strategy=0.1,
            k_neighbors=5,
            random_state=random_state,
        )))
    steps.append(("model", model))
    pipeline = ImbalancedPipeline(steps) if imbalance == "smote" else Pipeline(steps)
    pipeline.fit(x_train, target)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    submission = pd.DataFrame({"id": test["id"], "is_fraud": probabilities})
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--encoding",
        choices=["onehot", "frequency", "target"],
        default="onehot",
    )
    parser.add_argument(
        "--imbalance",
        choices=["none", "class_weight", "smote"],
        default="class_weight",
    )
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--submission", type=Path, default=Path("submissions/submission.csv"))
    args = parser.parse_args()
    train = pd.read_csv(args.data_dir / "train.csv")
    if "is_fraud" not in train or train["is_fraud"].nunique() < 2:
        raise ValueError("train.csv must contain a binary is_fraud target with both classes")
    if args.compare:
        combinations = [
            ("onehot", "none"),
            ("onehot", "class_weight"),
            ("onehot", "smote"),
            ("frequency", "class_weight"),
            ("target", "class_weight"),
        ]
        results = {}
        for encoding, imbalance in combinations:
            print(f"Running encoding={encoding}, imbalance={imbalance}")
            results[(encoding, imbalance)] = cross_validate(
                train, args.folds, args.random_state, args.workers, encoding, imbalance
            )
        for (encoding, imbalance), metrics in results.items():
            print(
                f"{encoding:9s} / {imbalance:12s} "
                f"ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f}"
            )
        best_encoding, best_imbalance = max(
            results,
            key=lambda key: results[key]["pr_auc"],
        )
        print(f"Best PR-AUC: encoding={best_encoding}, imbalance={best_imbalance}")
        args.encoding = best_encoding
        args.imbalance = best_imbalance
    else:
        metrics = cross_validate(
            train, args.folds, args.random_state, args.workers, args.encoding, args.imbalance
        )
        print(f"Mean ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Mean PR-AUC: {metrics['pr_auc']:.4f}")
    test_path = args.data_dir / "test.csv"
    if test_path.exists():
        train_submission(
            train,
            pd.read_csv(test_path),
            args.submission,
            args.random_state,
            args.encoding,
            args.imbalance,
        )
        print(f"Created {args.submission}")


if __name__ == "__main__":
    main()
