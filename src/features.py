"""Compact row-wise features for LightGBM / XGBoost.

Copy to src/features.py. Preserve features_v2.py for champion reproduction.
This module deliberately removes 14 candidate engineered features from v2.
No target or fitted statistics are used. AI flag is opt-in at model selection.
"""
import numpy as np
import pandas as pd

FEATURE_VERSION = "v3"
RAW_CATEGORICAL_COLUMNS = [
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
]
RAW_NUMERIC_COLUMNS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h",
    "distance_from_home_km", "card_age_months", "customer_age",
    "account_balance_usd", "cvv_retry_count", "velocity_score",
    "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
]
CORE_BOOLEAN_COLUMNS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch",
]
AI_BOOLEAN_COLUMNS = ["is_ai_generated_scam_attempt"]
RAW_BOOLEAN_COLUMNS = CORE_BOOLEAN_COLUMNS + AI_BOOLEAN_COLUMNS
RAW_LEGACY_COLUMNS = RAW_NUMERIC_COLUMNS + RAW_CATEGORICAL_COLUMNS
RAW_CORE_COLUMNS = RAW_LEGACY_COLUMNS + CORE_BOOLEAN_COLUMNS
RAW_ALL_COLUMNS = RAW_LEGACY_COLUMNS + RAW_BOOLEAN_COLUMNS

# Legacy engineered features retained as candidates (not proven useful).
VELOCITY_COLUMNS = ["txn_intensity_per_hour", "txn_intensity_per_day"]
CUSTOMER_COLUMNS = ["balance_amount_ratio", "prior_disputes_per_card_age"]
INTERACTION_COLUMNS = ["merchant_category__auth_method"]
ENGINEERED_NUMERIC_COLUMNS = VELOCITY_COLUMNS + CUSTOMER_COLUMNS

BINARY_RISK_COLUMNS = [
    "risk_flag_count", "network_risk_count", "vpn_ip_mismatch", "foreign_ip_mismatch",
]
CVV_RISK_COLUMNS = ["cvv_velocity_risk", "cvv_merchant_risk", "cvv_binary_risk"]
AUTH_RISK_COLUMNS = [
    "no_auth_merchant_risk", "no_auth_binary_risk",
    "new_merchant_risk", "new_merchant_no_auth",
]
TRANSACTION_RISK_COLUMNS = [
    "amount_velocity_risk", "amount_merchant_risk", "distance_foreign",
]
ADVANCED_NUMERIC_COLUMNS = (
    BINARY_RISK_COLUMNS + CVV_RISK_COLUMNS + AUTH_RISK_COLUMNS + TRANSACTION_RISK_COLUMNS
)
CATEGORICAL_COLUMNS = RAW_CATEGORICAL_COLUMNS + INTERACTION_COLUMNS
FEATURE_GROUPS = {
    "velocity": VELOCITY_COLUMNS, "customer": CUSTOMER_COLUMNS,
    "categorical": INTERACTION_COLUMNS,
    "binary_risk": BINARY_RISK_COLUMNS, "cvv_risk": CVV_RISK_COLUMNS,
    "auth_risk": AUTH_RISK_COLUMNS, "transaction_risk": TRANSACTION_RISK_COLUMNS,
}
COMPACT_LEGACY_COLUMNS = RAW_CORE_COLUMNS + ENGINEERED_NUMERIC_COLUMNS
COMPACT_COLUMNS = COMPACT_LEGACY_COLUMNS + INTERACTION_COLUMNS + ADVANCED_NUMERIC_COLUMNS
FEATURE_SETS = {
    "raw_legacy": RAW_LEGACY_COLUMNS,
    "raw_core": RAW_CORE_COLUMNS,
    "raw_all": RAW_ALL_COLUMNS,  # Only after confirming AI flag is available at prediction time.
    "compact_legacy": COMPACT_LEGACY_COLUMNS,
    "category_auth": COMPACT_LEGACY_COLUMNS + INTERACTION_COLUMNS,
    "binary_risk": COMPACT_LEGACY_COLUMNS + BINARY_RISK_COLUMNS,
    "cvv_risk": COMPACT_LEGACY_COLUMNS + ["risk_flag_count"] + CVV_RISK_COLUMNS,
    "auth_risk": COMPACT_LEGACY_COLUMNS + ["risk_flag_count"] + AUTH_RISK_COLUMNS,
    "transaction_risk": COMPACT_LEGACY_COLUMNS + TRANSACTION_RISK_COLUMNS,
    "compact_all": COMPACT_COLUMNS,
    "compact_all_with_ai": COMPACT_COLUMNS + AI_BOOLEAN_COLUMNS,
}


def _validate_columns(frame, columns):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing input columns: {missing}")


def _normalize_boolean(series):
    """Accept booleans, 0/1 and their string forms; preserve missing as NaN."""
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="raise")
        invalid = series.notna() & ~values.isin([0, 1])
    else:
        values = series.astype("string").str.strip().str.lower().map({
            "true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0,
        })
        invalid = series.notna() & values.isna()
    if invalid.any():
        raise ValueError(f"Invalid binary values in {series.name}: {series[invalid].unique()[:5]}")
    return values.astype("float32")


def _prepare_categoricals(df):
    for col in RAW_CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("string").fillna("__missing__").astype("category")


def add_velocity_features(df):
    hours = df["hours_since_last_txn"].clip(lower=0)
    txn_count = df["txn_count_last_24h"].clip(lower=0)
    # Legacy heuristic scores, not literal transaction rates.
    df["txn_intensity_per_hour"] = txn_count / (hours + 1 / 24)
    df["txn_intensity_per_day"] = txn_count / (hours / 24 + 1)


def add_customer_features(df):
    amount = df["amount_usd"].clip(lower=0)
    balance = df["account_balance_usd"].clip(lower=0)
    card_age = df["card_age_months"].clip(lower=0)
    disputes = df["prior_disputes"].clip(lower=0)
    df["balance_amount_ratio"] = balance / (amount + 1)
    df["prior_disputes_per_card_age"] = disputes / (card_age + 1)


def add_categorical_interactions(df):
    for col in INTERACTION_COLUMNS:
        left, right = col.split("__")
        # Keep legacy object dtype; notebooks align categories before fitting.
        df[col] = df[left].astype(str) + "__" + df[right].astype(str)


def add_binary_risk_features(df):
    # AI flag deliberately excluded from aggregates until availability is verified.
    df["risk_flag_count"] = df[CORE_BOOLEAN_COLUMNS].sum(axis=1, min_count=5)
    network = ["is_foreign_transaction", "used_vpn", "ip_country_mismatch"]
    df["network_risk_count"] = df[network].sum(axis=1, min_count=len(network))
    df["vpn_ip_mismatch"] = df["used_vpn"] * df["ip_country_mismatch"]
    df["foreign_ip_mismatch"] = df["is_foreign_transaction"] * df["ip_country_mismatch"]


def add_cvv_risk_features(df):
    cvv = df["cvv_retry_count"]
    df["cvv_velocity_risk"] = cvv * df["velocity_score"]
    df["cvv_merchant_risk"] = cvv * df["merchant_risk_score"]
    df["cvv_binary_risk"] = cvv * df["risk_flag_count"]


def add_auth_risk_features(df):
    auth = df["auth_method"]
    no_auth = auth.eq("No Authentication").astype("float32")
    no_auth = no_auth.mask(auth.eq("__missing__"))
    df["no_auth_merchant_risk"] = no_auth * df["merchant_risk_score"]
    df["no_auth_binary_risk"] = no_auth * df["risk_flag_count"]
    df["new_merchant_risk"] = df["is_new_merchant"] * df["merchant_risk_score"]
    df["new_merchant_no_auth"] = df["is_new_merchant"] * no_auth


def add_transaction_risk_features(df):
    amount_log = np.log1p(df["amount_usd"].clip(lower=0))
    df["amount_velocity_risk"] = amount_log * df["velocity_score"]
    df["amount_merchant_risk"] = amount_log * df["merchant_risk_score"]
    df["distance_foreign"] = df["distance_from_home_km"] * df["is_foreign_transaction"]


def add_features(frame: pd.DataFrame, *, include_advanced: bool = True) -> pd.DataFrame:
    """Copy raw input and produce the compact candidate feature pool.

    Advanced mode requires five core flags. The optional AI flag, if present,
    is normalized but never used in derived risk counts. Select columns using
    FEATURE_SETS rather than fitting on the whole returned frame (which can
    contain ID, the optional AI flag or other input columns).
    include_advanced=False creates only the retained legacy engineered groups;
    this is NOT the original 31-feature champion, which requires v2.
    """
    if "is_fraud" in frame.columns:
        raise ValueError("Drop is_fraud before calling add_features().")
    _validate_columns(frame, RAW_CORE_COLUMNS if include_advanced else RAW_LEGACY_COLUMNS)
    result = frame.copy()
    _prepare_categoricals(result)
    for col in RAW_BOOLEAN_COLUMNS:
        if col in result.columns:
            result[col] = _normalize_boolean(result[col])
    add_velocity_features(result)
    add_customer_features(result)
    add_categorical_interactions(result)
    if include_advanced:
        add_binary_risk_features(result)
        add_cvv_risk_features(result)
        add_auth_risk_features(result)
        add_transaction_risk_features(result)
    return result
