"""
Feature engineering cho dữ liệu giao dịch.

Hàm add_features() nhận một DataFrame (không có cột is_fraud)
và trả về DataFrame với các feature mới được thêm vào.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Danh sách cột gốc
# ---------------------------------------------------------------------------

RAW_CATEGORICAL_COLUMNS = [
    "merchant_category", "card_type", "auth_method", "channel", "device_type"
]

RAW_NUMERIC_COLUMNS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h",
    "distance_from_home_km", "card_age_months", "customer_age",
    "account_balance_usd", "cvv_retry_count", "velocity_score",
    "time_of_day_hour", "day_of_week", "merchant_risk_score",
    "prior_disputes",
]

# Cột tương tác (categorical × categorical)
INTERACTION_COLUMNS = [
    "merchant_category__channel",
    "merchant_category__auth_method",
    "channel__auth_method",
    "device_type__channel",
]

# Feature số được tạo bởi add_features()
ENGINEERED_NUMERIC_COLUMNS = [
    "amount_log1p", "amount_fraction", "amount_is_integer",
    "amount_rounded_5", "amount_rounded_10",
    "hours_since_last_txn_log1p", "txn_count_last_24h_log1p",
    "txn_intensity_per_hour", "txn_intensity_per_day",
    "is_recent_txn", "is_high_velocity", "balance_amount_ratio",
    "prior_disputes_per_card_age",
]


# ---------------------------------------------------------------------------
# Hàm chính
# ---------------------------------------------------------------------------

def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm các feature mới vào DataFrame.

    Parameters
    ----------
    frame : pd.DataFrame
        DataFrame đầu vào, KHÔNG chứa cột ``is_fraud``.

    Returns
    -------
    pd.DataFrame
        DataFrame mới với các feature bổ sung.

    Nhóm feature
    ------------
    - Amount  : log1p, phần thập phân, số tròn 5/10, số nguyên.
    - Velocity: log, intensity theo giờ/ngày, recent/high velocity.
    - Khách hàng: balance/amount, prior disputes/card age.
    - Tương tác: merchant_category × channel/auth_method;
                 channel × auth_method; device_type × channel.
    """
    result = frame.copy()

    amount = result["amount_usd"].clip(lower=0)
    hours = result["hours_since_last_txn"].clip(lower=0)
    txn_count = result["txn_count_last_24h"].clip(lower=0)
    card_age = result["card_age_months"].clip(lower=0)
    balance = result["account_balance_usd"].clip(lower=0)
    prior_disputes = result["prior_disputes"].clip(lower=0)

    # Amount features
    result["amount_log1p"] = np.log1p(amount)
    result["amount_fraction"] = amount.mod(1)
    result["amount_is_integer"] = amount.mod(1).eq(0).astype(np.int8)
    result["amount_rounded_5"] = amount.mod(5).eq(0).astype(np.int8)
    result["amount_rounded_10"] = amount.mod(10).eq(0).astype(np.int8)

    # Velocity features
    result["hours_since_last_txn_log1p"] = np.log1p(hours)
    result["txn_count_last_24h_log1p"] = np.log1p(txn_count)
    result["txn_intensity_per_hour"] = txn_count / (hours + 1 / 24)
    result["txn_intensity_per_day"] = txn_count / (hours / 24 + 1)
    result["is_recent_txn"] = hours.le(1).astype(np.int8)
    result["is_high_velocity"] = txn_count.ge(5).astype(np.int8)

    # Customer features
    result["balance_amount_ratio"] = balance / (amount + 1)
    result["prior_disputes_per_card_age"] = prior_disputes / (card_age + 1)

    # Categorical — fill missing và chuyển sang category dtype
    for col in RAW_CATEGORICAL_COLUMNS:
        result[col] = result[col].fillna("__missing__").astype("category")

    # Interaction features (categorical × categorical)
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
