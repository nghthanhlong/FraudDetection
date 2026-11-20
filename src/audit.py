"""
Audit utilities cho bài toán fraud/dispute detection.

Cung cấp các hàm kiểm toán độc lập:
- compute_mann_whitney_auc: Tính ROC-AUC đơn biến bằng Mann-Whitney U test.
- audit_univariate_leakage: Kiểm tra rò rỉ nhãn (leakage) đơn biến cho toàn bộ feature.
- audit_target_drift: Phân tích độ lệch tỷ lệ nhãn (target drift) theo chuỗi thời gian/ID.
- audit_numeric_sanity: Kiểm tra missing, Inf, giá trị cực trị và phân phối của biến số.
- audit_unseen_categories: Kiểm tra cardinality và rủi ro category mới (unseen categories).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_mann_whitney_auc(scores: np.ndarray, targets: np.ndarray) -> float:
    """
    Tính ROC-AUC bằng Mann-Whitney U test (Wilcoxon rank sum).
    
    Phương pháp này không phụ thuộc vào scikit-learn, tính trực tiếp
    trên rank với độ chính xác tuyệt đối.

    Parameters
    ----------
    scores : np.ndarray
        Mảng giá trị dự đoán hoặc giá trị feature đơn biến.
    targets : np.ndarray
        Mảng nhãn nhị phân (0 hoặc 1).

    Returns
    -------
    float
        Giá trị ROC-AUC (nằm trong khoảng [0, 1]).
    """
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=int)

    # Impute missing tạm thời bằng median để rank được toàn bộ dòng
    valid_mask = ~np.isnan(scores)
    if not np.all(valid_mask):
        scores = scores.copy()
        median_val = np.nanmedian(scores) if np.any(valid_mask) else 0.0
        scores[~valid_mask] = median_val

    n1 = int(np.sum(targets == 1))
    n0 = int(np.sum(targets == 0))
    if n1 == 0 or n0 == 0:
        return 0.5

    # Tính ranks
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Xử lý các giá trị bằng nhau (ties)
    scores_sorted = scores[order]
    unique_vals, inverse, counts = np.unique(scores_sorted, return_inverse=True, return_counts=True)
    if len(unique_vals) < len(scores):
        tie_ranks = np.bincount(inverse, weights=ranks[order]) / counts
        ranks[order] = tie_ranks[inverse]

    sum_ranks_pos = np.sum(ranks[targets == 1])
    u = sum_ranks_pos - (n1 * (n1 + 1)) / 2.0
    auc = u / (n1 * n0)
    return float(auc)


def audit_univariate_leakage(
    df: pd.DataFrame,
    target: Union[pd.Series, np.ndarray],
    feature_cols: Optional[List[str]] = None,
    leakage_threshold: float = 0.90,
) -> pd.DataFrame:
    """
    Tính univariate AUC và tương quan cho tất cả các feature nhằm phát hiện rò rỉ dữ liệu.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chứa các feature cần kiểm toán.
    target : pd.Series or np.ndarray
        Nhãn target nhị phân (0 và 1).
    feature_cols : list of str, optional
        Danh sách cột cần đánh giá. Mặc định là toàn bộ cột (trừ target).
    leakage_threshold : float, default=0.90
        Ngưỡng AUC cảnh báo nghi ngờ leakage.

    Returns
    -------
    pd.DataFrame
        Bảng tổng hợp xếp hạng feature theo univariate AUC giảm dần.
    """
    y = np.asarray(target, dtype=int)
    overall_event_rate = np.mean(y)

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ["is_fraud", "id"]]

    records = []
    for col in feature_cols:
        s = df[col]
        dtype = s.dtype

        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            vals = s.astype(float).values
            raw_auc = compute_mann_whitney_auc(vals, y)
            direction_auc = max(raw_auc, 1.0 - raw_auc)

            # Pearson correlation
            std_val = np.nanstd(vals)
            corr = np.corrcoef(vals, y)[0, 1] if std_val > 0 else 0.0

            warning = "LEAKAGE RISK (>0.90)" if direction_auc > leakage_threshold else (
                "HIGH (>0.80)" if direction_auc > 0.80 else "Normal"
            )

            records.append({
                "feature": col,
                "type": "numeric/bool",
                "raw_auc": round(raw_auc, 4),
                "univariate_auc": round(direction_auc, 4),
                "correlation": round(corr, 4),
                "leakage_warning": warning,
            })
        else:
            # Categorical: ánh xạ bằng out-of-group event rate
            target_series = pd.Series(y, index=df.index)
            event_rates = target_series.groupby(s, observed=False).mean()
            mapped = (
                s.astype(object)
                .map(event_rates)
                .fillna(overall_event_rate)
                .to_numpy(dtype=float)
            )
            raw_auc = compute_mann_whitney_auc(mapped, y)
            direction_auc = max(raw_auc, 1.0 - raw_auc)

            warning = "LEAKAGE RISK (>0.90)" if direction_auc > leakage_threshold else (
                "HIGH (>0.80)" if direction_auc > 0.80 else "Normal"
            )

            records.append({
                "feature": col,
                "type": "categorical",
                "raw_auc": round(raw_auc, 4),
                "univariate_auc": round(direction_auc, 4),
                "correlation": np.nan,
                "leakage_warning": warning,
            })

    result_df = pd.DataFrame(records).sort_values(by="univariate_auc", ascending=False).reset_index(drop=True)
    return result_df


def audit_target_drift(
    df: pd.DataFrame,
    id_col: str = "id",
    target_col: str = "is_fraud",
    n_bins: int = 12,
) -> pd.DataFrame:
    """
    Kiểm tra độ lệch tỷ lệ nhãn (target drift) theo các phân đoạn chuỗi thời gian (ID bins).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chứa cột ID và Target.
    id_col : str, default='id'
        Tên cột thứ tự thời gian/ID.
    target_col : str, default='is_fraud'
        Tên cột nhãn.
    n_bins : int, default=12
        Số khoảng thời gian cần chia.

    Returns
    -------
    pd.DataFrame
        Bảng thống kê tỷ lệ nhãn qua từng khoảng.
    """
    df_sorted = df.sort_values(by=id_col).copy()
    labels = [f"Period_{i+1:02d}" for i in range(n_bins)]
    df_sorted["period_bin"] = pd.qcut(df_sorted[id_col], q=n_bins, labels=labels)

    summary = df_sorted.groupby("period_bin", observed=False).agg(
        id_min=(id_col, "min"),
        id_max=(id_col, "max"),
        total_txns=(target_col, "count"),
        fraud_txns=(target_col, "sum"),
        fraud_rate=(target_col, "mean"),
    )
    summary["fraud_rate_pct"] = (summary["fraud_rate"] * 100).round(4)
    return summary


def audit_numeric_sanity(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Kiểm toán missing, Inf, phân phối và các phân vị ngoại lai (outliers).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu.
    numeric_cols : list of str, optional
        Danh sách các cột số cần kiểm tra.

    Returns
    -------
    pd.DataFrame
        Bảng kiểm toán chất lượng biến số.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number, bool]).columns.tolist()

    records = []
    n_total = len(df)

    for col in numeric_cols:
        s = df[col].astype(float)
        n_nan = int(s.isna().sum())
        pct_nan = round((n_nan / n_total) * 100, 4)

        n_inf = int((np.isposinf(s) | np.isneginf(s)).sum())

        valid = s[np.isfinite(s)]
        if len(valid) > 0:
            val_min = round(float(valid.min()), 4)
            p001 = round(float(valid.quantile(0.001)), 4)
            p50 = round(float(valid.quantile(0.50)), 4)
            p999 = round(float(valid.quantile(0.999)), 4)
            val_max = round(float(valid.max()), 4)
        else:
            val_min = p001 = p50 = p999 = val_max = np.nan

        records.append({
            "column": col,
            "missing_count": n_nan,
            "missing_pct": pct_nan,
            "inf_count": n_inf,
            "min": val_min,
            "p0.1%": p001,
            "median (p50)": p50,
            "p99.9%": p999,
            "max": val_max,
        })

    return pd.DataFrame(records)


def audit_unseen_categories(
    train_df: pd.DataFrame,
    eval_dfs: Dict[str, pd.DataFrame],
    cat_cols: List[str],
) -> pd.DataFrame:
    """
    Kiểm tra Cardinality và rủi ro Category chưa từng thấy (Unseen categories)
    giữa tập Train và các tập đánh giá (Valid / Test).

    Parameters
    ----------
    train_df : pd.DataFrame
        Tập dữ liệu huấn luyện.
    eval_dfs : dict of {str: pd.DataFrame}
        Dictionary các tập đánh giá cần so sánh (vd: {'Valid': df_va, 'Test': df_te}).
    cat_cols : list of str
        Danh sách các cột categorical.

    Returns
    -------
    pd.DataFrame
        Bảng báo cáo cardinality và unseen categories.
    """
    records = []
    for col in cat_cols:
        if col not in train_df.columns:
            raise KeyError(
                f"Categorical column '{col}' is missing from train_df. "
                "Pass feature-engineered DataFrames when auditing interaction columns."
            )
        train_cats = set(train_df[col].dropna().astype(str).unique())
        row = {
            "column": col,
            "train_cardinality": len(train_cats),
        }
        for name, ev_df in eval_dfs.items():
            if col not in ev_df.columns:
                raise KeyError(
                    f"Categorical column '{col}' is missing from eval_df '{name}'. "
                    "Pass feature-engineered DataFrames when auditing interaction columns."
                )
            ev_cats = set(ev_df[col].dropna().astype(str).unique())
            unseen = ev_cats - train_cats
            row[f"{name}_cardinality"] = len(ev_cats)
            row[f"unseen_in_{name}"] = len(unseen)

        records.append(row)

    return pd.DataFrame(records)
