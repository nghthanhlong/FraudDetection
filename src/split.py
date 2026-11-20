"""
Chiến lược phân chia dữ liệu (Data Splitting) cho bài toán gian lận.

Cung cấp:
- create_stratified_kfold: Gán nhãn fold theo Stratified K-Fold Cross-Validation,
  bảo toàn tỷ lệ gian lận chính xác trên từng fold.
- time_based_split: Chia train/validation/test theo thứ tự thời gian tăng dần,
  hỗ trợ thiết lập buffer/gap nhằm chống rò rỉ độ trễ nhãn (label delay).
- summarize_kfold_splits: Báo cáo kích thước và phân phối target của từng fold.
- summarize_splits: Báo cáo kích thước và phân phối target của các tập phân tách.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def create_stratified_kfold(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    fold_col: str = "fold",
) -> pd.DataFrame:
    """
    Tạo cột 'fold' (từ 0 đến n_splits - 1) bằng Stratified K-Fold.
    
    Đảm bảo mỗi fold có cùng tỷ lệ positive/negative, chạy độc lập
    bằng NumPy/Pandas không phụ thuộc môi trường ngoài.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đầu vào chứa cột target.
    target_col : str, default='is_fraud'
        Tên cột nhãn nhị phân.
    n_splits : int, default=5
        Số lượng folds (khuyến nghị K=5).
    shuffle : bool, default=True
        Xáo trộn ngẫu nhiên trước khi chia.
    random_state : int, default=42
        Seed ngẫu nhiên để tái lập kết quả.
    fold_col : str, default='fold'
        Tên cột fold mới được thêm vào.

    Returns
    -------
    pd.DataFrame
        DataFrame ban đầu kèm cột 'fold' kiểu int.
    """
    df_out = df.copy()
    y = df_out[target_col].astype(int).values
    n = len(df_out)

    rng = np.random.default_rng(random_state)
    folds = np.zeros(n, dtype=int)

    # Phân chia riêng cho từng lớp (0 và 1) để đảm bảo stratified
    for cls in [0, 1]:
        cls_indices = np.where(y == cls)[0]
        if shuffle:
            rng.shuffle(cls_indices)
        # Gán nhãn fold tuần tự cho các index của lớp này
        for i, idx in enumerate(cls_indices):
            folds[idx] = i % n_splits

    df_out[fold_col] = folds
    return df_out


def summarize_kfold_splits(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
    fold_col: str = "fold",
) -> pd.DataFrame:
    """
    Báo cáo thống kê kích thước, số ca fraud và tỷ lệ fraud trên từng fold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã có cột fold.
    target_col : str, default='is_fraud'
        Tên cột target.
    fold_col : str, default='fold'
        Tên cột fold.

    Returns
    -------
    pd.DataFrame
        Bảng thống kê các fold.
    """
    records = []
    n_splits = df[fold_col].nunique()
    total_len = len(df)

    for f in range(n_splits):
        val_mask = df[fold_col] == f
        tr_mask = ~val_mask

        n_val = int(val_mask.sum())
        n_val_fraud = int(df.loc[val_mask, target_col].sum())
        val_fraud_rate = (n_val_fraud / n_val) * 100

        n_tr = int(tr_mask.sum())
        n_tr_fraud = int(df.loc[tr_mask, target_col].sum())
        tr_fraud_rate = (n_tr_fraud / n_tr) * 100

        records.append({
            "fold": f,
            "train_size": n_tr,
            "train_fraud": n_tr_fraud,
            "train_fraud_rate_%": round(tr_fraud_rate, 4),
            "val_size": n_val,
            "val_fraud": n_val_fraud,
            "val_fraud_rate_%": round(val_fraud_rate, 4),
        })

    summary_df = pd.DataFrame(records)
    return summary_df


def time_based_split(
    df: pd.DataFrame,
    id_col: str = "id",
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    buffer_size: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    Phân chia tập dữ liệu theo thứ tự thời gian (ID tăng dần) không xáo trộn.
    Hữu ích để làm Time-based Holdout Sanity Check trước khi nộp submission.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đầu vào.
    id_col : str, default='id'
        Cột xác định thứ tự thời gian.
    train_ratio : float, default=0.70
        Tỷ lệ phần trăm cho tập Train.
    valid_ratio : float, default=0.15
        Tỷ lệ phần trăm cho tập Validation.
    buffer_size : int, default=0
        Số lượng bản ghi bị loại bỏ (gap) giữa Train và Valid.

    Returns
    -------
    dict of {str: pd.DataFrame}
        Dictionary chứa các tập: 'train', 'valid', 'test_holdout'.
    """
    df_sorted = df.sort_values(by=id_col).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    if buffer_size > 0:
        train_df = df_sorted.iloc[: train_end - buffer_size].copy()
        valid_df = df_sorted.iloc[train_end : valid_end - buffer_size].copy()
        test_df = df_sorted.iloc[valid_end:].copy()
    else:
        train_df = df_sorted.iloc[:train_end].copy()
        valid_df = df_sorted.iloc[train_end:valid_end].copy()
        test_df = df_sorted.iloc[valid_end:].copy()

    return {
        "train": train_df,
        "valid": valid_df,
        "test_holdout": test_df,
    }


def summarize_splits(
    splits: Dict[str, pd.DataFrame],
    id_col: str = "id",
    target_col: str = "is_fraud",
) -> pd.DataFrame:
    """
    Tạo bảng báo cáo kích thước, khoảng ID và tỷ lệ nhãn của từng tập dữ liệu.
    """
    records = []
    for name, data in splits.items():
        n_rows = len(data)
        id_min = data[id_col].min() if id_col in data else None
        id_max = data[id_col].max() if id_col in data else None

        if target_col in data:
            n_pos = int(data[target_col].sum())
            rate_pct = round(float(data[target_col].mean()) * 100, 4)
        else:
            n_pos = None
            rate_pct = None

        records.append({
            "split": name,
            "size": n_rows,
            "id_min": id_min,
            "id_max": id_max,
            "fraud_count": n_pos,
            "fraud_rate_pct": rate_pct,
        })

    return pd.DataFrame(records)
