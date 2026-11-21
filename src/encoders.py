"""
Encoder utilities dùng trong cross-validation.

Các hàm này được thiết kế để gọi BÊN TRONG từng fold nhằm tránh data leakage:
- frequency_encode(): tính frequency map chỉ từ train fold.
- make_transformer(): tạo ColumnTransformer với các chiến lược encoding khác nhau.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src.features import RAW_CATEGORICAL_COLUMNS, INTERACTION_COLUMNS

RANDOM_STATE = 42


def frequency_encode(
    train_fold,
    valid_fold,
    columns: list[str] | None = None,
):
    """
    Thêm frequency encoding cho các cột categorical.

    Frequency map được tính **chỉ trên train_fold** rồi áp dụng
    sang valid_fold (và sau này là tập test), tránh leakage.

    Parameters
    ----------
    train_fold, valid_fold : pd.DataFrame
        Hai phần của một fold cross-validation.
    columns : list[str] | None
        Danh sách cột cần encode. Mặc định là
        RAW_CATEGORICAL_COLUMNS + INTERACTION_COLUMNS.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        train_fold và valid_fold đã được thêm cột ``{col}_frequency``.
    """
    if columns is None:
        columns = RAW_CATEGORICAL_COLUMNS + INTERACTION_COLUMNS

    train_encoded = train_fold.copy()
    valid_encoded = valid_fold.copy()

    for col in columns:
        freq_map = train_encoded[col].astype(str).value_counts(normalize=True)
        train_encoded[f"{col}_frequency"] = (
            train_encoded[col].astype(str).map(freq_map).fillna(0.0).astype(np.float32)
        )
        valid_encoded[f"{col}_frequency"] = (
            valid_encoded[col].astype(str).map(freq_map).fillna(0.0).astype(np.float32)
        )

    return train_encoded, valid_encoded


def make_transformer(
    numeric_columns: list[str],
    encoding: str = "onehot",
) -> ColumnTransformer:
    """
    Tạo ColumnTransformer với chiến lược encoding được chỉ định.

    Parameters
    ----------
    numeric_columns : list[str]
        Danh sách tên các cột số (sẽ được StandardScaler xử lý).
    encoding : {"onehot", "target", "frequency"}
        - ``"onehot"``   : OneHotEncoder cho các cột categorical.
        - ``"target"``   : TargetEncoder cho các cột categorical.
        - ``"frequency"``: Không encode categorical ở đây — frequency
          encoding phải được thực hiện trước bằng
          :func:`frequency_encode`, rồi truyền các cột tần suất
          vào ``numeric_columns``.

    Returns
    -------
    ColumnTransformer
    """
    categorical_columns = RAW_CATEGORICAL_COLUMNS + INTERACTION_COLUMNS

    if encoding == "target":
        categorical_transformer = TargetEncoder(
            target_type="binary", smooth="auto", random_state=RANDOM_STATE
        )
    elif encoding == "frequency":
        # Frequency encoding đã được thực hiện ngoài pipeline;
        # không cần xử lý categorical ở đây.
        categorical_columns = []
        categorical_transformer = None
    else:  # "onehot"
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore", sparse_output=True, dtype=np.float32
        )

    transformers = [("numeric", StandardScaler(with_mean=False), numeric_columns)]
    if categorical_transformer is not None:
        transformers.append(("categorical", categorical_transformer, categorical_columns))

    return ColumnTransformer(transformers=transformers, sparse_threshold=0.3)
