# FraudDetection

## Machine learning

Đặt các tệp dữ liệu Kaggle vào `data/`:

- `train.csv` phải có cột `is_fraud`.
- `test.csv` là dữ liệu cần dự đoán và phải có cột `id`.
- `sample_submission.csv` dùng để đối chiếu schema submission nếu cần.

Cài dependency và chạy validation:

```powershell
python -m pip install -r requirements.txt
python .\model_validation.py --data-dir .\data
```

Pipeline thực hiện:

1. So sánh one-hot, frequency encoding và target encoding; encoder được fit
   **chỉ trên train của từng fold**.
2. Tạo feature amount/velocity, giao dịch tròn, tỷ lệ balance/amount,
   prior disputes/card age và các interaction category.
3. So sánh baseline, `class_weight="balanced"` và SMOTE với cùng stratified folds;
   PR-AUC được dùng làm metric chính vì lớp fraud chiếm thiểu số.
4. Đánh giá bằng Stratified 5-fold với ROC-AUC và PR-AUC (PR-AUC phù hợp hơn
   khi fraud là lớp thiểu số).
5. Fit lại trên toàn bộ train và ghi xác suất fraud vào
   `submissions/submission.csv` nếu có `test.csv`.

File [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) dùng để kiểm tra schema,
missing values, mất cân bằng lớp và các dấu hiệu fraud trước khi train.
