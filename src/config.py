"""Central metric configuration shared by the modeling notebooks."""

PRIMARY_METRIC = "roc_auc"
SECONDARY_METRIC = "average_precision"
OUTPUT_VERSION = "v4"

METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "average_precision": "AP",
}

if PRIMARY_METRIC == SECONDARY_METRIC:
    raise ValueError("PRIMARY_METRIC and SECONDARY_METRIC must be different.")
if {PRIMARY_METRIC, SECONDARY_METRIC} != set(METRIC_LABELS):
    raise ValueError("Metric configuration must contain ROC-AUC and average precision.")
