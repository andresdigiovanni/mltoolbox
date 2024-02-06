import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_optimal_thresholds(y_true, y_pred_proba):
    thresholds = np.arange(0.1, 1.0, 0.01)
    thresholds = np.round(thresholds, 2)

    optimal_threshold_accuracy = None
    optimal_threshold_roc_auc = None
    optimal_threshold_precision = None
    optimal_threshold_recall = None
    optimal_threshold_f1_score = None

    optimal_accuracy = 0
    optimal_roc_auc = 0
    optimal_precision = 0
    optimal_recall = 0
    optimal_f1_score = 0

    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        if accuracy > optimal_accuracy:
            optimal_threshold_accuracy = threshold
            optimal_accuracy = accuracy

        if roc_auc > optimal_roc_auc:
            optimal_threshold_roc_auc = threshold
            optimal_roc_auc = roc_auc

        if precision > optimal_precision:
            optimal_threshold_precision = threshold
            optimal_precision = precision

        if recall > optimal_recall:
            optimal_threshold_recall = threshold
            optimal_recall = recall

        if f1 > optimal_f1_score:
            optimal_threshold_f1_score = threshold
            optimal_f1_score = f1

    return (
        optimal_threshold_accuracy,
        optimal_threshold_roc_auc,
        optimal_threshold_precision,
        optimal_threshold_recall,
        optimal_threshold_f1_score,
    )
