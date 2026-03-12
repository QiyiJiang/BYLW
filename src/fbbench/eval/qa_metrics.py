from __future__ import annotations

from typing import Dict

import pandas as pd


def compute_qa_metrics(labels_df: pd.DataFrame) -> Dict[str, float]:
    """
    根据人工标注结果计算 Accuracy 与 SupportedAccuracy。

    要求 labels_df 包含：
    - label: correct / incorrect / insufficient / hallucinated
    - evidence_supported: bool
    """
    n = len(labels_df)
    if n == 0:
        return {"accuracy": 0.0, "supported_accuracy": 0.0}

    correct_mask = labels_df["label"] == "correct"
    accuracy = correct_mask.mean()

    supported_mask = correct_mask & labels_df["evidence_supported"].astype(bool)
    supported_accuracy = supported_mask.mean()
    return {
        "accuracy": float(accuracy),
        "supported_accuracy": float(supported_accuracy),
    }

