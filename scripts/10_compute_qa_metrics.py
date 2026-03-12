import argparse
from pathlib import Path

import pandas as pd

from fbbench.eval.qa_metrics import compute_qa_metrics
from fbbench.utils.config_loader import load_yaml_with_env
from fbbench.utils.io_utils import ensure_dir, read_table, write_table
from fbbench.utils.logging_utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 qa_labels.csv 计算 QA Accuracy / SupportedAccuracy（表 5-2）。")
    parser.add_argument("--paths-config", type=str, default="config/paths.yaml")
    parser.add_argument(
        "--labels-path",
        type=str,
        default="results/annotations/qa_labels.csv",
        help="自动/人工标注结果路径（包含 label 和 evidence_supported）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/qa/table_5_2.csv",
        help="输出表 5-2 的 CSV 路径",
    )
    args = parser.parse_args()

    logger = setup_logger("compute_qa_metrics")

    paths_cfg = load_yaml_with_env(Path(args.paths_config))
    ensure_dir(Path(args.output).parent)

    labels_path = Path(args.labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    logger.info("Loading QA labels from %s", labels_path)
    labels_df = read_table(labels_path)

    # 期望至少包含这几列：financebench_id, method, label, evidence_supported
    required_cols = {"financebench_id", "method", "label", "evidence_supported"}
    missing = required_cols - set(labels_df.columns)
    if missing:
        raise ValueError(f"Labels file is missing required columns: {missing}")

    methods = sorted(labels_df["method"].unique().tolist())
    rows = []
    for method in methods:
        sub = labels_df[labels_df["method"] == method]
        metrics = compute_qa_metrics(sub)
        rows.append(
            {
                "method": method,
                "accuracy": metrics["accuracy"],
                "supported_accuracy": metrics["supported_accuracy"],
                "num_examples": len(sub),
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output)
    write_table(out_df, out_path)
    logger.info("Saved QA metrics (table 5-2) to %s", out_path)


if __name__ == "__main__":
    main()

