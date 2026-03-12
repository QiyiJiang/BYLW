import argparse
from pathlib import Path

import pandas as pd

from fbbench.eval.qa_metrics import compute_qa_metrics
from fbbench.utils.config_loader import load_yaml_with_env
from fbbench.eval.retrieval_metrics import compute_retrieval_metrics
from fbbench.utils.io_utils import read_table, write_table, ensure_dir
from fbbench.utils.logging_utils import setup_logger
from fbbench.gold_mapping import build_gold_page_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Type-wise analysis for ColQwen vs BGE.")
    parser.add_argument("--paths-config", type=str, default="config/paths.yaml")
    args = parser.parse_args()

    logger = setup_logger("typewise_analysis")

    paths_cfg = load_yaml_with_env(Path(args.paths_config))
    questions_path = Path(paths_cfg["tables"]["questions"])
    retrieval_dir = Path(paths_cfg["results"]["retrieval_dir"])
    qa_dir = Path(paths_cfg["results"]["qa_dir"])

    out_path = Path(paths_cfg["results"]["qa_dir"]) / "table_5_4.csv"
    ensure_dir(out_path.parent)

    questions_df = read_table(questions_path)
    type_df = questions_df[["financebench_id", "question_type"]]
    gold_df = build_gold_page_mapping(questions_path)

    # 检索结果
    bge_retr = read_table(retrieval_dir / "bge_raw_results.parquet")
    col_retr = read_table(retrieval_dir / "colqwen_raw_results.parquet")

    # QA 结果（自动/人工标注），需要补充 question_type 信息
    qa_labels_path = Path(paths_cfg["results"]["annotations_dir"]) / "qa_labels.csv"
    qa_labels_df = read_table(qa_labels_path)
    # 将 questions 中的 question_type 合并进 QA 标注表
    qa_labels_df = qa_labels_df.merge(
        type_df,
        on="financebench_id",
        how="left",
    )

    type_values = questions_df["question_type"].unique().tolist()

    rows = []
    for qtype in type_values:
        logger.info("Processing question_type=%s", qtype)
        # 检索指标
        bge_metrics = compute_retrieval_metrics(
            results_df=bge_retr,
            gold_df=gold_df,
            ks=(1, 3, 5),
            question_type_df=type_df,
            type_filter=qtype,
        )
        col_metrics = compute_retrieval_metrics(
            results_df=col_retr,
            gold_df=gold_df,
            ks=(1, 3, 5),
            question_type_df=type_df,
            type_filter=qtype,
        )

        # QA 指标（过滤对应 question_type 与方法）
        subset_labels_bge = qa_labels_df[
            (qa_labels_df["question_type"] == qtype) & (qa_labels_df["method"] == "bge")
        ]
        subset_labels_col = qa_labels_df[
            (qa_labels_df["question_type"] == qtype) & (qa_labels_df["method"] == "colqwen")
        ]
        bge_qa = compute_qa_metrics(subset_labels_bge) if not subset_labels_bge.empty else {
            "accuracy": 0.0,
            "supported_accuracy": 0.0,
        }
        col_qa = compute_qa_metrics(subset_labels_col) if not subset_labels_col.empty else {
            "accuracy": 0.0,
            "supported_accuracy": 0.0,
        }

        rows.append(
            {
                "question_type": qtype,
                "method": "bge",
                "Recall@1": bge_metrics["recall@1"],
                "Recall@3": bge_metrics["recall@3"],
                "Recall@5": bge_metrics["recall@5"],
                "MRR": bge_metrics["mrr"],
                "Accuracy": bge_qa["accuracy"],
                "SupportedAccuracy": bge_qa["supported_accuracy"],
            }
        )
        rows.append(
            {
                "question_type": qtype,
                "method": "colqwen",
                "Recall@1": col_metrics["recall@1"],
                "Recall@3": col_metrics["recall@3"],
                "Recall@5": col_metrics["recall@5"],
                "MRR": col_metrics["mrr"],
                "Accuracy": col_qa["accuracy"],
                "SupportedAccuracy": col_qa["supported_accuracy"],
            }
        )

    out_df = pd.DataFrame(rows)
    write_table(out_df, out_path)
    logger.info("Saved type-wise analysis table to %s", out_path)


if __name__ == "__main__":
    main()

