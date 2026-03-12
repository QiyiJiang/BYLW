import argparse
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from fbbench.gold_mapping import build_gold_page_mapping
from fbbench.utils.config_loader import load_yaml_with_env
from fbbench.qa.qa_pipeline import (
    QaConfig,
    load_qa_config,
    load_llm_client,
    run_qa_for_question,
    select_evidence_pages,
)
from fbbench.utils.io_utils import ensure_dir, read_table, write_table
from fbbench.utils.logging_utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QA experiments for different retrieval methods.")
    parser.add_argument("--paths-config", type=str, default="config/paths.yaml")
    parser.add_argument("--api-config", type=str, default="config/api.yaml")
    parser.add_argument("--experiment-config", type=str, default="config/experiment.yaml")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["qwen_only", "bm25", "bge", "colqwen", "oracle"],
    )
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="并发处理问题的工作进程数（默认 20）",
    )
    args = parser.parse_args()

    logger = setup_logger("run_qa")

    paths_cfg = load_yaml_with_env(Path(args.paths_config))
    questions_path = Path(paths_cfg["tables"]["questions"])
    pages_path = Path(paths_cfg["tables"]["pages"])
    retrieval_dir = Path(paths_cfg["results"]["retrieval_dir"])
    qa_dir = Path(paths_cfg["results"]["qa_dir"])
    ensure_dir(qa_dir)

    questions_df = read_table(questions_path)
    pages_df = read_table(pages_path)
    gold_df = build_gold_page_mapping(questions_path)

    qa_cfg: QaConfig = load_qa_config(Path(args.experiment_config))
    if args.topk is not None:
        top_k = args.topk
    else:
        top_k = qa_cfg.default_topk

    llm = load_llm_client(Path(args.api_config))

    logger.info("Total questions: %d, Top-k=%d, num_workers=%d", len(questions_df), top_k, args.num_workers)

    for method in args.methods:
        logger.info("Running QA method: %s", method)
        rows = []

        # 加载对应检索结果（Qwen-only 与 Oracle 单独处理）
        retr_df = None
        if method in {"bm25", "bge", "colqwen"}:
            raw_path = retrieval_dir / f"{method}_raw_results.parquet"
            retr_df = read_table(raw_path)

        def _process_single(qrow):
            fid = qrow["financebench_id"]
            question = qrow["question"]

            evidence_pages = []
            evidence_page_uids: List[str] = []

            if method == "qwen_only":
                evidence_pages = []
                evidence_page_uids = []
            elif method == "oracle":
                gold_row = gold_df[gold_df["financebench_id"] == fid]
                if not gold_row.empty:
                    gold_uids = gold_row.iloc[0]["gold_page_uids"]
                    evidence_page_uids = list(gold_uids)
                    evidence_pages = select_evidence_pages(pages_df, evidence_page_uids, top_k=len(evidence_page_uids))
            else:
                assert retr_df is not None
                sub = retr_df[retr_df["financebench_id"] == fid].sort_values("rank")
                page_uids = sub["page_uid"].tolist()
                evidence_page_uids = page_uids[:top_k]
                evidence_pages = select_evidence_pages(pages_df, evidence_page_uids, top_k=top_k)

            qa_result = run_qa_for_question(
                llm=llm,
                qa_cfg=qa_cfg,
                question=question,
                evidence_pages=evidence_pages,
            )

            return {
                "financebench_id": fid,
                "method": method,
                "question": question,
                "pred_answer": qa_result["answer_text"],
                "model_raw_output": qa_result["raw_output"],
                "evidence_pages_used": evidence_page_uids,
                "evidence_pages_text": qa_result["evidence_pages_text"],
            }

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(_process_single, qrow): idx
                for idx, (_, qrow) in enumerate(questions_df.iterrows())
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"QA {method}"):
                rows.append(future.result())

        out_path = qa_dir / f"{method}_answers.parquet"
        write_table(pd.DataFrame(rows), out_path)
        logger.info("Saved QA outputs for %s to %s", method, out_path)


if __name__ == "__main__":
    main()

