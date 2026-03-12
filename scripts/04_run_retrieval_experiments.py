import argparse
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from fbbench.eval.retrieval_metrics import compute_retrieval_metrics
from fbbench.utils.config_loader import load_yaml_with_env
from fbbench.gold_mapping import build_gold_page_mapping
from fbbench.retrievers.bge_retriever import BgeRetriever
from fbbench.retrievers.bm25_retriever import Bm25Retriever
from fbbench.retrievers.colqwen_retriever import ColQwenRetriever
from fbbench.retrievers.interfaces import BaseRetriever, RetrievalResult
from fbbench.utils.io_utils import ensure_dir, read_table, write_table
from fbbench.utils.logging_utils import setup_logger
from fbbench.api_clients.base_client import ApiConfig


def build_retriever(
    name: str,
    paths_cfg: dict,
    api_cfg: dict,
) -> BaseRetriever:
    pages_path = Path(paths_cfg["tables"]["pages"])
    paths_config_path = Path("config/paths.yaml")

    if name == "bm25":
        return Bm25Retriever(pages_path)
    if name == "bge":
        emb_cfg = api_cfg["embedding"]
        cfg = ApiConfig(
            base_url=emb_cfg["base_url"],
            model=emb_cfg["model"],
            api_key_env=emb_cfg["api_key_env"],
            timeout=emb_cfg.get("timeout", 60),
        )
        return BgeRetriever(pages_path, cfg, paths_config_path)
    if name == "colqwen":
        # ColQwen 多向量 + BM25 + BGE 的 hybrid 检索，
        # 其中 BGE 同样需要 embedding API 配置。
        emb_cfg = api_cfg["embedding"]
        cfg = ApiConfig(
            base_url=emb_cfg["base_url"],
            model=emb_cfg["model"],
            api_key_env=emb_cfg["api_key_env"],
            timeout=emb_cfg.get("timeout", 60),
        )
        return ColQwenRetriever(pages_path, paths_config_path, api_config=cfg)
    raise ValueError(f"Unknown retriever: {name}")


def run_retrieval_for_method(
    method: str,
    retriever: BaseRetriever,
    questions_df: pd.DataFrame,
    max_k: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    for _, row in tqdm(
        questions_df.iterrows(),
        total=len(questions_df),
        desc=f"Retrieval {method}",
    ):
        fid = row["financebench_id"]
        q_text = row["question"]
        results: List[RetrievalResult] = retriever.retrieve(q_text, top_k=max_k)
        for rank, r in enumerate(results, start=1):
            rows.append(
                {
                    "financebench_id": fid,
                    "method": method,
                    "rank": rank,
                    "page_uid": r.page_uid,
                    "score": r.score,
                    "doc_name": r.doc_name,
                    "page_id": r.page_id,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval experiments for BM25/BGE/ColQwen.")
    parser.add_argument("--paths-config", type=str, default="config/paths.yaml")
    parser.add_argument("--api-config", type=str, default="config/api.yaml")
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=["bm25", "bge", "colqwen"],
    )
    parser.add_argument(
        "--topk_list",
        nargs="+",
        type=int,
        default=[1, 3, 5],
    )
    args = parser.parse_args()

    logger = setup_logger("run_retrieval")

    paths_cfg = load_yaml_with_env(Path(args.paths_config))
    api_cfg = load_yaml_with_env(Path(args.api_config))

    questions_path = Path(paths_cfg["tables"]["questions"])
    results_dir = Path(paths_cfg["results"]["retrieval_dir"])
    ensure_dir(results_dir)

    questions_df = read_table(questions_path)
    logger.info("Loaded questions: %d", len(questions_df))

    gold_df = build_gold_page_mapping(questions_path)

    all_summary_rows = []
    max_k = max(args.topk_list)

    for method in args.retrievers:
        logger.info("Running retriever: %s", method)
        retriever = build_retriever(method, paths_cfg, api_cfg)
        result_df = run_retrieval_for_method(method, retriever, questions_df, max_k=max_k)
        raw_path = results_dir / f"{method}_raw_results.parquet"
        write_table(result_df, raw_path)
        logger.info("Saved raw results to %s", raw_path)

        metrics = compute_retrieval_metrics(
            results_df=result_df,
            gold_df=gold_df,
            ks=tuple(sorted(set(args.topk_list))),
        )
        summary_row = {
            "method": method,
            "Recall@1": metrics["recall@1"],
            "Recall@3": metrics["recall@3"],
            "Recall@5": metrics["recall@5"],
            "MRR": metrics["mrr"],
        }
        all_summary_rows.append(summary_row)

    summary_df = pd.DataFrame(all_summary_rows)
    summary_path = results_dir / "table_5_1.csv"
    write_table(summary_df, summary_path)
    logger.info("Saved summary metrics to %s", summary_path)


if __name__ == "__main__":
    main()

