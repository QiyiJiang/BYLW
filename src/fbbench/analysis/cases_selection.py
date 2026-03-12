from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from fbbench.utils.io_utils import read_table


def select_cases(
    questions_path: Path,
    retrieval_dir: Path,
    qa_dir: Path,
    gold_pages_path: Path,
) -> Dict[str, List[Dict]]:
    questions_df = read_table(questions_path)
    gold_df = read_table(gold_pages_path)

    bm25_retr = read_table(retrieval_dir / "bm25_raw_results.parquet")
    bge_retr = read_table(retrieval_dir / "bge_raw_results.parquet")
    col_retr = read_table(retrieval_dir / "colqwen_raw_results.parquet")

    qa_df = read_table(qa_dir / "colqwen_answers.parquet")  # 也可以加载其他方法

    # 构造便于查询的结构
    gold_map = {row["financebench_id"]: set(row["gold_page_uids"]) for _, row in gold_df.iterrows()}

    def hit_any_gold(sub_df: pd.DataFrame, fid) -> bool:
        uids = set(sub_df["page_uid"].tolist())
        return bool(uids & gold_map.get(fid, set()))

    cases_a = []
    cases_b = []
    cases_c = []

    for _, q in tqdm(
        questions_df.iterrows(),
        total=len(questions_df),
        desc="Select cases",
    ):
        fid = q["financebench_id"]
        q_text = q["question"]

        bm25_sub = bm25_retr[bm25_retr["financebench_id"] == fid].sort_values("rank").head(5)
        bge_sub = bge_retr[bge_retr["financebench_id"] == fid].sort_values("rank").head(5)
        col_sub = col_retr[col_retr["financebench_id"] == fid].sort_values("rank").head(5)

        bm25_hit = hit_any_gold(bm25_sub, fid)
        bge_hit = hit_any_gold(bge_sub, fid)
        col_hit = hit_any_gold(col_sub, fid)

        # 案例 A：BM25/BGE 均未命中，ColQwen 命中
        if not bm25_hit and not bge_hit and col_hit and len(cases_a) < 5:
            cases_a.append(
                {
                    "financebench_id": fid,
                    "question": q_text,
                    "bm25_top5": bm25_sub.to_dict(orient="records"),
                    "bge_top5": bge_sub.to_dict(orient="records"),
                    "colqwen_top5": col_sub.to_dict(orient="records"),
                }
            )

        # 案例 B：BM25/BGE 命中但排序靠后，ColQwen 排名更靠前
        if bm25_hit or bge_hit:
            # 找到各自第一个命中 rank
            def first_hit_rank(sub_df: pd.DataFrame, fid) -> int:
                gset = gold_map.get(fid, set())
                for _, r in sub_df.iterrows():
                    if r["page_uid"] in gset:
                        return int(r["rank"])
                return 9999

            bm25_rank = first_hit_rank(bm25_retr[bm25_retr["financebench_id"] == fid], fid)
            bge_rank = first_hit_rank(bge_retr[bge_retr["financebench_id"] == fid], fid)
            col_rank = first_hit_rank(col_retr[col_retr["financebench_id"] == fid], fid)

            if col_rank < min(bm25_rank, bge_rank) and min(bm25_rank, bge_rank) > 3 and len(cases_b) < 5:
                cases_b.append(
                    {
                        "financebench_id": fid,
                        "question": q_text,
                    "bm25_top5": bm25_sub.to_dict(orient="records"),
                    "bge_top5": bge_sub.to_dict(orient="records"),
                    "colqwen_top5": col_sub.to_dict(orient="records"),
                    }
                )

        # 案例 C：检索命中但回答仍出错（需要结合人工标签，留待脚本中与标签 join）

    return {
        "case_a": cases_a,
        "case_b": cases_b,
        "case_c": cases_c,
    }

