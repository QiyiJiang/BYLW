from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def compute_hit_rank_for_question(
    ranked_page_uids: List[str],
    gold_page_uids: Iterable[str],
) -> Optional[int]:
    """返回第一个命中 gold page 的 rank（1-based），未命中返回 None。"""
    gold_set = set(gold_page_uids)
    for idx, uid in enumerate(ranked_page_uids, start=1):
        if uid in gold_set:
            return idx
    return None


def compute_retrieval_metrics(
    results_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    ks: Tuple[int, ...] = (1, 3, 5),
    question_type_df: Optional[pd.DataFrame] = None,
    type_filter: Optional[str] = None,
) -> Dict[str, float]:
    """
    给定检索结果与 gold pages，计算 Recall@k 与 MRR。

    results_df: 每行包含 financebench_id, rank, page_uid。
    gold_df: 每行包含 financebench_id, gold_page_uids。
    question_type_df: 可选，包含 financebench_id, question_type；若提供 + type_filter，则只在该子集上计算。
    """
    merged = results_df.merge(gold_df, on="financebench_id", how="inner")
    if question_type_df is not None and type_filter is not None:
        merged = merged.merge(question_type_df, on="financebench_id", how="left")
        merged = merged[merged["question_type"] == type_filter]

    grouped = merged.groupby("financebench_id")

    hit_ranks: List[Optional[int]] = []
    for fid, group in grouped:
        group = group.sort_values("rank")
        ranked_uids = group["page_uid"].tolist()
        gold_uids = group.iloc[0]["gold_page_uids"]
        hit_rank = compute_hit_rank_for_question(ranked_uids, gold_uids)
        hit_ranks.append(hit_rank)

    n = len(hit_ranks)
    if n == 0:
        return {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0}

    metrics: Dict[str, float] = {}
    for k in ks:
        hits = sum(1 for r in hit_ranks if r is not None and r <= k)
        metrics[f"recall@{k}"] = hits / n
    # MRR
    mrr_sum = 0.0
    for r in hit_ranks:
        if r is not None:
            mrr_sum += 1.0 / float(r)
    metrics["mrr"] = mrr_sum / n
    return metrics

