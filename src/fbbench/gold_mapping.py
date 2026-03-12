from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

from fbbench.utils.io_utils import read_table, write_table


def build_gold_page_mapping(
    questions_path: Path,
) -> pd.DataFrame:
    """
    根据问题表中的 evidence 字段，构建 gold page 映射表。

    要求 questions 表包含：
    - financebench_id
    - evidence: 列表，每个元素包含 evidence_doc_name, evidence_page_num（0-indexed）
    """
    q_df = read_table(questions_path)

    rows: List[Dict[str, object]] = []
    for _, row in q_df.iterrows():
        fid = row.get("financebench_id")
        raw = row.get("evidence")

        # 将 evidence 统一转换为 Python 列表，避免对数组做布尔判断
        if raw is None:
            evidence_list: List[Dict[str, object]] = []
        elif isinstance(raw, list):
            evidence_list = raw
        else:
            # 例如 numpy 数组 / pandas Series，转为列表
            try:
                evidence_list = list(raw)
            except TypeError:
                evidence_list = []

        page_uids: Set[str] = set()
        pairs: Set[Tuple[str, int]] = set()
        for ev in evidence_list:
            # 防御性：确保 ev 是 dict
            if not isinstance(ev, dict):
                continue
            doc_name = ev.get("doc_name") or ev.get("evidence_doc_name")
            page_num = ev.get("evidence_page_num")
            if doc_name is None or page_num is None:
                continue
            page_idx = int(page_num)
            pairs.add((doc_name, page_idx))
            page_uids.add(f"{doc_name}_{page_idx}")

        rows.append(
            {
                "financebench_id": fid,
                "gold_pairs": list(pairs),
                "gold_page_uids": list(page_uids),
            }
        )

    gold_df = pd.DataFrame(rows)
    return gold_df


def save_gold_page_mapping(
    questions_path: Path,
    output_path: Path,
) -> None:
    gold_df = build_gold_page_mapping(questions_path)
    write_table(gold_df, output_path)

