from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from fbbench.utils.io_utils import read_jsonl


def load_open_source_records(path: Path) -> List[Dict[str, Any]]:
    """读取 financebench_open_source.jsonl 原始记录。"""
    return read_jsonl(path)


def load_doc_info_records(path: Path) -> List[Dict[str, Any]]:
    """读取 financebench_document_information.jsonl 原始记录。"""
    return read_jsonl(path)


def build_questions_table(
    open_source_path: Path,
    doc_info_path: Path,
) -> pd.DataFrame:
    """
    按官方 README 的 join 方式，将 open source 样本与文档信息表联合成问题表。

    输出字段：
    - financebench_id
    - question
    - answer
    - evidence
    - question_type
    - question_reasoning
    - company
    - doc_name
    """
    records = load_open_source_records(open_source_path)
    docs = load_doc_info_records(doc_info_path)

    doc_df = pd.DataFrame(docs)
    # 假定文档信息表包含 document_id 和 document_name
    if "document_name" in doc_df.columns:
        doc_df = doc_df.rename(columns={"document_name": "doc_name"})

    rows = []
    for rec in records:
        row: Dict[str, Any] = {
            "financebench_id": rec.get("financebench_id"),
            "question": rec.get("question"),
            "answer": rec.get("answer"),
            "evidence": rec.get("evidence"),
            "question_type": rec.get("question_type"),
            "question_reasoning": rec.get("question_reasoning"),
            "company": rec.get("company"),
        }
        doc_name = None
        document_id = rec.get("document_id")
        if document_id is not None and "document_id" in doc_df.columns:
            doc_row = doc_df.loc[doc_df["document_id"] == document_id]
            if not doc_row.empty:
                doc_name = doc_row.iloc[0].get("doc_name")
        row["doc_name"] = doc_name
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

