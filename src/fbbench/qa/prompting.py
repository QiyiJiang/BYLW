from __future__ import annotations

from typing import List

import pandas as pd


def format_evidence_pages(pages: List[dict]) -> str:
    """
    将页面上下文组织为统一格式文本：
    [Document: {doc_name} | Page: {page_id}]
    {page_text}
    """
    parts = []
    for p in pages:
        header = f"[Document: {p['doc_name']} | Page: {p['page_id']}]"
        parts.append(header)
        parts.append(str(p["page_text"]))
        parts.append("")  # 空行分隔
    return "\n".join(parts)


def build_prompt(question: str, pages: List[dict]) -> str:
    retrieved_pages_text = format_evidence_pages(pages) if pages else ""
    prompt = (
        "You are a financial document QA assistant.\n"
        "Answer the question only based on the provided evidence pages.\n"
        'If the evidence is insufficient, answer "Insufficient evidence".\n'
        "Give a concise answer first, then provide the supporting page numbers.\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Evidence Pages:\n"
        f"{retrieved_pages_text}"
    )
    return prompt

