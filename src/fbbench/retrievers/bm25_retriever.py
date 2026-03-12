from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from fbbench.retrievers.interfaces import BaseRetriever, RetrievalResult
from fbbench.utils.io_utils import read_table


def simple_tokenize(text: str) -> List[str]:
    return text.lower().split()


class Bm25Retriever(BaseRetriever):
    def __init__(self, pages_path: Path) -> None:
        self.pages_df = read_table(pages_path)
        corpus = [simple_tokenize(t or "") for t in self.pages_df["page_text"].astype(str).tolist()]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, question: str, top_k: int) -> List[RetrievalResult]:
        query_tokens = simple_tokenize(question)
        scores = np.array(self.bm25.get_scores(query_tokens), dtype="float32")
        top_idx = np.argsort(-scores)[:top_k]
        results: List[RetrievalResult] = []
        for idx in top_idx:
            row = self.pages_df.iloc[int(idx)]
            results.append(
                RetrievalResult(
                    page_uid=str(row["page_uid"]),
                    score=float(scores[int(idx)]),
                    doc_name=str(row["doc_name"]),
                    page_id=int(row["page_id"]),
                )
            )
        return results


