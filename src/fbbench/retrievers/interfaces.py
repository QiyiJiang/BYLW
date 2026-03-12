from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class RetrievalResult:
    page_uid: str
    score: float
    doc_name: str
    page_id: int


class BaseRetriever(Protocol):
    def retrieve(self, question: str, top_k: int) -> List[RetrievalResult]:
        ...

