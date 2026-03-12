from __future__ import annotations

from pathlib import Path
from typing import List

from fbbench.api_clients.base_client import ApiConfig
from fbbench.api_clients.embedding_client import EmbeddingClient
from fbbench.index.milvus_client import MilvusIndex, connect_milvus
from fbbench.retrievers.interfaces import BaseRetriever, RetrievalResult
from fbbench.utils.config_loader import load_yaml_with_env
from fbbench.utils.io_utils import read_table


class BgeRetriever(BaseRetriever):
    def __init__(
        self,
        pages_path: Path,
        api_config: ApiConfig,
        paths_config_path: Path,
    ) -> None:
        self.pages_df = read_table(pages_path)
        self.api_config = api_config
        self.client = EmbeddingClient(api_config)

        paths_cfg = load_yaml_with_env(paths_config_path)
        milvus_db_path = Path(paths_cfg["milvus"]["db_path"])
        self.milvus_index = MilvusIndex(connect_milvus(milvus_db_path))

    def retrieve(self, question: str, top_k: int) -> List[RetrievalResult]:
        q_vec = self.client.embed_texts([question], input_type="query")[0]
        hits_batches = self.milvus_index.search_vectors(
            collection_name="pages_bge",
            query_vectors=[q_vec.tolist()],
            top_k=top_k,
        )
        hits = hits_batches[0] if hits_batches else []
        # 由于索引中是 chunk 级别的向量，这里需要按页面聚合：取每个 page_uid 的最佳 chunk 作为该页的分数。
        best_by_page: dict[str, RetrievalResult] = {}
        for h in hits:
            raw_uid = str(h["page_uid"])
            base_uid = raw_uid.split("::", 1)[0]
            score = float(h["score"])
            doc_name = str(h["doc_name"])
            page_id = int(h["page_id"])
            cur = best_by_page.get(base_uid)
            if cur is None or score > cur.score:
                best_by_page[base_uid] = RetrievalResult(
                    page_uid=base_uid,
                    score=score,
                    doc_name=doc_name,
                    page_id=page_id,
                )
        # 根据 score 排序，取前 top_k 页
        return sorted(best_by_page.values(), key=lambda r: r.score, reverse=True)[:top_k]

