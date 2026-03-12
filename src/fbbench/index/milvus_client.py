from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)
from pymilvus.milvus_client.index import IndexParams

from fbbench.utils.io_utils import ensure_dir


@dataclass
class MilvusConfig:
    uri: str  # e.g. "data/milvus/db.db" for local lite


def build_lite_uri(db_path: Path) -> str:
    """
    构造 Milvus Lite 本地存储 URI。

    pymilvus 要求本地文件路径必须以 .db 结尾，或者使用 http/unix 等协议。
    这里统一将路径标准化为以 .db 结尾的本地文件。
    """
    if db_path.suffix != ".db":
        db_path = db_path.with_suffix(".db")
    ensure_dir(db_path.parent)
    # 直接返回本地文件路径，例如 "data/milvus/db.db"
    return str(db_path)


def connect_milvus(db_path: Path) -> MilvusClient:
    uri = build_lite_uri(db_path)
    client = MilvusClient(uri=uri)
    return client


class MilvusIndex:
    """
    对 Milvus Lite 的简单封装，统一管理 collection 创建与向量检索。
    这里只实现向量索引部分，BM25 仍由 rank-bm25 负责。
    """

    def __init__(self, client: MilvusClient) -> None:
        self.client = client

    def _ensure_vector_collection(
        self,
        collection_name: str,
        dim: int,
    ) -> None:
        if self.client.has_collection(collection_name):
            return

        schema = CollectionSchema(
            fields=[
                FieldSchema(
                    name="page_uid",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=256,
                ),
                FieldSchema(
                    name="doc_name",
                    dtype=DataType.VARCHAR,
                    max_length=256,
                ),
                FieldSchema(
                    name="page_id",
                    dtype=DataType.INT64,
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=dim,
                ),
            ],
            description=f"{collection_name} page-level vector index",
        )
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
        )
        # 创建简单的向量索引
        index_params = IndexParams()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            index_name=f"{collection_name}_embedding_idx",
            metric_type="COSINE",
            params={"nlist": 1024},
        )
        self.client.create_index(
            collection_name=collection_name,
            index_params=index_params,
        )

    def upsert_vectors(
        self,
        collection_name: str,
        page_uids: Sequence[str],
        doc_names: Sequence[str],
        page_ids: Sequence[int],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if not page_uids:
            return
        dim = len(vectors[0])
        self._ensure_vector_collection(collection_name, dim)

        # MilvusClient.insert 期望的是“行列表”，每一行是一个字段到值的字典
        rows = []
        for uid, doc, pid, vec in zip(page_uids, doc_names, page_ids, vectors):
            rows.append(
                {
                    "page_uid": str(uid),
                    "doc_name": str(doc),
                    "page_id": int(pid),
                    "embedding": list(vec),
                }
            )
        self.client.insert(collection_name=collection_name, data=rows)

    def search_vectors(
        self,
        collection_name: str,
        query_vectors: Sequence[Sequence[float]],
        top_k: int,
    ) -> List[List[Dict[str, Any]]]:
        if not self.client.has_collection(collection_name):
            return [[] for _ in query_vectors]

        results = self.client.search(
            collection_name=collection_name,
            data=list(query_vectors),
            limit=top_k,
            output_fields=["page_uid", "doc_name", "page_id"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
        )
        # 转成更方便的结构
        all_hits: List[List[Dict[str, Any]]] = []
        for hits in results:
            cur: List[Dict[str, Any]] = []
            for hit in hits:
                cur.append(
                    {
                        "page_uid": hit.entity.get("page_uid"),
                        "doc_name": hit.entity.get("doc_name"),
                        "page_id": int(hit.entity.get("page_id")),
                        "score": float(hit.score),
                    }
                )
            all_hits.append(cur)
        return all_hits

