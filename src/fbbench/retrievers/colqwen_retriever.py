from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict

import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from sentence_transformers import SentenceTransformer

from fbbench.index.milvus_client import MilvusIndex, connect_milvus
from fbbench.utils.config_loader import load_yaml_with_env
from fbbench.retrievers.interfaces import BaseRetriever, RetrievalResult
from fbbench.utils.io_utils import read_table
from fbbench.api_clients.base_client import ApiConfig
from fbbench.retrievers.bm25_retriever import Bm25Retriever
from fbbench.retrievers.bge_retriever import BgeRetriever


class ColQwenRetriever(BaseRetriever):
    """
    使用本地 ColQwen2.5 模型做页面图像检索，向量索引存储在 Milvus Lite 中。

    模型链接: https://huggingface.co/vidore/colqwen2.5-v0.2
    """

    def __init__(
        self,
        pages_path: Path,
        paths_config_path: Path,
        api_config: Optional[ApiConfig] = None,
        model_name: str = "vidore/colqwen2.5-v0.2",
        device: Optional[str] = None,
    ) -> None:
        self.pages_df = read_table(pages_path)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ColQwen2_5.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device,
        ).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(self.model_name)

        paths_cfg = load_yaml_with_env(paths_config_path)
        milvus_db_path = Path(paths_cfg["milvus"]["db_path"])
        self.milvus_index = MilvusIndex(connect_milvus(milvus_db_path))

        # 额外初始化 BM25 与 BGE，用于后续的融合检索（hybrid retrieval）。
        self.bm25 = Bm25Retriever(pages_path)
        self.bge: Optional[BgeRetriever] = None
        if api_config is not None:
            self.bge = BgeRetriever(pages_path, api_config, paths_config_path)

        # 预先构建一个 page_uid -> 行索引 的映射，方便根据 page_uid 找到文本内容做 rerank。
        self._page_index_by_uid: Dict[str, int] = {
            str(row["page_uid"]): int(idx) for idx, row in self.pages_df.iterrows()
        }

        # 文本 rerank 模型（BGE-large-zh，sentence-transformers 版本），用于对候选页面做精排。
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rerank_model = SentenceTransformer("BAAI/bge-large-zh-v1.5", device=device)

    def _encode_query_multi(self, question: str) -> torch.Tensor:
        """
        返回 ColQwen 对 query 的 multi-vector 表示，形状为 [M, D]。
        后续在 Python 侧实现类似 MaxSim 的 late interaction 聚合。
        """
        queries = [question]
        batch_queries = self.processor.process_queries(queries).to(self.model.device)
        with torch.no_grad():
            q_embs = self.model(**batch_queries)
        # q_embs: [1, M, D] -> [M, D]
        return q_embs[0].detach().cpu()

    def _retrieve_colqwen_only(self, question: str, top_k: int) -> List[RetrievalResult]:
        # 1) 使用 ColQwen 输出 query 的 multi-vector 表示 [M, D]
        q_multi = self._encode_query_multi(question)
        num_q_tokens = q_multi.shape[0]

        # 2) 对每个 query token 向量分别在 Milvus 中检索，近似实现 MaxSim：
        #    - 对于第 i 个 query token，Milvus 会返回若干文档子向量（page_uid::mvK）的 top_k 命中，
        #      我们把这些命中作为该 token 在各个 page 上的近似最大相似度。
        # 为了尽量提高召回，这里对每个 query token 从 Milvus 中取更大的候选集。
        hits_batches = self.milvus_index.search_vectors(
            collection_name="pages_colqwen",
            query_vectors=q_multi.tolist(),
            top_k=top_k * 8,  # 每个 token 多取一些，后续再按页面聚合
        )

        # 3) 先为每个 page、每个 query token 记录「该 token 对该 page 的最大 score」
        #    page_token_scores[page_uid][i] = max score
        page_token_scores: Dict[str, Dict[int, float]] = {}
        page_meta: Dict[str, tuple[str, int]] = {}

        if hits_batches:
            for token_idx, hits in enumerate(hits_batches):
                for h in hits:
                    raw_uid = str(h["page_uid"])  # 形如 page_uid::mvK
                    base_uid = raw_uid.split("::", 1)[0]
                    score = float(h["score"])
                    doc_name = str(h["doc_name"])
                    page_id = int(h["page_id"])

                    if base_uid not in page_token_scores:
                        page_token_scores[base_uid] = {}
                        page_meta[base_uid] = (doc_name, page_id)

                    cur_best = page_token_scores[base_uid].get(token_idx)
                    if cur_best is None or score > cur_best:
                        page_token_scores[base_uid][token_idx] = score

        # 4) 对每个 page，将所有 query token 的最大相似度求和，作为最终得分
        results: List[RetrievalResult] = []
        for page_uid, token_scores in page_token_scores.items():
            # 只对出现过命中的 token 求和，相当于 MaxSim 的近似实现
            total_score = sum(token_scores.values())
            doc_name, page_id = page_meta[page_uid]
            results.append(
                RetrievalResult(
                    page_uid=page_uid,
                    score=total_score,
                    doc_name=doc_name,
                    page_id=page_id,
                )
            )

        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]

    def retrieve(self, question: str, top_k: int) -> List[RetrievalResult]:
        """
        对外暴露的 ColQwen 检索接口。

        逻辑：
        1. 使用 ColQwen multi-vector + Milvus 做图像多向量检索；
        2. 同时使用 BM25 / BGE 做文本检索；
        3. 将三个检索结果做 rank-based 融合（Reciprocal Rank Fusion），输出统一的 top_k。

        对调用方而言，方法名仍然是 "colqwen"，但内部实际上已经是 hybrid 检索。
        """
        # 为了融合时有更丰富的候选集，这里每个子检索器取的 K 明显大于最终 top_k
        fusion_k = max(top_k * 10, top_k)

        # ColQwen 图像多向量检索
        colqwen_results = self._retrieve_colqwen_only(question, top_k=fusion_k)

        # BM25 文本检索
        bm25_results = self.bm25.retrieve(question, top_k=fusion_k)

        # BGE 检索（如果可用）
        bge_results: List[RetrievalResult] = []
        if self.bge is not None:
            bge_results = self.bge.retrieve(question, top_k=fusion_k)

        # 使用 Reciprocal Rank Fusion (RRF) 做简单稳健的 rank 融合：
        # score(page) = sum_m 1 / (k0 + rank_m(page))
        k0 = 60.0

        def collect_scores(results: List[RetrievalResult], weight: float = 1.0) -> Dict[str, float]:
            scores: Dict[str, float] = {}
            for rank, r in enumerate(results, start=1):
                inc = weight * (1.0 / (k0 + rank))
                scores[r.page_uid] = scores.get(r.page_uid, 0.0) + inc
            return scores

        fused_scores: Dict[str, float] = {}
        meta: Dict[str, RetrievalResult] = {}

        def merge_from(results: List[RetrievalResult], weight: float = 1.0) -> None:
            nonlocal fused_scores, meta
            partial = collect_scores(results, weight=weight)
            for uid, s in partial.items():
                fused_scores[uid] = fused_scores.get(uid, 0.0) + s
            for r in results:
                # 任一检索器都把 meta 补全成同一页的信息即可
                if uid := r.page_uid:
                    if uid not in meta:
                        meta[uid] = r

        # 三路融合：这里可以根据需要调整权重（目前都设为 1.0）
        merge_from(colqwen_results, weight=1.0)
        merge_from(bm25_results, weight=1.0)
        if bge_results:
            merge_from(bge_results, weight=1.0)

        # 先根据融合分数取一个较大的候选集，再用 rerank 模型做精排
        candidate_uids = sorted(
            fused_scores.keys(), key=lambda u: fused_scores[u], reverse=True
        )
        # rerank 候选集规模，可以按需调整，这里限制在最多 100 个页面
        max_rerank_candidates = 100
        candidate_uids = candidate_uids[:max_rerank_candidates]

        # 准备 rerank 所需的文本：使用页面的 page_text 作为 passage
        passages: List[str] = []
        valid_uids: List[str] = []
        for uid in candidate_uids:
            idx = self._page_index_by_uid.get(uid)
            if idx is None:
                continue
            row = self.pages_df.iloc[idx]
            text = str(row.get("page_text") or "")
            if not text:
                # 若文本为空，仍然保留，但用一个占位符，避免编码报错
                text = "空页面内容"
            passages.append(text)
            valid_uids.append(uid)

        if not valid_uids:
            # 兜底：如果因为某些原因没有可 rerank 的文本，就退回到融合分数排序
            fallback_uids = candidate_uids[:top_k]
            fallback_results: List[RetrievalResult] = []
            for uid in fallback_uids:
                base = meta[uid]
                fallback_results.append(
                    RetrievalResult(
                        page_uid=base.page_uid,
                        score=fused_scores[uid],
                        doc_name=base.doc_name,
                        page_id=base.page_id,
                    )
                )
            return fallback_results

        # 使用 BGE rerank 模型对 (question, passage) 进行打分：
        # 这里采用 sentence-transformers 的 encode 方式，计算点积相似度
        q_emb = self.rerank_model.encode([question], normalize_embeddings=False)  # [1, D]
        p_emb = self.rerank_model.encode(passages, normalize_embeddings=False)  # [N, D]
        # 相似度矩阵 [1, N]
        sim = (q_emb @ p_emb.T)[0]

        # 根据 rerank 分数重新排序候选页面
        rerank_pairs = list(zip(valid_uids, sim.tolist()))
        rerank_pairs.sort(key=lambda x: x[1], reverse=True)
        rerank_pairs = rerank_pairs[:top_k]

        final_results: List[RetrievalResult] = []
        for uid, score in rerank_pairs:
            base = meta[uid]
            final_results.append(
                RetrievalResult(
                    page_uid=base.page_uid,
                    score=float(score),
                    doc_name=base.doc_name,
                    page_id=base.page_id,
                )
            )

        return final_results

