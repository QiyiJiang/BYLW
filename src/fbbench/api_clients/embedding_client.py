from __future__ import annotations

from typing import List, Literal

import numpy as np

from fbbench.api_clients.base_client import ApiConfig, BaseApiClient


MAX_CHARS_PER_TEXT = 4000  # 简单按字符截断，防止超过模型 token 上限


class EmbeddingClient(BaseApiClient):
    """
    硅基流动 /v1/embeddings 接口封装（兼容 OpenAI 风格）。

    参考文档：https://docs.siliconflow.cn/cn/api-reference/embeddings/create-embeddings
    """

    def embed_texts(
        self,
        texts: List[str],
        input_type: Literal["query", "document"],
    ) -> np.ndarray:
        """
        简单版本：对每条文本做字符级截断，然后直接请求 /embeddings。
        不再分 chunk，也不做池化。
        """
        if not texts:
            return np.zeros((0, 0), dtype="float32")

        # 粗略按字符截断，并避免空字符串（硅基接口不允许空字符串）
        cleaned_texts: List[str] = []
        for t in texts:
            s = (t or "").strip()
            if not s:
                s = "[EMPTY]"
            cleaned_texts.append(s[:MAX_CHARS_PER_TEXT])

        payload = {
            "model": self.config.model,
            "input": cleaned_texts,
            "encoding_format": "float",
        }
        data = self.post_json("/embeddings", payload)
        vectors = [item["embedding"] for item in data["data"]]
        return np.asarray(vectors, dtype="float32")


