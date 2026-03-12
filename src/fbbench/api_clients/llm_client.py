from __future__ import annotations

from typing import List

from fbbench.api_clients.base_client import ApiConfig, BaseApiClient


class LlmClient(BaseApiClient):
    """
    兼容 OpenAI /v1/chat/completions 风格的 Qwen 客户端。
    """

    def generate_answer(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """
        调用 DeepSeek（经由火山方舟 OpenAI 接口）的 chat.completions。

        参考文档：https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions
        这里沿用 OpenAI 风格的 messages 格式。
        """
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self.post_json("/chat/completions", payload)
        return data["choices"][0]["message"]["content"].strip()


