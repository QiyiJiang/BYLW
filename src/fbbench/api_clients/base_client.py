from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


@dataclass
class ApiConfig:
    base_url: str
    model: str
    api_key_env: str
    timeout: int = 60


class BaseApiClient:
    def __init__(
        self,
        config: ApiConfig,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> None:
        self.config = config
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {config.api_key_env} is not set")
        self.api_key = api_key
        # 懒初始化客户端，避免在多进程中出问题
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        return self._client

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = path
        last_err: Optional[Exception] = None
        last_body: Optional[str] = None
        last_status: Optional[int] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.post(url, json=payload, headers=self._headers())
                last_status = resp.status_code
                last_body = resp.text
                resp.raise_for_status()
                return resp.json()
            except Exception as e:  # noqa: BLE001
                last_err = e
                sleep_time = self.backoff_factor ** (attempt - 1)
                time.sleep(sleep_time)
        raise RuntimeError(
            f"API request failed after {self.max_retries} retries: {last_err}; "
            f"last_status={last_status}, last_body={last_body}"
        )


