"""Asynchronous HTML fetching with retry and back-off."""

from __future__ import annotations

import asyncio
import random

import httpx


class HtmlFetcher:
    """Fetch HTML pages concurrently with basic retry handling."""

    def __init__(
        self,
        timeout_s: float = 10.0,
        max_concurrency: int = 8,
        use_env_proxy: bool = False,
    ) -> None:
        limits = httpx.Limits(max_connections=20, max_keepalive_connections=20)
        self.client = httpx.AsyncClient(
            timeout=timeout_s,
            headers={"User-Agent": "cap-predictor/1.0"},
            trust_env=use_env_proxy,
            limits=limits,
        )
        self._sem = asyncio.Semaphore(max_concurrency)

    async def get(self, url: str, *, max_retries: int = 3) -> str | None:
        """Return the body of ``url`` or ``None`` on failure."""

        async with self._sem:
            delay = 0.5
            for _ in range(max_retries):
                try:
                    resp = await self.client.get(url, follow_redirects=True)
                except (httpx.ProxyError, httpx.ConnectError, httpx.ReadTimeout):
                    resp = None
                if resp and resp.status_code == 200 and resp.text:
                    return resp.text
                if resp and resp.status_code in (403, 429):
                    await asyncio.sleep(delay + random.random())
                    delay *= 2
                    continue
                if resp is None:
                    await asyncio.sleep(delay + random.random())
                    delay *= 2
                else:
                    return None
            return None

    async def aclose(self) -> None:
        await self.client.aclose()


__all__ = ["HtmlFetcher"]
