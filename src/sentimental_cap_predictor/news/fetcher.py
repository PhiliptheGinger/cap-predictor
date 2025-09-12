"""Asynchronous HTML fetching with retry and back-off."""

from __future__ import annotations

import asyncio
import logging
import random

import httpx

from .store import log_error

logger = logging.getLogger(__name__)


class HtmlFetcher:
    """Fetch HTML pages concurrently with retry, logging and back-off."""

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
        """Return the body of ``url`` or ``None`` on failure.

        Each request attempt is logged. Transient failures trigger exponential
        back-off and are retried up to ``max_retries`` times. Final failures are
        persisted to the ``errors`` table via :func:`log_error`.
        """

        async with self._sem:
            delay = 0.5
            for attempt in range(1, max_retries + 1):
                logger.info("GET %s (attempt %s/%s)", url, attempt, max_retries)
                try:
                    resp = await self.client.get(url, follow_redirects=True)
                except (httpx.ProxyError, httpx.ConnectError, httpx.ReadTimeout) as exc:
                    logger.warning("Request error for %s: %s", url, exc)
                    resp = None
                if resp and resp.status_code == 200 and resp.text:
                    return resp.text
                reason = (
                    f"status {resp.status_code}" if resp is not None else "network error"
                )
                if attempt < max_retries and (
                    resp is None or resp.status_code in (403, 429)
                ):
                    sleep_for = delay + random.random()
                    logger.info(
                        "Retrying %s in %.2fs due to %s", url, sleep_for, reason
                    )
                    await asyncio.sleep(sleep_for)
                    delay *= 2
                    continue
                if resp is not None and resp.status_code not in (403, 429):
                    logger.warning("Giving up on %s due to %s", url, reason)
                    break
            logger.error("Failed to fetch %s after %s attempts", url, max_retries)
            log_error(url, "fetch", reason)
            return None

    async def aclose(self) -> None:
        await self.client.aclose()


__all__ = ["HtmlFetcher"]
