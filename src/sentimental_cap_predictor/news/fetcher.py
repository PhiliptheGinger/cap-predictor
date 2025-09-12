"""Asynchronous HTML fetching with retry and back-off."""

from __future__ import annotations

import asyncio
import logging
import os
import random

import httpx

from .store import log_error

try:  # pragma: no cover - optional dependency
    from playwright.async_api import async_playwright
except Exception:  # pragma: no cover - optional dependency
    async_playwright = None

logger = logging.getLogger(__name__)


class HtmlFetcher:
    """Fetch HTML pages concurrently with retry, logging and back-off."""

    def __init__(
        self,
        timeout_s: float = 10.0,
        max_concurrency: int = 8,
        use_env_proxy: bool = False,
    ) -> None:
        limits = httpx.Limits(
            max_connections=20,
            max_keepalive_connections=20,
        )
        self.client = httpx.AsyncClient(
            timeout=timeout_s,
            headers={"User-Agent": "cap-predictor/1.0"},
            trust_env=use_env_proxy,
            limits=limits,
        )
        self._sem = asyncio.Semaphore(max_concurrency)

    async def get(self, url: str, *, max_retries: int = 3) -> str | None:
        """Return the body of ``url`` or ``None`` on failure.

        Each request attempt is logged. Transient failures trigger
        exponential back-off and are retried up to ``max_retries`` times.
        Final failures are persisted to the ``errors`` table via
        :func:`log_error`.
        """

        async with self._sem:
            delay = 0.5
            resp: httpx.Response | None = None
            for attempt in range(1, max_retries + 1):
                logger.info(
                    "GET %s (attempt %s/%s)",
                    url,
                    attempt,
                    max_retries,
                )
                try:
                    resp = await self.client.get(url, follow_redirects=True)
                except (
                    httpx.ProxyError,
                    httpx.ConnectError,
                    httpx.ReadTimeout,
                ) as exc:
                    logger.warning("Request error for %s: %s", url, exc)
                    resp = None
                if resp and resp.status_code == 200 and resp.text:
                    return resp.text
                reason = (
                    f"status {resp.status_code}"
                    if resp is not None
                    else "network error"
                )
                if attempt < max_retries and (
                    resp is None or resp.status_code in (403, 429)
                ):
                    sleep_for = delay + random.random()
                    logger.info(
                        "Retrying %s in %.2fs due to %s",
                        url,
                        sleep_for,
                        reason,
                    )
                    await asyncio.sleep(sleep_for)
                    delay *= 2
                    continue
                if resp is not None and resp.status_code not in (403, 429):
                    logger.warning("Giving up on %s due to %s", url, reason)
                    break

            use_browser = os.getenv("NEWS_USE_PLAYWRIGHT", "0") == "1"
            blocked = resp is None or resp.status_code in (403, 429)
            if use_browser and blocked:
                if async_playwright is None:
                    msg = "Playwright not installed; skipping browser fetch for %s"  # noqa: E501
                    logger.warning(msg, url)
                else:
                    try:
                        logger.info("Using Playwright to fetch %s", url)
                        timeout = getattr(self.client.timeout, "read", 10.0)
                        html = await _fetch_with_playwright(url, timeout)
                        if html:
                            return html
                    except Exception as exc:  # pragma: no cover
                        logger.warning(
                            "Playwright fetch failed for %s: %s",
                            url,
                            exc,
                        )

            logger.error(
                "Failed to fetch %s after %s attempts",
                url,
                max_retries,
            )
            log_error(url, "fetch", reason)
            return None

    async def aclose(self) -> None:
        await self.client.aclose()


async def _fetch_with_playwright(url: str, timeout_s: float) -> str | None:
    """Return ``url`` content via Playwright or ``None`` on failure."""

    if async_playwright is None:  # pragma: no cover - sanity check
        return None
    timeout_ms = int(timeout_s * 1000)
    async with async_playwright() as pw:  # pragma: no cover - network/browser
        browser = await pw.firefox.launch()
        page = await browser.new_page()
        try:
            await page.goto(url, timeout=timeout_ms)
            return await page.content()
        finally:
            await browser.close()


__all__ = ["HtmlFetcher"]
