from __future__ import annotations

import os
from typing import Any, Dict

import requests
from loguru import logger


class AlpacaClient:
    """Minimal Alpaca REST client.

    Parameters
    ----------
    api_key, api_secret:
        Credentials used for authentication.  When omitted the values are
        pulled from the ``ALPACA_API_KEY`` and ``ALPACA_API_SECRET``
        environment variables.
    base_url:
        Alpaca REST endpoint.  Defaults to the paper trading API.
    dry_run:
        When ``True`` the client logs actions instead of performing real
        HTTP requests.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str | None = None,
        dry_run: bool = True,
    ) -> None:
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET", "")
        self.base_url = base_url or "https://paper-api.alpaca.markets"
        self.dry_run = dry_run

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        type: str = "market",
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """Submit an order to Alpaca.

        In ``dry_run`` mode the payload is logged and returned immediately.
        """

        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": type,
            "time_in_force": time_in_force,
        }
        if self.dry_run:
            logger.info("Dry run order: {}", payload)
            return {"dry_run": True, **payload}

        response = requests.post(
            f"{self.base_url}/v2/orders",
            json=payload,
            headers=self._headers(),
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
