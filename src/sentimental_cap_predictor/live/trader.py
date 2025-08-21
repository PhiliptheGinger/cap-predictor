from __future__ import annotations

from loguru import logger

from .alpaca_client import AlpacaClient


class Trader:
    """High level trading interface."""

    def __init__(
        self, client: AlpacaClient | None = None, dry_run: bool = True
    ) -> None:
        self.client = client or AlpacaClient(dry_run=dry_run)
        self.dry_run = dry_run

    def buy(self, symbol: str, qty: int):
        logger.debug("Buying %s x%s", symbol, qty)
        return self.client.submit_order(symbol=symbol, qty=qty, side="buy")

    def sell(self, symbol: str, qty: int):
        logger.debug("Selling %s x%s", symbol, qty)
        return self.client.submit_order(symbol=symbol, qty=qty, side="sell")
