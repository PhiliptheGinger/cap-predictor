from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd


@dataclass
class BacktestResult:
    """Holds the outputs of a backtest run."""

    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    trade_pnls: pd.Series
    holding_periods: pd.Series

    def return_series(self) -> pd.Series:
        """Return the periodic returns derived from the equity curve."""
        return self.equity_curve.pct_change().fillna(0)

    def export_trades(self, path: str) -> None:
        """Export the trade log to ``path`` as CSV."""
        export_df = self.trades.copy()
        if len(self.trade_pnls) == len(export_df):
            export_df = export_df.assign(pnl=self.trade_pnls, holding_period=self.holding_periods)
        export_df.to_csv(path, index=False)
