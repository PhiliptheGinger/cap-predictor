from sentimental_cap_predictor.trader_utils.risk import enforce_limits


def test_enforce_limits_no_breach():
    pnl = [1.0, -1.0, 2.0, -0.5]
    states = enforce_limits(pnl, window=3, loss_limit=10.0)
    assert states == ["LIVE", "LIVE", "LIVE", "LIVE"]


def test_enforce_limits_loss_breach_sets_flat():
    pnl = [10.0, -5.0, -5.0, -5.0, 2.0]
    states = enforce_limits(pnl, window=3, loss_limit=10.0)
    assert states[3] == "FLAT"
    assert states[3:] == ["FLAT", "FLAT"]


def test_enforce_limits_profit_breach_sets_flat():
    pnl = [4.0, 4.0, 4.0]
    states = enforce_limits(pnl, window=3, loss_limit=10.0, profit_limit=10.0)
    assert states == ["LIVE", "LIVE", "FLAT"]
