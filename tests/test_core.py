"""Invariants for the ledger, SQL metrics and simulator. Run: pytest -q   (or: python -m tests.test_core)"""
import sqlite3
import tempfile
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from loyalty_analytics import db, metrics, simulator as sim


@lru_cache(maxsize=1)
def _data():
    path = Path(tempfile.mkdtemp()) / "test.db"
    db.build_db(path, n_users=1200, seed=7)
    return path, metrics.load_data(path)


# ------------------------------------------------------------------ ledger / SQL
def test_ledger_never_goes_negative():
    path, _ = _data()
    with closing(sqlite3.connect(path)) as con:
        bad = con.execute("""
            SELECT COUNT(*) FROM (
              SELECT user_id,
                     SUM(CASE event_type WHEN 'earn' THEN points ELSE -points END) AS bal
              FROM reward_events GROUP BY user_id) WHERE bal < 0""").fetchone()[0]
    assert bad == 0


def test_fifo_matching_accounts_for_every_redeemed_point():
    _, d = _data()
    assert d.lags.points.sum() == d.users.redeemed.sum()
    assert (d.lags.lag_days >= 0).all()


def test_cohort_matrix_is_complete_where_observable():
    _, d = _data()
    pct, size = metrics.cohort_matrix(d)
    assert (pct[0] == 1.0).all()
    observable = np.add.outer(
        [pd.Period(i).year * 12 + pd.Period(i).month - 1 for i in pct.index], pct.columns.values) <= d.last_idx
    assert not pct.where(observable).isna().to_numpy()[observable].any()
    assert (pct.fillna(0) <= 1.0).to_numpy().all()


def test_breakage_uses_only_matured_lots():
    _, d = _data()
    b = metrics.breakage_by_tier(d)
    assert ((b.breakage_rate >= 0) & (b.breakage_rate <= 1)).all()
    assert (b.matured_earned.drop("All") <= d.users.groupby("tier").earned.sum()).all()


def test_burn_to_earn_and_breakage_are_consistent():
    _, d = _data()
    r = metrics.reward_summary(d).loc["All"]
    assert 0 < r.burn_to_earn < 1
    assert abs(r.earned - r.redeemed - r.expired - r.outstanding) < 1e-6


# ------------------------------------------------------------------ tier ROI
def test_reference_tier_has_zero_delta():
    _, d = _data()
    e = metrics.tier_economics(d, margin=0.25, discount=0.01)
    assert e.iloc[0].d_ltv_net == 0 and not e.iloc[0].pays_for_itself
    assert (e.reward_cost_ltv >= 0).all()


# ------------------------------------------------------------------ simulator
BASE = sim.Baseline(base_rate=0.02, retention=0.6, arpu=30.0, margin=0.25, breakage=0.4)
RESP = sim.Response(max_lift=0.10, steepness=0.05)


def test_expected_months_geometric():
    assert sim.expected_months(0.5) == 2.0
    assert sim.expected_months(0.0) == 1.0


def test_baseline_rate_reproduces_baseline_retention():
    assert abs(sim.retention_at(BASE.base_rate, BASE, RESP) - BASE.retention) < 1e-12


def test_retention_rises_and_saturates_with_rate():
    r = [sim.retention_at(x, BASE, RESP) for x in (0.02, 0.05, 0.10, 0.30)]
    assert r == sorted(r) and r[-1] <= BASE.retention + RESP.max_lift + 1e-9
    assert (r[1] - r[0]) / 0.03 > (r[3] - r[2]) / 0.20      # diminishing returns


def test_breakeven_retention_makes_net_ltv_equal():
    out = sim.simulate({"base": 0.02, "B": 0.10}, BASE, RESP)
    b = out.loc["B"]
    r_star = b.breakeven_retention
    ltv_at_star = (b.contribution_per_month - b.reward_cost_per_month) * sim.expected_months(r_star)
    assert abs(ltv_at_star - out.loc["base", "ltv_net"]) < 1e-9


def test_reward_cost_scales_with_rate_and_breakage():
    lo, hi = sim.evaluate(0.05, BASE, RESP), sim.evaluate(0.10, BASE, RESP)
    assert hi["reward_cost_per_month"] > lo["reward_cost_per_month"]
    assert abs(lo["reward_cost_per_month"] - 30 * 0.05 * 0.6) < 1e-9


def test_fit_recovers_known_response():
    truth = sim.Response(max_lift=0.12, steepness=0.06)
    rates = [0.05, 0.10, 0.15]
    rets = [sim.retention_at(x, BASE, truth) for x in rates]
    fit = sim.fit_response(BASE.base_rate, BASE.retention, rates, rets)
    assert all(abs(sim.retention_at(x, BASE, fit) - r) < 0.003 for x, r in zip(rates, rets))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok  ", name)
