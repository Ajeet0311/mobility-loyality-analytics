"""Loyalty-policy simulator.

Model (deliberately simple, every assumption is an explicit input):
  * An active rider stays active next month with probability r  ->  expected life = 1 / (1 - r).
  * A reward rate x lifts retention with diminishing returns:
        r(x) = r_base + max_lift * (1 - exp(-(x - x_base) / steepness))
  * Per active month a rider spends `arpu` (gross fares) and earns arpu * x in points.
    Points that expire are free, so reward cost = arpu * x * (1 - breakage).
  * LTV (net) = (arpu * margin - reward cost) * expected life.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Baseline:
    base_rate: float      # reward rate of the reference tier (e.g. 0.02)
    retention: float      # monthly retention observed at base_rate
    arpu: float           # gross fares per active user-month
    margin: float         # contribution margin on fares, before rewards
    breakage: float       # share of earned points that expire unredeemed
    discount: float = 0.0  # monthly discount rate for LTV


@dataclass(frozen=True)
class Response:
    max_lift: float       # ceiling on the retention lift (0.08 = +8pp)
    steepness: float      # reward-rate distance at which 63% of max_lift is reached


def expected_months(retention: float, discount: float = 0.0) -> float:
    """Expected number of active months under geometric retention (counts the first month)."""
    return 1.0 / (1.0 - retention / (1.0 + discount))


def retention_at(rate: float, base: Baseline, resp: Response) -> float:
    lift = resp.max_lift * (1.0 - np.exp(-(rate - base.base_rate) / resp.steepness))
    return float(np.clip(base.retention + lift, 0.01, 0.99))


def breakeven_retention(monthly_net: float, target_ltv: float, discount: float = 0.0) -> float:
    """Retention at which monthly_net * expected_months(r) equals target_ltv (may exceed 1 = impossible)."""
    return (1.0 + discount) * (1.0 - monthly_net / target_ltv)


def evaluate(rate: float, base: Baseline, resp: Response, cohort_size: int = 10_000) -> dict:
    r = retention_at(rate, base, resp)
    life = expected_months(r, base.discount)
    contribution_pm = base.arpu * base.margin
    reward_pm = base.arpu * rate * (1.0 - base.breakage)
    return {
        "reward_rate": rate,
        "retention": r,
        "expected_months": life,
        "retained_after_12m": cohort_size * r ** 12,
        "contribution_per_month": contribution_pm,
        "reward_cost_per_month": reward_pm,
        "ltv_gross": contribution_pm * life,
        "reward_cost_per_user": reward_pm * life,
        "ltv_net": (contribution_pm - reward_pm) * life,
        "cohort_reward_cost": reward_pm * life * cohort_size,
        "cohort_net_value": (contribution_pm - reward_pm) * life * cohort_size,
    }


def simulate(scenarios: dict[str, float], base: Baseline, resp: Response,
             cohort_size: int = 10_000) -> pd.DataFrame:
    """One row per scenario; the first scenario is the comparison baseline."""
    rows = []
    for name, rate in scenarios.items():
        rows.append({"scenario": name, **evaluate(rate, base, resp, cohort_size)})
    df = pd.DataFrame(rows).set_index("scenario")
    ref = df.iloc[0]
    df["d_retention_pp"] = (df.retention - ref.retention) * 100
    df["d_reward_cost_per_user"] = df.reward_cost_per_user - ref.reward_cost_per_user
    df["d_ltv_net"] = df.ltv_net - ref.ltv_net
    df["breakeven_retention"] = [
        breakeven_retention(r.contribution_per_month - r.reward_cost_per_month, ref.ltv_net, base.discount)
        for r in df.itertuples()
    ]
    df["pays_for_itself"] = df.d_ltv_net > 0
    return df


def sweep(rates, base: Baseline, resp: Response) -> pd.DataFrame:
    return pd.DataFrame([evaluate(float(x), base, resp, 1) for x in rates])


def fit_response(base_rate: float, base_retention: float, rates, retentions) -> Response:
    """Least-squares fit of (max_lift, steepness) to observed (reward rate, retention) points above base."""
    rates, rets = np.asarray(rates, float), np.asarray(retentions, float)
    keep = rates > base_rate
    lift_grid = np.linspace(0.005, 0.30, 150)[:, None]
    steep_grid = np.linspace(0.01, 0.40, 150)[None, :]
    sse = np.zeros((150, 150))
    for x, r in zip(rates[keep], rets[keep]):
        pred = lift_grid * (1.0 - np.exp(-(x - base_rate) / steep_grid))
        sse += (pred - (r - base_retention)) ** 2
    i, j = np.unravel_index(np.argmin(sse), sse.shape)
    return Response(float(lift_grid[i, 0]), float(steep_grid[0, j]))
