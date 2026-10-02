"""pandas layer: pull aggregates from SQL, then derive the KPIs the dashboard shows."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import db
from .config import DB_PATH, EXPIRY_DAYS, POINT_VALUE
from .simulator import Baseline, breakeven_retention, expected_months


@dataclass
class Data:
    as_of: pd.Timestamp
    last_idx: int                 # last fully observed calendar month (year*12 + month-1)
    expiry_days: int
    tiers: pd.DataFrame
    users: pd.DataFrame           # one row per user: ride stats + points totals
    monthly: pd.DataFrame
    cohort: pd.DataFrame
    mom: pd.DataFrame
    rewards_month: pd.DataFrame
    breakage: pd.DataFrame
    first_redeem: pd.DataFrame
    lags: pd.DataFrame            # FIFO redemption lots (see redemption_lags)

    @property
    def tier_order(self) -> list[str]:
        return list(self.tiers.tier)


# --------------------------------------------------------------------------- loading
def month_idx(ts: pd.Timestamp) -> int:
    return ts.year * 12 + ts.month - 1


def idx_label(idx) -> str:
    return f"{int(idx) // 12}-{int(idx) % 12 + 1:02d}"


def redemption_lags(events: pd.DataFrame) -> pd.DataFrame:
    """Match each redemption to the oldest unexpired earn lots (FIFO).

    Returns one row per (redemption, lot) slice: points taken from the lot and the
    days between earning and redeeming them.
    """
    out, books = [], {}
    for r in events.itertuples(index=False):
        book = books.setdefault(r.user_id, OrderedDict())     # lot_id -> [earn_ts, points_left]
        if r.event_type == "earn":
            book[r.event_id] = [r.event_ts, r.points]
        elif r.event_type == "expire":
            book.pop(r.lot_event_id, None)
        else:
            need = r.points
            while need > 0 and book:
                lot_id, lot = next(iter(book.items()))
                take = min(need, lot[1])
                out.append((r.user_id, r.tier, lot_id, lot[0], take,
                            (r.event_ts - lot[0]).total_seconds() / 86400))
                lot[1] -= take
                need -= take
                if lot[1] == 0:
                    book.popitem(last=False)
    return pd.DataFrame(out, columns=["user_id", "tier", "lot_id", "lot_ts", "points", "lag_days"])


def load_data(path: Path = DB_PATH, expiry_days: int = EXPIRY_DAYS) -> Data:
    as_of = pd.Timestamp(db.run("as_of", path).loc[0, "as_of"])
    month_end = (as_of + pd.offsets.MonthEnd(0)).normalize() == as_of.normalize()
    last_idx = month_idx(as_of) if month_end else month_idx(as_of) - 1   # drop a partial month
    p = dict(last_idx=last_idx)

    users = db.run("user_ride_stats", path)
    for c in ("signup_ts", "first_ride_ts", "last_ride_ts", "second_ride_ts"):
        users[c] = pd.to_datetime(users[c])
    users = (users.merge(db.run("user_rewards", path), on="user_id", how="left")
                  .fillna({"earned": 0, "redeemed": 0, "expired": 0}))

    events = db.run("reward_events_raw", path)
    events["event_ts"] = pd.to_datetime(events["event_ts"])
    first_redeem = db.run("first_redeem", path)

    return Data(
        as_of=as_of, last_idx=last_idx, expiry_days=expiry_days,
        tiers=db.run("tiers", path),
        users=users,
        monthly=db.run("monthly_activity", path),
        cohort=db.run("cohort_activity", path, **p),
        mom=db.run("mom_activity", path, **p),
        rewards_month=db.run("reward_by_month_tier", path),
        breakage=db.run("breakage_matured", path, as_of=as_of.strftime("%Y-%m-%d %H:%M:%S"),
                        expiry_days=expiry_days),
        first_redeem=first_redeem,
        lags=redemption_lags(events),
    )


# --------------------------------------------------------------------------- engagement
def engagement_kpis(d: Data) -> dict:
    riders = d.users[d.users.n_rides > 0]
    seasoned = riders[riders.first_ride_ts <= d.as_of - pd.Timedelta(days=30)]
    days_to_2nd = (seasoned.second_ride_ts - seasoned.first_ride_ts).dt.total_seconds() / 86400
    return {
        "users": len(d.users),
        "rides_per_user": riders.n_rides.mean(),
        "median_rides_per_user": riders.n_rides.median(),
        "rides_per_active_month": riders.n_rides.sum() / riders.active_months.sum(),
        "median_days_between_rides": riders.avg_gap_days.median(),
        "repeat_rate": (riders.n_rides >= 2).mean(),               # 2+ rides ever
        "repeat_rate_30d": (days_to_2nd <= 30).mean(),            # 2nd ride within 30 days (users >=30d old)
        "revenue_per_user": riders.gross_revenue.mean(),
        "median_revenue_per_user": riders.gross_revenue.median(),
        "revenue_per_active_month": riders.gross_revenue.sum() / riders.active_months.sum(),
    }


def retention_by_tier(d: Data) -> pd.DataFrame:
    """Pooled month-over-month retention: share of active users who ride again next month."""
    g = d.mom.groupby("tier")[["active_users", "retained_users"]].sum().reindex(d.tier_order)
    g.loc["All"] = g.sum()
    g["mom_retention"] = g.retained_users / g.active_users
    return g


def mom_series(d: Data) -> pd.DataFrame:
    """Month-over-month retention per calendar month, per tier and overall."""
    cols = ["active_users", "retained_users"]
    g = d.mom.groupby(["m_idx", "tier"])[cols].sum().reset_index()
    total = d.mom.groupby("m_idx")[cols].sum().reset_index().assign(tier="All")
    g = pd.concat([g, total], ignore_index=True)
    g["retention"] = g.retained_users / g.active_users
    g["month"] = g.m_idx.map(idx_label)
    return g


def cohort_matrix(d: Data, tier: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    c = d.cohort if tier in (None, "All") else d.cohort[d.cohort.tier == tier]
    n = c.groupby(["cohort_idx", "month_offset"]).active_users.sum().unstack()
    observable = np.add.outer(n.index.values, n.columns.values) <= d.last_idx
    n = n.where(~observable | n.notna(), 0)               # observed-but-empty cell = 0, not blank
    size = n[0]
    pct = n.div(size, axis=0)
    pct.index = [idx_label(i) for i in pct.index]
    return pct, size.set_axis(pct.index)


def m1_retention(d: Data) -> pd.DataFrame:
    """Share of each tier's users who ride in the month after their first, with a 95% CI vs the first tier."""
    c = d.cohort[d.cohort.cohort_idx <= d.last_idx - 1]
    n = c[c.month_offset == 0].groupby("tier").active_users.sum().reindex(d.tier_order)
    k = c[c.month_offset == 1].groupby("tier").active_users.sum().reindex(d.tier_order).fillna(0)
    out = pd.DataFrame({"users": n, "returned": k, "m1_retention": k / n})
    p0, n0 = out.m1_retention.iloc[0], out.users.iloc[0]
    se = np.sqrt(out.m1_retention * (1 - out.m1_retention) / out.users + p0 * (1 - p0) / n0)
    diff = (out.m1_retention - p0) * 100
    out["diff_pp"], out["ci_low_pp"], out["ci_high_pp"] = diff, diff - 196 * se, diff + 196 * se
    out.iloc[0, -3:] = np.nan
    return out


def empirical_ltv_curve(d: Data, tier: str | None = None) -> pd.DataFrame:
    """Cumulative gross revenue per acquired user by months since first ride (cohorts that have reached each age)."""
    c = d.cohort if tier in (None, "All") else d.cohort[d.cohort.tier == tier]
    size = c[c.month_offset == 0].groupby("cohort_idx").active_users.sum()
    rev = c.groupby(["cohort_idx", "month_offset"]).revenue.sum()
    rows = []
    for k in range(0, d.last_idx - int(size.index.min()) + 1):
        elig = size.index[size.index + k <= d.last_idx]
        if len(elig) == 0:
            break
        r = sum(rev.get((i, k), 0.0) for i in elig)
        rows.append((k, r / size[elig].sum(), int(size[elig].sum())))
    out = pd.DataFrame(rows, columns=["month_offset", "revenue_per_user", "users"])
    out["cum_revenue_per_user"] = out.revenue_per_user.cumsum()
    return out


# --------------------------------------------------------------------------- rewards
def breakage_by_tier(d: Data) -> pd.DataFrame:
    b = d.breakage.set_index("tier").reindex(d.tier_order)
    b.loc["All"] = b.sum()
    b["breakage_rate"] = b.matured_expired / b.matured_earned
    return b


def wquantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    o = np.argsort(values)
    v, w = np.asarray(values)[o], np.asarray(weights)[o]
    return float(v[np.searchsorted(np.cumsum(w), q * w.sum())])


def matured_lags(d: Data) -> pd.DataFrame:
    """Redemption slices from lots old enough to have fully played out (no right-censoring)."""
    return d.lags[d.lags.lot_ts <= d.as_of - pd.Timedelta(days=d.expiry_days)]


def reward_summary(d: Data) -> pd.DataFrame:
    """Earn / redeem / expire totals, Burn-to-Earn, Breakage and redemption speed per tier (+ All)."""
    t = d.users.groupby("tier")[["earned", "redeemed", "expired"]].sum().reindex(d.tier_order)
    t.loc["All"] = t.sum()
    t["outstanding"] = t.earned - t.redeemed - t.expired
    t["burn_to_earn"] = t.redeemed / t.earned
    t["breakage_rate"] = breakage_by_tier(d).breakage_rate
    ml = matured_lags(d)
    for name in t.index:
        s = ml if name == "All" else ml[ml.tier == name]
        t.loc[name, "median_days_to_redeem"] = wquantile(s.lag_days.values, s.points.values, 0.5) if len(s) else np.nan
        t.loc[name, "p90_days_to_redeem"] = wquantile(s.lag_days.values, s.points.values, 0.9) if len(s) else np.nan
    fr = d.first_redeem.dropna(subset=["days_to_first_redeem"])
    t["median_days_to_first_redeem"] = fr.groupby("tier").days_to_first_redeem.median()
    t.loc["All", "median_days_to_first_redeem"] = fr.days_to_first_redeem.median()
    earners = d.users[d.users.earned > 0]
    t["share_users_ever_redeemed"] = (earners.redeemed > 0).groupby(earners.tier).mean()
    t.loc["All", "share_users_ever_redeemed"] = (earners.redeemed > 0).mean()
    for c in ("earned", "redeemed", "expired", "outstanding"):
        t[f"{c}_value"] = t[c] * POINT_VALUE
    return t


def redemption_curve(d: Data, tier: str | None = None, step: int = 5) -> pd.DataFrame:
    """Share of matured earned points redeemed within N days. Plateaus at 1 - breakage."""
    ml = matured_lags(d)
    b = breakage_by_tier(d)
    name = tier or "All"
    s = ml if name == "All" else ml[ml.tier == name]
    days = np.arange(0, d.expiry_days + 1, step)
    cum = [s.points[s.lag_days <= x].sum() / b.loc[name, "matured_earned"] for x in days]
    return pd.DataFrame({"days_since_earned": days, "share_redeemed": cum})


# --------------------------------------------------------------------------- LTV and tier ROI
def ltv_summary(d: Data, margin: float, discount: float = 0.0) -> dict:
    """Blended simple LTV: contribution per active month x expected active months."""
    e = engagement_kpis(d)
    r = retention_by_tier(d).loc["All", "mom_retention"]
    rs = reward_summary(d).loc["All"]
    months = expected_months(r, discount)
    contribution_pm = e["revenue_per_active_month"] * margin
    users_months = d.users.active_months.sum()
    reward_pm = rs.earned_value / users_months * (1 - rs.breakage_rate)
    return {"retention": r, "expected_months": months,
            "contribution_per_month": contribution_pm, "reward_cost_per_month": reward_pm,
            "ltv_gross": contribution_pm * months, "reward_cost_ltv": reward_pm * months,
            "ltv_net": (contribution_pm - reward_pm) * months}


def tier_economics(d: Data, margin: float, discount: float = 0.0) -> pd.DataFrame:
    """Does each tier's retention lift pay for its reward cost? Compared with the first (reference) tier."""
    ret, brk, m1 = retention_by_tier(d), breakage_by_tier(d), m1_retention(d)
    rows = []
    for t in d.tiers.itertuples():
        u = d.users[d.users.tier == t.tier]
        months_active = u.active_months.sum()
        r = ret.loc[t.tier, "mom_retention"]
        arpu = u.gross_revenue.sum() / months_active
        reward_pm = u.earned.sum() * POINT_VALUE / months_active * (1 - brk.loc[t.tier, "breakage_rate"])
        life = expected_months(r, discount)
        rows.append({
            "tier": t.tier, "reward_rate": t.reward_rate, "users": len(u),
            "mom_retention": r, "m1_retention": m1.loc[t.tier, "m1_retention"],
            "m1_diff_pp": m1.loc[t.tier, "diff_pp"],
            "m1_ci_low_pp": m1.loc[t.tier, "ci_low_pp"], "m1_ci_high_pp": m1.loc[t.tier, "ci_high_pp"],
            "arpu_per_active_month": arpu, "breakage_rate": brk.loc[t.tier, "breakage_rate"],
            "expected_months": life,
            "contribution_per_month": arpu * margin, "reward_cost_per_month": reward_pm,
            "ltv_gross": arpu * margin * life, "reward_cost_ltv": reward_pm * life,
            "ltv_net": (arpu * margin - reward_pm) * life,
        })
    df = pd.DataFrame(rows).set_index("tier")
    ref = df.iloc[0]
    df["d_retention_pp"] = (df.mom_retention - ref.mom_retention) * 100
    df["d_ltv_gross"] = df.ltv_gross - ref.ltv_gross
    df["d_reward_cost"] = df.reward_cost_ltv - ref.reward_cost_ltv
    df["d_ltv_net"] = df.ltv_net - ref.ltv_net
    df["incremental_roi"] = df.d_ltv_net / df.d_reward_cost.where(df.d_reward_cost > 0)
    df["breakeven_retention"] = [
        breakeven_retention(r.contribution_per_month - r.reward_cost_per_month, ref.ltv_net, discount)
        for r in df.itertuples()]
    df["pays_for_itself"] = df.d_ltv_net > 0
    return df


def baseline_from_data(d: Data, margin: float, discount: float = 0.0) -> Baseline:
    """Seed the simulator with what the reference tier actually looks like."""
    e = tier_economics(d, margin, discount).iloc[0]
    return Baseline(base_rate=float(e.reward_rate), retention=float(e.mom_retention),
                    arpu=float(e.arpu_per_active_month), margin=margin,
                    breakage=float(e.breakage_rate), discount=discount)
