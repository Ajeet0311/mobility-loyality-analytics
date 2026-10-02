"""Synthetic rides + points ledger.

Ground truth baked in (so the analytics have something real to recover):
  * Monthly churn falls with the tier's reward rate, with diminishing returns.
  * Points follow a FIFO ledger: earn on every ride, redeem against later rides
    (at most half the fare), expire EXPIRY_DAYS after being earned.
  * Riders differ in ride frequency, fare level and willingness to redeem.
"""
from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import EXPIRY_DAYS, MIN_REDEEM_POINTS, POINT_VALUE, TIERS

BASE_CHURN = 0.25            # monthly churn at the Base reward rate
MAX_CHURN_REDUCTION = 0.10   # most churn the rewards can remove
SATURATION = 0.05            # reward-rate scale of the diminishing returns
ONE_AND_DONE = 0.20          # share of riders who never ride a second time
TIER_MIX = [0.40, 0.35, 0.25]

_FMT = "%Y-%m-%d %H:%M:%S"


def _next_month(d: datetime) -> datetime:
    return datetime(d.year + d.month // 12, d.month % 12 + 1, 1)


def generate(n_users: int = 6000, seed: int = 42,
             start: str = "2025-10-01", end: str = "2026-09-30") -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    t0 = datetime.fromisoformat(start)
    t1 = datetime.fromisoformat(end) + timedelta(hours=23, minutes=59, seconds=59)
    expiry = timedelta(days=EXPIRY_DAYS)
    names, rates = list(TIERS), list(TIERS.values())
    base_rate = rates[0]

    tier_idx = rng.choice(len(names), n_users, p=TIER_MIX)
    signup_s = rng.uniform(0, (t1 - t0).total_seconds() - 86400, n_users)

    users, rides, events = [], [], []
    ride_id = event_id = 0

    for uid in range(1, n_users + 1):
        rate = rates[tier_idx[uid - 1]]
        signup = t0 + timedelta(seconds=float(signup_s[uid - 1]))
        users.append((uid, signup.strftime(_FMT), names[tier_idx[uid - 1]]))

        # --- rider traits -------------------------------------------------
        churn = BASE_CHURN - MAX_CHURN_REDUCTION * (1 - math.exp(-(rate - base_rate) / SATURATION))
        churn = float(np.clip(churn * rng.lognormal(0, 0.25), 0.02, 0.95))
        lam = rng.gamma(2.0, 1.6)                 # rides per live month (mean ~3.2)
        fare_scale = rng.lognormal(0, 0.25)
        redeem_p = rng.beta(2, 2)                 # chance of redeeming when eligible

        # --- ride times ---------------------------------------------------
        first = min(signup + timedelta(hours=float(rng.uniform(0, 48))), t1)
        times = [first]
        if rng.random() >= ONE_AND_DONE:
            alive = True
            m0 = datetime(first.year, first.month, 1)   # first ride can spill into the next month
            k = 0
            while m0 <= t1:
                lo, hi = max(m0, first), min(_next_month(m0), t1)
                if k > 0:
                    alive = alive and rng.random() > churn
                    if not alive:
                        break
                days = (hi - lo).total_seconds() / 86400
                n = rng.poisson(lam * days / 30)
                times += [lo + timedelta(seconds=float(s))
                          for s in rng.uniform(0, (hi - lo).total_seconds(), n)]
                m0, k = _next_month(m0), k + 1
        times.sort()

        # --- rides + FIFO points ledger ----------------------------------
        lots: deque = deque()                    # [earn_event_id, earn_ts, points_left]
        balance = 0
        for ts in times:
            while lots and lots[0][1] + expiry <= ts:        # expire aged-out lots first
                lot_id, lot_ts, left = lots.popleft()
                event_id += 1
                events.append((event_id, uid, (lot_ts + expiry).strftime(_FMT), "expire", left, None, lot_id))
                balance -= left

            ride_id += 1
            fare = round(max(float(rng.lognormal(2.1, 0.5)) * fare_scale, 2.0), 2)
            rides.append((ride_id, uid, ts.strftime(_FMT), fare))

            pts = int(fare * rate / POINT_VALUE)
            if pts > 0:
                event_id += 1
                events.append((event_id, uid, ts.strftime(_FMT), "earn", pts, ride_id, None))
                lots.append([event_id, ts, pts])
                balance += pts

            if balance >= MIN_REDEEM_POINTS and rng.random() < redeem_p:
                need = min(balance, int(0.5 * fare / POINT_VALUE))
                event_id += 1
                events.append((event_id, uid, ts.strftime(_FMT), "redeem", need, ride_id, None))
                balance -= need
                while need > 0:
                    take = min(need, lots[0][2])
                    lots[0][2] -= take
                    need -= take
                    if lots[0][2] == 0:
                        lots.popleft()

        for lot_id, lot_ts, left in lots:                    # lapse before the data ends
            if lot_ts + expiry <= t1:
                event_id += 1
                events.append((event_id, uid, (lot_ts + expiry).strftime(_FMT), "expire", left, None, lot_id))

    ev = pd.DataFrame(events, columns=["event_id", "user_id", "event_ts", "event_type",
                                       "points", "ride_id", "lot_event_id"])
    ev[["ride_id", "lot_event_id"]] = ev[["ride_id", "lot_event_id"]].astype("Int64")
    return {
        "meta": pd.DataFrame({"key": ["as_of"], "value": [t1.strftime(_FMT)]}),
        "tiers": pd.DataFrame({"tier": names, "reward_rate": rates}),
        "users": pd.DataFrame(users, columns=["user_id", "signup_ts", "tier"]),
        "rides": pd.DataFrame(rides, columns=["ride_id", "user_id", "ride_ts", "fare"]),
        "reward_events": ev,
    }
