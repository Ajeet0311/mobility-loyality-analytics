# Mobility Loyalty & LTV Dashboard

Answers the questions a loyalty team actually asks about a ride-hailing / mobility rewards program, then lets you test a policy change before shipping it.

**Stack:** SQL (SQLite) for aggregation · pandas + NumPy for derived metrics and the simulator · Plotly for charts · Streamlit for the dashboard.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py          # builds data/loyalty.db on first run (few seconds)
pytest -q                     # 12 invariant tests (+1 headless render test)
```

Rebuild with different data: `python -m loyalty_analytics.db --users 10000 --seed 7`, or use the sidebar.

## Which question lives where

| Question | Metric | Computed in |
|---|---|---|
| How often do users ride? | rides/user, rides per active month, median days between rides | `user_ride_stats` (SQL, window functions) |
| % repeat users | riders with 2+ rides; 2nd ride within 30 days | SQL + `engagement_kpis` |
| Monthly retention | cohort matrix; rolling month-over-month; first-month return with 95% CI | `cohort_activity`, `mom_activity` (SQL) |
| Revenue per user | mean/median per user, per active month | SQL + pandas |
| Simple LTV | contribution per active month × `1/(1 − r)`, gross and net of rewards | `ltv_summary` |
| Reward currency earned / redeemed | points and currency value, monthly and by tier | `reward_by_month_tier` (SQL) |
| Burn-to-Earn | redeemed ÷ earned | `reward_summary` |
| Breakage | expired ÷ earned, **matured lots only** | `breakage_matured` (SQL) |
| How quickly redeemed | FIFO lot matching → median/P90 days, redemption curve; days to first redemption | pandas (`redemption_lags`) + `first_redeem` (SQL) |
| Does a tier justify its cost? | Δ retention vs Δ reward cost → Δ net LTV, break-even retention, margin sensitivity | `tier_economics` |
| Policy simulator | Tier A 5% vs Tier B 10% → retention, reward cost, net LTV | `simulator.py` |

## Design choices worth knowing

- **Reward cost is accrual-based:** `points earned × point value × (1 − breakage)`. Expired points cost nothing, which is why breakage matters for tier economics.
- **Breakage uses only matured points** (earned ≥ expiry window ago). Counting young, still-redeemable points makes breakage look artificially low. Redemption-speed stats use the same filter to avoid right-censoring.
- **Burn-to-Earn is a flow ratio** and understates the long-run rate while the program is growing. The dashboard shows `1 − breakage` alongside it.
- **Simple LTV** = `(fares × margin − reward cost) × 1/(1 − r/(1+discount))`. The Revenue & LTV tab plots it against the observed cumulative curve so you can see the gap.

## Policy simulator

`r(x) = r_base + max_lift · (1 − exp(−(x − x_base)/steepness))`. Diminishing returns on retention, linear growth in cost, so there is a reward rate beyond which net LTV falls.

- **Calibrated mode** fits `max_lift` and `steepness` to the observed tiers. With only two tiers above the reference, the fit is loose; the app warns when it hits the edge of its search range.
- **Manual mode** lets you set the response directly.
- Outputs per scenario: estimated retention, reward cost per active month and per lifetime, cohort-level cost, net LTV, and the **break-even retention** the tier needs to match the baseline.

## Illustrative results (synthetic data, seed 42, 25% margin, 1%/month discount)

These come from generated data with a built-in tier effect, so they show the method, not a real-world finding.

- 68% of riders repeat; ~3.2 rides per active month; pooled monthly retention 63%, first-month return ~51%.
- Revenue per rider averages $76.80 but the median is $34.6 (heavy-rider skew). Simple LTV: $20.2 gross, $17.0 after rewards.
- 2.39M points earned, 72% burned, 23% breakage on matured points, median 9.6 days to redeem.
- Tier B lifts first-month return by +5.5 pp (CI +2.2 to +8.9). Tier A's lift (+0.4 pp) is indistinguishable from zero.
- At 25% margin neither tier pays for itself (net LTV vs Base: A −$1.75, B −$3.09). Tier B breaks even near a 45% margin.

## Using your own data

Replace `generate()` in `loyalty_analytics/datagen.py` (or load into SQLite directly) so these tables exist, per `sql/schema.sql`:
`users(user_id, signup_ts, tier)`, `rides(ride_id, user_id, ride_ts, fare)`, `reward_events(event_id, user_id, event_ts, event_type earn|redeem|expire, points, ride_id, lot_event_id)`, `tiers(tier, reward_rate)`, and `meta(key='as_of', value)`. Expiry events should carry the `lot_event_id` of the earn they lapse. The SQL uses CTEs and window functions, so it ports to Postgres/BigQuery with date-function changes (`strftime`, `julianday`).

## Limitations

- **Observational tier comparison.** Tiers here are randomly assigned. In real programs riders usually earn tiers by riding more, which inflates apparent retention lift. Use a holdout or randomized rollout before acting.
- Geometric LTV assumes constant retention and runs conservative against observed cohorts.
- The simulator holds ride frequency and fares fixed; it models retention and cost only.

## Layout

```
app.py                      Streamlit dashboard (6 tabs)
loyalty_analytics/
  config.py                 point value, expiry, tiers
  datagen.py                synthetic rides + FIFO points ledger
  db.py                     build SQLite, run named queries
  metrics.py                SQL results → KPIs, tier economics
  simulator.py              retention-response model, scenarios, calibration
sql/schema.sql, metrics.sql
tests/                      ledger invariants, SQL/pandas checks, simulator maths
```
