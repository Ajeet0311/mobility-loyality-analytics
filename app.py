"""Mobility Loyalty & LTV dashboard.   Run:  streamlit run app.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from loyalty_analytics import db, metrics as m, simulator as sim
from loyalty_analytics.config import CURRENCY, DB_PATH, EXPIRY_DAYS, POINT_VALUE

st.set_page_config(page_title="Mobility Loyalty & LTV", layout="wide")

COLORS = {"Base": "#8793a6", "Tier A": "#2a9d8f", "Tier B": "#e07a1f", "All": "#264653"}
FALLBACK = ["#8793a6", "#2a9d8f", "#e07a1f", "#7b5ea7", "#c44569"]


# ----------------------------------------------------------------------------- helpers
def usd(x, nd=2):
    return "–" if pd.isna(x) else f"{CURRENCY}{x:,.{nd}f}"


def pct(x, nd=1):
    return "–" if pd.isna(x) else f"{x * 100:.{nd}f}%"


def pp(x, nd=1):
    return "–" if pd.isna(x) else f"{x:+.{nd}f} pp"


def clip(v, lo, hi):
    return float(min(max(v, lo), hi))


def color(name, i=0):
    return COLORS.get(name, FALLBACK[i % len(FALLBACK)])


def show(fig, height=380):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2))
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:                         # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def table(df: pd.DataFrame, fmt: dict, rename: dict | None = None):
    out = pd.DataFrame({(rename or {}).get(c, c): df[c].map(f) for c, f in fmt.items()}, index=df.index)
    try:
        st.dataframe(out, width="stretch")
    except Exception:
        st.dataframe(out, use_container_width=True)


def ltv_stack(df: pd.DataFrame, title: str, net="ltv_net", cost="reward_cost_ltv"):
    """Gross contribution LTV = net LTV + reward cost, one stacked bar per row."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df[net], name="Net LTV (after rewards)", marker_color="#2a9d8f"))
    fig.add_trace(go.Bar(x=df.index, y=df[cost], name="Reward cost", marker_color="#e07a1f"))
    fig.update_layout(barmode="stack", title=title, yaxis_title=f"{CURRENCY} per acquired user")
    return fig


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Running SQL + pandas metrics…")
def get_data(stamp: float) -> m.Data:
    return m.load_data()


@st.cache_data(show_spinner=False)
def margin_sensitivity(stamp: float, discount: float) -> pd.DataFrame:
    """Net-LTV gain vs the reference tier at each contribution margin (rows = margin %)."""
    d_ = get_data(stamp)
    grid = np.arange(10, 66, 5)
    return pd.DataFrame({mg: m.tier_economics(d_, mg / 100, discount).d_ltv_net for mg in grid}).T


if not DB_PATH.exists():
    with st.spinner("First run: generating synthetic rides and points ledger…"):
        db.build_db()
d = get_data(DB_PATH.stat().st_mtime)

with st.sidebar:
    st.header("Assumptions")
    margin = st.slider("Contribution margin on fares (%)", 5, 60, 25,
                       help="Share of gross fare left after driver pay and variable costs, before rewards.") / 100
    discount = st.slider("Monthly discount rate (%)", 0.0, 3.0, 1.0, 0.1) / 100
    st.caption(f"1 point = {CURRENCY}{POINT_VALUE:.2f} · points expire after {EXPIRY_DAYS} days · "
               f"data through {d.as_of:%d %b %Y}")
    with st.expander("Regenerate synthetic data"):
        n_users = st.number_input("Users", 1000, 30000, 6000, 1000)
        seed = st.number_input("Seed", 0, 9999, 42)
        if st.button("Rebuild database"):
            with st.spinner("Rebuilding…"):
                db.build_db(n_users=int(n_users), seed=int(seed))
            st.cache_data.clear()
            st.rerun()

eng = m.engagement_kpis(d)
ret = m.retention_by_tier(d)
m1 = m.m1_retention(d)
rw = m.reward_summary(d)
ltv = m.ltv_summary(d, margin, discount)
econ = m.tier_economics(d, margin, discount)
tiers = d.tier_order
ref = tiers[0]

st.title("Mobility Loyalty & LTV")
st.caption("Rides, retention, lifetime value and the economics of the points program.")

tab_over, tab_ret, tab_ltv, tab_rew, tab_roi, tab_sim = st.tabs(
    ["Overview", "Retention", "Revenue & LTV", "Rewards", "Tier ROI", "Policy simulator"])

# ----------------------------------------------------------------------------- overview
with tab_over:
    st.subheader("Engagement")
    c = st.columns(5)
    c[0].metric("Users", f"{eng['users']:,}")
    c[1].metric("Rides per user", f"{eng['rides_per_user']:.1f}", help="Mean rides per rider to date.")
    c[2].metric("Rides per active month", f"{eng['rides_per_active_month']:.1f}",
                help="Rides divided by months in which the rider rode at least once.")
    c[3].metric("Repeat users", pct(eng["repeat_rate"]), help="Share of riders with 2 or more rides.")
    c[4].metric("Monthly retention", pct(ret.loc["All", "mom_retention"]),
                help="Of riders active in a month, the share who ride again the next month (pooled).")

    st.subheader("Revenue and value")
    c = st.columns(5)
    c[0].metric("Revenue per user", usd(eng["revenue_per_user"]), help="Mean gross fares per rider to date.")
    c[1].metric("Revenue per active month", usd(eng["revenue_per_active_month"]))
    c[2].metric("Expected active months", f"{ltv['expected_months']:.1f}", help="1 / (1 − monthly retention).")
    c[3].metric("Simple LTV (gross)", usd(ltv["ltv_gross"]), help="Contribution per active month × expected months.")
    c[4].metric("LTV after rewards", usd(ltv["ltv_net"]), delta=f"−{usd(ltv['reward_cost_ltv'])} rewards",
                delta_color="off")

    st.subheader("Rewards")
    a = rw.loc["All"]
    c = st.columns(5)
    c[0].metric("Points earned", f"{a.earned:,.0f}", help=f"Worth {usd(a.earned_value, 0)}.")
    c[1].metric("Points redeemed", f"{a.redeemed:,.0f}", help=f"Worth {usd(a.redeemed_value, 0)}.")
    c[2].metric("Burn-to-Earn", pct(a.burn_to_earn), help="Points redeemed ÷ points earned.")
    c[3].metric("Breakage", pct(a.breakage_rate),
                help="Expired ÷ earned, counting only points old enough to have reached expiry.")
    c[4].metric("Median days to redeem", f"{a.median_days_to_redeem:.0f}",
                help="Points-weighted, FIFO-matched, from fully matured lots.")

    mo = d.monthly.set_index("month")
    left, right = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mo.index, y=mo.active_users - mo.new_users, name="Returning", marker_color="#264653"))
    fig.add_trace(go.Bar(x=mo.index, y=mo.new_users, name="New", marker_color="#2a9d8f"))
    fig.update_layout(barmode="stack", title="Active riders per month")
    with left:
        show(fig)
    fig = go.Figure(go.Bar(x=mo.index, y=mo.gross_revenue, marker_color="#264653"))
    fig.update_layout(title=f"Gross fares per month ({CURRENCY})")
    with right:
        show(fig)

# ----------------------------------------------------------------------------- retention
with tab_ret:
    left, right = st.columns([3, 2])
    with left:
        pick = st.selectbox("Cohort view", ["All"] + tiers)
        pct_m, size = m.cohort_matrix(d, pick)
        fig = go.Figure(go.Heatmap(
            z=pct_m.values * 100, x=[f"M{k}" for k in pct_m.columns],
            y=[f"{i}  (n={int(size[i]):,})" for i in pct_m.index],
            colorscale="Teal", zmin=0, zmax=100, texttemplate="%{z:.0f}", colorbar=dict(title="%")))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="Share of cohort still riding, by months since first ride")
        show(fig, 460)
    with right:
        s = m.mom_series(d)
        fig = go.Figure()
        for i, t in enumerate(tiers + ["All"]):
            x = s[s.tier == t].sort_values("m_idx")
            fig.add_trace(go.Scatter(x=x.month, y=x.retention * 100, name=t, mode="lines+markers",
                                     line=dict(color=color(t, i), width=3 if t == "All" else 2)))
        fig.update_layout(title="Month-over-month retention (%)", yaxis_range=[0, 100])
        show(fig, 460)

    st.markdown("**Does a higher reward rate change first-month return?**")
    left, right = st.columns([2, 3])
    with left:
        diff = m1.iloc[1:]
        fig = go.Figure(go.Bar(
            x=diff.index, y=diff.diff_pp, marker_color=[color(t, i + 1) for i, t in enumerate(diff.index)],
            error_y=dict(type="data", symmetric=False, array=diff.ci_high_pp - diff.diff_pp,
                         arrayminus=diff.diff_pp - diff.ci_low_pp)))
        fig.add_hline(y=0, line_color="#999")
        fig.update_layout(title=f"First-month return vs {ref} (pp, 95% CI)")
        show(fig, 320)
    with right:
        table(m1, {"users": lambda v: f"{v:,.0f}", "m1_retention": pct, "diff_pp": pp,
                   "ci_low_pp": pp, "ci_high_pp": pp},
              {"users": "Users", "m1_retention": "Rode in month 2", "diff_pp": f"vs {ref}",
               "ci_low_pp": "CI low", "ci_high_pp": "CI high"})
        st.caption("If the interval spans zero, the data can't distinguish that tier from the reference.")

# ----------------------------------------------------------------------------- revenue & LTV
with tab_ltv:
    c = st.columns(5)
    c[0].metric("Revenue per user", usd(eng["revenue_per_user"]))
    c[1].metric("Median revenue per user", usd(eng["median_revenue_per_user"]),
                help="Far below the mean: a minority of heavy riders drives most revenue.")
    c[2].metric("Median days between rides", f"{eng['median_days_between_rides']:.0f}")
    c[3].metric("2nd ride within 30 days", pct(eng["repeat_rate_30d"]), help="Among riders at least 30 days old.")
    c[4].metric("Contribution per active month", usd(ltv["contribution_per_month"]))

    left, right = st.columns(2)
    riders = d.users[d.users.n_rides > 0]
    fig = go.Figure(go.Histogram(x=riders.n_rides.clip(upper=30), xbins=dict(start=0.5, end=30.5, size=1),
                                 marker_color="#264653"))
    fig.update_layout(title="Rides per rider (30+ grouped)", xaxis_title="Rides", yaxis_title="Riders")
    with left:
        show(fig)

    curve = m.empirical_ltv_curve(d)
    fig = go.Figure(go.Scatter(x=curve.month_offset, y=curve.cum_revenue_per_user * margin, mode="lines+markers",
                               name="Observed", line=dict(color="#264653", width=3)))
    fig.add_hline(y=ltv["ltv_gross"], line_dash="dash", line_color="#e07a1f",
                  annotation_text=f"Simple model: {usd(ltv['ltv_gross'])}")
    fig.update_layout(title="Observed cumulative contribution per acquired user",
                      xaxis_title="Months since first ride", yaxis_title=CURRENCY)
    with right:
        show(fig)
    obs = curve.cum_revenue_per_user.iloc[-1] * margin
    st.caption(f"Observed {usd(obs)} after {int(curve.month_offset.iloc[-1])} months (still growing) vs the "
               f"simple model's lifetime {usd(ltv['ltv_gross'])}. The model runs conservative: pooled retention blends "
               "a weak first month with stickier survivors. Use it to compare tiers and policies, not as a forecast.")

    st.markdown("**LTV by tier**")
    table(econ, {"mom_retention": pct, "expected_months": lambda v: f"{v:.2f}", "ltv_gross": usd,
                 "reward_cost_ltv": usd, "ltv_net": usd},
          {"mom_retention": "Monthly retention", "expected_months": "Expected months",
           "ltv_gross": "LTV (gross)", "reward_cost_ltv": "Reward cost", "ltv_net": "LTV (net)"})

# ----------------------------------------------------------------------------- rewards
with tab_rew:
    a = rw.loc["All"]
    c = st.columns(5)
    c[0].metric("Earned", f"{a.earned:,.0f} pts", help=f"{usd(a.earned_value, 0)}")
    c[1].metric("Redeemed", f"{a.redeemed:,.0f} pts", help=f"{usd(a.redeemed_value, 0)}")
    c[2].metric("Expired", f"{a.expired:,.0f} pts", help=f"{usd(a.expired_value, 0)}")
    c[3].metric("Outstanding liability", usd(a.outstanding_value, 0), help="Earned − redeemed − expired, at point value.")
    c[4].metric("Users who ever redeemed", pct(a.share_users_ever_redeemed))

    rm = d.rewards_month.groupby("month")[["earned", "redeemed", "expired"]].sum()
    left, right = st.columns(2)
    fig = go.Figure()
    for col, clr in (("earned", "#264653"), ("redeemed", "#2a9d8f"), ("expired", "#e07a1f")):
        fig.add_trace(go.Bar(x=rm.index, y=rm[col], name=col.title(), marker_color=clr))
    fig.update_layout(barmode="group", title="Points per month")
    with left:
        show(fig)
    fig = go.Figure()
    for i, t in enumerate(tiers):
        x = m.redemption_curve(d, t)
        fig.add_trace(go.Scatter(x=x.days_since_earned, y=x.share_redeemed * 100, name=t,
                                 line=dict(color=color(t, i), width=3)))
    fig.update_layout(title="Share of earned points redeemed within N days", yaxis_title="%",
                      xaxis_title="Days since earned", yaxis_range=[0, 100])
    with right:
        show(fig)
    st.caption("Curves flatten at 1 − breakage. Built from points old enough to have reached expiry, so they are not cut off by the data window.")

    st.markdown("**By tier**")
    table(rw, {"burn_to_earn": pct, "breakage_rate": pct, "median_days_to_redeem": lambda v: f"{v:.0f}",
               "p90_days_to_redeem": lambda v: f"{v:.0f}", "median_days_to_first_redeem": lambda v: f"{v:.0f}",
               "share_users_ever_redeemed": pct, "outstanding_value": lambda v: usd(v, 0)},
          {"burn_to_earn": "Burn-to-Earn", "breakage_rate": "Breakage",
           "median_days_to_redeem": "Median days to redeem", "p90_days_to_redeem": "P90 days",
           "median_days_to_first_redeem": "Days to first redemption",
           "share_users_ever_redeemed": "Ever redeemed", "outstanding_value": "Liability"})
    with st.expander("Definitions"):
        st.markdown(
            "- **Burn-to-Earn** = points redeemed ÷ points earned over the same period. "
            "Understates the long-run rate while the program is still growing, because redemption lags earning.\n"
            f"- **Breakage** = points expired ÷ points earned, using only points earned at least {EXPIRY_DAYS} days "
            "ago (old enough to have either been redeemed or expired).\n"
            "- **Days to redeem** = FIFO match of each redemption to the oldest unexpired points, weighted by points.")

# ----------------------------------------------------------------------------- tier ROI
with tab_roi:
    st.markdown(f"**Does each tier's retention lift pay for its reward cost?** Compared with {ref}, at a "
                f"{margin:.0%} contribution margin.")
    for i, t in enumerate(tiers[1:], start=1):
        e = econ.loc[t]
        msg = (f"**{t} ({pct(e.reward_rate, 0)})** — retention {pp(e.d_retention_pp)}, lifetime reward cost "
               f"{usd(e.d_reward_cost)} higher, net LTV {usd(e.d_ltv_net)} vs {ref}. Needs {pct(e.breakeven_retention)} "
               f"monthly retention to break even; observed {pct(e.mom_retention)}.")
        (st.success if e.pays_for_itself else st.warning)(msg)

    left, right = st.columns(2)
    with left:
        show(ltv_stack(econ, "Lifetime value per active rider, by tier"))
    with right:
        sens = margin_sensitivity(DB_PATH.stat().st_mtime, discount)
        fig = go.Figure()
        for i, t in enumerate(tiers[1:], start=1):
            fig.add_trace(go.Scatter(x=sens.index, y=sens[t], name=t, line=dict(color=color(t, i), width=3)))
        fig.add_hline(y=0, line_color="#999")
        fig.update_layout(title=f"Net LTV gain vs {ref} by contribution margin",
                          xaxis_title="Contribution margin (%)", yaxis_title=CURRENCY)
        show(fig)
    table(econ, {"reward_rate": lambda v: pct(v, 0), "users": lambda v: f"{v:,.0f}", "mom_retention": pct,
                 "arpu_per_active_month": usd, "breakage_rate": pct, "reward_cost_per_month": usd,
                 "ltv_gross": usd, "reward_cost_ltv": usd, "ltv_net": usd, "d_ltv_net": usd,
                 "incremental_roi": lambda v: "–" if pd.isna(v) else f"{v:.0%}"},
          {"reward_rate": "Reward rate", "users": "Users", "mom_retention": "Monthly retention",
           "arpu_per_active_month": "Fares / active month", "breakage_rate": "Breakage",
           "reward_cost_per_month": "Reward cost / month", "ltv_gross": "LTV (gross)",
           "reward_cost_ltv": "Reward cost", "ltv_net": "LTV (net)", "d_ltv_net": f"Net LTV vs {ref}",
           "incremental_roi": "Incremental ROI"})
    with st.expander("How much to trust this"):
        st.markdown(
            "Tier comparisons are **observational**. Here every rider was assigned a tier at random, so the gaps are "
            "a fair read. In production, riders usually *earn* tiers by riding more, which makes higher tiers look "
            "stickier regardless of the rewards. Before acting, use a holdout group or randomized rollout, or at least "
            "compare riders matched on first-month behaviour.\n\n"
            "Reward cost is accrual-based: points earned × point value × (1 − breakage).")

# ----------------------------------------------------------------------------- simulator
with tab_sim:
    st.markdown("**If Tier A gives 5% and Tier B gives 10%, what happens to retention and reward cost?**")
    base0 = m.baseline_from_data(d, margin, discount)
    obs_tiers = econ[econ.reward_rate > base0.base_rate]

    c = st.columns(4)
    rate_a = c[0].slider("Tier A reward rate (%)", 0.0, 20.0, 5.0, 0.5) / 100
    rate_b = c[1].slider("Tier B reward rate (%)", 0.0, 20.0, 10.0, 0.5) / 100
    cohort_n = c[2].number_input("Riders in cohort", 1000, 1_000_000, 10_000, 1000)
    mode = c[3].radio("Retention response", ["Calibrated to data", "Manual"], horizontal=True)

    with st.expander("Baseline and response assumptions"):
        b1, b2, b3, b4 = st.columns(4)
        r0 = b1.number_input(f"{ref} monthly retention (%)", 30.0, 95.0, clip(round(base0.retention * 100, 1), 30.0, 95.0), 0.5) / 100
        arpu = b2.number_input("Fares per active month", 5.0, 200.0, clip(round(base0.arpu, 2), 5.0, 200.0), 1.0)
        brk = b3.number_input("Breakage (%)", 0.0, 90.0, clip(round(base0.breakage * 100, 1), 0.0, 90.0), 1.0) / 100
        x0 = b4.number_input(f"{ref} reward rate (%)", 0.0, 20.0, clip(round(base0.base_rate * 100, 1), 0.0, 20.0), 0.5) / 100
        base = sim.Baseline(base_rate=x0, retention=r0, arpu=arpu, margin=margin, breakage=brk, discount=discount)
        if mode == "Manual":
            k1, k2 = st.columns(2)
            resp = sim.Response(k1.slider("Max retention lift (pp)", 1.0, 30.0, 8.0, 0.5) / 100,
                                k2.slider("Reward rate at which 63% of the lift is reached (%)", 1.0, 40.0, 5.0, 0.5) / 100)
        else:
            resp = sim.fit_response(x0, r0, obs_tiers.reward_rate, obs_tiers.mom_retention)
            st.caption(f"Fitted to the observed tiers: max lift {resp.max_lift * 100:.1f} pp, "
                       f"63% reached at {resp.steepness * 100:.1f}% reward rate.")
            if resp.max_lift >= 0.295 or resp.steepness >= 0.395:
                st.warning("The fit hit the edge of its search range: observed lifts don't show diminishing returns, "
                           "so extrapolating beyond 10% is unreliable. Try Manual.")

    scen = {f"Baseline ({x0:.0%})": x0, f"Tier A ({rate_a:.1%})": rate_a, f"Tier B ({rate_b:.1%})": rate_b}
    res = sim.simulate(scen, base, resp, int(cohort_n))
    names = list(res.index)
    ra, rb = res.iloc[1], res.iloc[2]

    cols = st.columns(2)
    for col, nm, r in ((cols[0], names[1], ra), (cols[1], names[2], rb)):
        with col:
            st.markdown(f"##### {nm}")
            k = st.columns(4)
            k[0].metric("Est. retention", pct(r.retention), pp(r.d_retention_pp))
            k[1].metric("Reward cost / active month", usd(r.reward_cost_per_month))
            k[2].metric("Lifetime reward cost / rider", usd(r.reward_cost_per_user),
                        f"{r.d_reward_cost_per_user:+.2f} vs base", delta_color="inverse")
            k[3].metric("Net LTV", usd(r.ltv_net), f"{r.d_ltv_net:+.2f} vs base")
            st.caption(f"Break-even retention {pct(r.breakeven_retention)} vs estimated {pct(r.retention)} → "
                       + ("pays for itself." if r.pays_for_itself else "does not pay for itself."))

    d_ret = (rb.retention - ra.retention) * 100
    d_cost = rb.reward_cost_per_user - ra.reward_cost_per_user
    d_net = rb.ltv_net - ra.ltv_net
    st.info(f"**{names[2]} vs {names[1]}:** retention {d_ret:+.1f} pp, lifetime reward cost {d_cost:+.2f} {CURRENCY} per rider "
            f"({usd(rb.cohort_reward_cost - ra.cohort_reward_cost, 0)} across {int(cohort_n):,} riders), "
            f"net LTV {d_net:+.2f} {CURRENCY} per rider.")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        months = np.arange(0, 13)
        for i, (nm, r) in enumerate(res.iterrows()):
            fig.add_trace(go.Scatter(x=months, y=100 * r.retention ** months, name=nm,
                                     line=dict(color=FALLBACK[i], width=3)))
        fig.update_layout(title="Riders still active (% of cohort)", xaxis_title="Months since first ride")
        show(fig)
    with right:
        show(ltv_stack(res, "Lifetime value per rider", cost="reward_cost_per_user"))

    rates = np.linspace(0, 0.20, 41)
    sw = sim.sweep(rates, base, resp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rates * 100, y=sw.ltv_net, name="Net LTV", line=dict(color="#2a9d8f", width=3)))
    fig.add_trace(go.Scatter(x=rates * 100, y=sw.retention * 100, name="Retention (right axis)", yaxis="y2",
                             line=dict(color="#264653", dash="dot")))
    fig.add_trace(go.Scatter(x=res.reward_rate * 100, y=res.ltv_net, mode="markers+text", text=names,
                             textposition="top center", marker=dict(size=11, color="#e07a1f"), name="Scenarios"))
    fig.update_layout(title="Net LTV and retention across reward rates", xaxis_title="Reward rate (%)",
                      yaxis_title=f"Net LTV ({CURRENCY})",
                      yaxis2=dict(overlaying="y", side="right", title="Retention (%)", showgrid=False))
    show(fig, 400)

    show_cols = {"retention": pct, "expected_months": lambda v: f"{v:.2f}", "retained_after_12m": lambda v: f"{v:,.0f}",
                 "reward_cost_per_month": usd, "reward_cost_per_user": usd, "cohort_reward_cost": lambda v: usd(v, 0),
                 "ltv_net": usd, "breakeven_retention": pct}
    table(res, show_cols, {"retention": "Retention", "expected_months": "Expected months",
                           "retained_after_12m": "Active after 12m", "reward_cost_per_month": "Reward cost / month",
                           "reward_cost_per_user": "Lifetime reward cost", "cohort_reward_cost": "Cohort reward cost",
                           "ltv_net": "Net LTV", "breakeven_retention": "Break-even retention"})
    with st.expander("Model"):
        st.markdown(
            "- Retention responds to the reward rate with diminishing returns: "
            "`r(x) = r_base + max_lift · (1 − exp(−(x − x_base) / steepness))`.\n"
            "- Expected life = `1 / (1 − r/(1+discount))` active months.\n"
            "- Reward cost per active month = `fares × rate × (1 − breakage)`; expired points cost nothing.\n"
            "- Net LTV = `(fares × margin − reward cost) × expected life`.\n"
            "- Break-even retention is the retention a scenario needs to match the baseline's net LTV.")
