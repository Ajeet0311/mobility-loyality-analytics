-- Named queries, loaded by loyalty_analytics/db.py (split on "-- name:").
-- Dialect: SQLite (CTEs + window functions). Month index = year*12 + month-1.
-- Parameters: :as_of  :last_idx (last fully observed month)  :expiry_days

-- name: as_of
SELECT value AS as_of FROM meta WHERE key = 'as_of';

-- name: tiers
SELECT tier, reward_rate FROM tiers ORDER BY reward_rate;

-- Q: How often do users ride? What share repeat? Revenue per user?
-- name: user_ride_stats
WITH ordered AS (
    SELECT user_id, ride_ts, fare,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ride_ts, ride_id) AS rn,
           LAG(ride_ts) OVER (PARTITION BY user_id ORDER BY ride_ts, ride_id)  AS prev_ts
    FROM rides
),
agg AS (
    SELECT user_id,
           COUNT(*)                                           AS n_rides,
           SUM(fare)                                          AS gross_revenue,
           MIN(ride_ts)                                       AS first_ride_ts,
           MAX(ride_ts)                                       AS last_ride_ts,
           MIN(CASE WHEN rn = 2 THEN ride_ts END)             AS second_ride_ts,
           COUNT(DISTINCT strftime('%Y-%m', ride_ts))         AS active_months,
           AVG(julianday(ride_ts) - julianday(prev_ts))       AS avg_gap_days
    FROM ordered
    GROUP BY user_id
)
SELECT u.user_id, u.tier, u.signup_ts,
       COALESCE(a.n_rides, 0)        AS n_rides,
       COALESCE(a.gross_revenue, 0)  AS gross_revenue,
       COALESCE(a.active_months, 0)  AS active_months,
       a.first_ride_ts, a.last_ride_ts, a.second_ride_ts, a.avg_gap_days
FROM users u
LEFT JOIN agg a USING (user_id);

-- name: monthly_activity
WITH first_ride AS (
    SELECT user_id, strftime('%Y-%m', MIN(ride_ts)) AS first_month
    FROM rides GROUP BY user_id
)
SELECT strftime('%Y-%m', r.ride_ts)  AS month,
       COUNT(DISTINCT r.user_id)     AS active_users,
       COUNT(*)                      AS rides,
       SUM(r.fare)                   AS gross_revenue,
       COUNT(DISTINCT CASE WHEN f.first_month = strftime('%Y-%m', r.ride_ts)
                           THEN r.user_id END) AS new_users
FROM rides r
JOIN first_ride f USING (user_id)
GROUP BY 1
ORDER BY 1;

-- Q: Monthly retention. Cohort = month of first ride; "active" = at least one ride that month.
-- name: cohort_activity
WITH ride_m AS (
    SELECT r.user_id, u.tier, r.fare,
           CAST(strftime('%Y', r.ride_ts) AS INTEGER) * 12
         + CAST(strftime('%m', r.ride_ts) AS INTEGER) - 1 AS m_idx
    FROM rides r
    JOIN users u USING (user_id)
),
first_m AS (
    SELECT user_id, MIN(m_idx) AS cohort_idx FROM ride_m GROUP BY user_id
)
SELECT f.cohort_idx,
       m.tier,
       m.m_idx - f.cohort_idx      AS month_offset,
       COUNT(DISTINCT m.user_id)   AS active_users,
       COUNT(*)                    AS rides,
       SUM(m.fare)                 AS revenue
FROM ride_m m
JOIN first_m f USING (user_id)
WHERE m.m_idx <= :last_idx
GROUP BY f.cohort_idx, m.tier, m.m_idx - f.cohort_idx;

-- Rolling month-over-month retention: of users active in month M, how many ride in M+1?
-- name: mom_activity
WITH active AS (
    SELECT DISTINCT r.user_id, u.tier,
           CAST(strftime('%Y', r.ride_ts) AS INTEGER) * 12
         + CAST(strftime('%m', r.ride_ts) AS INTEGER) - 1 AS m_idx
    FROM rides r
    JOIN users u USING (user_id)
)
SELECT a.tier, a.m_idx,
       COUNT(*)           AS active_users,
       COUNT(n.user_id)   AS retained_users
FROM active a
LEFT JOIN active n ON n.user_id = a.user_id AND n.m_idx = a.m_idx + 1
WHERE a.m_idx < :last_idx
GROUP BY a.tier, a.m_idx;

-- Q: How much is earned / redeemed / expired?
-- name: reward_by_month_tier
SELECT strftime('%Y-%m', e.event_ts) AS month,
       u.tier,
       SUM(CASE WHEN e.event_type = 'earn'   THEN e.points ELSE 0 END) AS earned,
       SUM(CASE WHEN e.event_type = 'redeem' THEN e.points ELSE 0 END) AS redeemed,
       SUM(CASE WHEN e.event_type = 'expire' THEN e.points ELSE 0 END) AS expired
FROM reward_events e
JOIN users u USING (user_id)
GROUP BY 1, 2
ORDER BY 1, 2;

-- name: user_rewards
SELECT user_id,
       SUM(CASE WHEN event_type = 'earn'   THEN points ELSE 0 END) AS earned,
       SUM(CASE WHEN event_type = 'redeem' THEN points ELSE 0 END) AS redeemed,
       SUM(CASE WHEN event_type = 'expire' THEN points ELSE 0 END) AS expired
FROM reward_events
GROUP BY user_id;

-- Q: Breakage. Only lots old enough to have hit their expiry date count in the denominator;
--    otherwise young, still-redeemable points make breakage look artificially low.
-- name: breakage_matured
WITH matured AS (
    SELECT event_id, user_id, points
    FROM reward_events
    WHERE event_type = 'earn'
      AND julianday(:as_of) - julianday(event_ts) >= :expiry_days
),
lapsed AS (
    SELECT lot_event_id, SUM(points) AS expired_points
    FROM reward_events
    WHERE event_type = 'expire'
    GROUP BY lot_event_id
)
SELECT u.tier,
       SUM(m.points)                        AS matured_earned,
       COALESCE(SUM(l.expired_points), 0)   AS matured_expired
FROM matured m
JOIN users u USING (user_id)
LEFT JOIN lapsed l ON l.lot_event_id = m.event_id
GROUP BY u.tier;

-- Q: How quickly are rewards redeemed? (first earn -> first redemption, per user)
-- name: first_redeem
SELECT u.user_id, u.tier,
       MIN(CASE WHEN e.event_type = 'earn'   THEN e.event_ts END) AS first_earn_ts,
       MIN(CASE WHEN e.event_type = 'redeem' THEN e.event_ts END) AS first_redeem_ts,
       julianday(MIN(CASE WHEN e.event_type = 'redeem' THEN e.event_ts END))
     - julianday(MIN(CASE WHEN e.event_type = 'earn'   THEN e.event_ts END)) AS days_to_first_redeem
FROM users u
JOIN reward_events e USING (user_id)
GROUP BY u.user_id, u.tier;

-- Raw ledger, ordered for the FIFO lot matching done in pandas (redemption lag).
-- name: reward_events_raw
SELECT e.event_id, e.user_id, u.tier, e.event_ts, e.event_type, e.points, e.lot_event_id
FROM reward_events e
JOIN users u USING (user_id)
ORDER BY e.user_id, e.event_ts, e.event_id;
