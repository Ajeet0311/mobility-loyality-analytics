-- One dimension (users), one ride fact, one reward-ledger fact.
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE tiers (
    tier         TEXT PRIMARY KEY,
    reward_rate  REAL NOT NULL                 -- share of fare returned as points
);

CREATE TABLE users (
    user_id    INTEGER PRIMARY KEY,
    signup_ts  TEXT NOT NULL,                  -- 'YYYY-MM-DD HH:MM:SS'
    tier       TEXT NOT NULL REFERENCES tiers(tier)
);

CREATE TABLE rides (
    ride_id  INTEGER PRIMARY KEY,
    user_id  INTEGER NOT NULL REFERENCES users(user_id),
    ride_ts  TEXT NOT NULL,
    fare     REAL NOT NULL                     -- gross fare, before reward redemption
);

-- Append-only points ledger. points is always positive; event_type gives the sign.
CREATE TABLE reward_events (
    event_id      INTEGER PRIMARY KEY,
    user_id       INTEGER NOT NULL REFERENCES users(user_id),
    event_ts      TEXT NOT NULL,
    event_type    TEXT NOT NULL CHECK (event_type IN ('earn', 'redeem', 'expire')),
    points        INTEGER NOT NULL CHECK (points > 0),
    ride_id       INTEGER REFERENCES rides(ride_id),          -- earn / redeem
    lot_event_id  INTEGER REFERENCES reward_events(event_id)  -- expire: the earn lot that lapsed
);

CREATE INDEX idx_rides_user_ts  ON rides(user_id, ride_ts);
CREATE INDEX idx_events_user_ts ON reward_events(user_id, event_ts);
CREATE INDEX idx_events_type    ON reward_events(event_type);
