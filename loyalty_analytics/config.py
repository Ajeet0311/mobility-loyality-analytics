"""Program constants shared by the data generator, SQL parameters and analytics."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "loyalty.db"
SQL_DIR = ROOT / "sql"

CURRENCY = "$"
POINT_VALUE = 0.01        # 1 reward point = $0.01 of ride credit
EXPIRY_DAYS = 180         # unredeemed points expire this long after they were earned
MIN_REDEEM_POINTS = 200   # minimum balance before a rider can redeem

# Tier -> reward rate (share of fare returned as points). "Base" is the reference tier.
TIERS = {"Base": 0.02, "Tier A": 0.05, "Tier B": 0.10}
