"""SQLite helpers: build the database and run named queries from sql/metrics.sql."""
from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import DB_PATH, SQL_DIR
from .datagen import generate


def build_db(path: Path = DB_PATH, n_users: int = 6000, seed: int = 42) -> Path:
    """(Re)create the database from synthetic data. Swap `generate` for your own loader."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    tables = generate(n_users=n_users, seed=seed)
    with closing(sqlite3.connect(path)) as con:
        con.executescript((SQL_DIR / "schema.sql").read_text())
        for name, df in tables.items():
            df.to_sql(name, con, if_exists="append", index=False)
        con.commit()
    return path


@lru_cache(maxsize=None)
def _queries() -> dict[str, str]:
    text = (SQL_DIR / "metrics.sql").read_text()
    out = {}
    for block in re.split(r"^-- name:\s*", text, flags=re.M)[1:]:
        name, _, sql = block.partition("\n")
        out[name.strip()] = sql.strip().rstrip(";")
    return out


def run(name: str, path: Path = DB_PATH, **params) -> pd.DataFrame:
    """Run a named query from sql/metrics.sql with :named parameters."""
    with closing(sqlite3.connect(path)) as con:
        return pd.read_sql_query(_queries()[name], con, params=params)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the synthetic loyalty database.")
    ap.add_argument("--users", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print("Built", build_db(n_users=args.users, seed=args.seed))
