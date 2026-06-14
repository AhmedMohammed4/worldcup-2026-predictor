"""
Brick 1: Match data pipeline.

Pulls World Cup match data (2018, 2022, 2026) from openfootball/worldcup.json
and international results from a public CSV into SQLite.

All operations are idempotent (upsert). Safe to re-run daily.
"""

import io
import json
import sqlite3
from pathlib import Path

import requests
import pandas as pd

from teams import normalize_team

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "worldcup.db"

# openfootball raw URLs (master branch)
OPENFOOTBALL_BASE = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master"
)
WC_YEARS = [2018, 2022, 2026]

# International results CSV
INTL_RESULTS_URL = (
    "https://raw.githubusercontent.com/JamshedAli18/"
    "International-football-results-from-1872-to-2024/main/results.csv"
)


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def create_tables(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            tournament_year INTEGER NOT NULL,
            date TEXT,
            round TEXT,
            group_name TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_goals INTEGER,
            away_goals INTEGER,
            ground TEXT,
            PRIMARY KEY (tournament_year, date, home_team, away_team)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS international_matches (
            date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_goals INTEGER,
            away_goals INTEGER,
            tournament TEXT,
            neutral INTEGER,
            PRIMARY KEY (date, home_team, away_team)
        )
    """)
    conn.commit()


def ingest_worldcup(conn: sqlite3.Connection):
    """Pull World Cup fixtures/results from openfootball and upsert."""
    total = 0
    for year in WC_YEARS:
        url = f"{OPENFOOTBALL_BASE}/{year}/worldcup.json"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        matches = data.get("matches", [])
        for m in matches:
            home = normalize_team(m["team1"])
            away = normalize_team(m["team2"])
            score = m.get("score", {})
            ft = score.get("ft") if score else None
            home_goals = ft[0] if ft else None
            away_goals = ft[1] if ft else None

            conn.execute("""
                INSERT INTO matches
                    (tournament_year, date, round, group_name,
                     home_team, away_team, home_goals, away_goals, ground)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tournament_year, date, home_team, away_team)
                DO UPDATE SET
                    round = excluded.round,
                    group_name = excluded.group_name,
                    home_goals = excluded.home_goals,
                    away_goals = excluded.away_goals,
                    ground = excluded.ground
            """, (
                year,
                m.get("date"),
                m.get("round"),
                m.get("group"),
                home,
                away,
                home_goals,
                away_goals,
                m.get("ground"),
            ))
        total += len(matches)
        print(f"  World Cup {year}: {len(matches)} matches")

    conn.commit()
    return total


def ingest_international(conn: sqlite3.Connection):
    """Pull international results CSV and upsert."""
    print("  Downloading international results CSV...")
    resp = requests.get(INTL_RESULTS_URL, timeout=60)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df["home_team"] = df["home_team"].apply(normalize_team)
    df["away_team"] = df["away_team"].apply(normalize_team)
    df["neutral"] = df["neutral"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0})

    # Drop rows with missing team names or scores
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])

    rows = df[["date", "home_team", "away_team", "home_score", "away_score",
               "tournament", "neutral"]].values.tolist()

    conn.executemany("""
        INSERT INTO international_matches
            (date, home_team, away_team, home_goals, away_goals, tournament, neutral)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, home_team, away_team)
        DO UPDATE SET
            home_goals = excluded.home_goals,
            away_goals = excluded.away_goals,
            tournament = excluded.tournament,
            neutral = excluded.neutral
    """, rows)
    conn.commit()

    count = len(rows)
    print(f"  International matches: {count}")
    return count


def main():
    print("Ingesting match data into", DB_PATH)
    conn = get_db()
    create_tables(conn)

    print("\n--- World Cup fixtures ---")
    wc_count = ingest_worldcup(conn)

    print("\n--- International results ---")
    intl_count = ingest_international(conn)

    # Verify counts
    print("\n--- Verification ---")
    for table in ["matches", "international_matches"]:
        (count,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        print(f"  {table}: {count} rows in DB")

    wc_played = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE home_goals IS NOT NULL"
    ).fetchone()[0]
    wc_unplayed = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE home_goals IS NULL"
    ).fetchone()[0]
    print(f"  World Cup played: {wc_played}, unplayed: {wc_unplayed}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
