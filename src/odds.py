"""
Brick 2: Odds ingestion.

Pulls current 2026 World Cup match odds (1X2 and totals) from the-odds-api
into the SQLite `odds` table. Idempotent - safe to re-run.

Usage:
    python odds.py
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from teams import normalize_team

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "worldcup.db"
API_KEY = os.getenv("THE_ODDS_API_KEY", "")

BASE_URL = "https://api.the-odds-api.com/v4"

# the-odds-api sport keys to try for World Cup
# The exact key may vary - we discover it dynamically.
SPORT_KEY_CANDIDATES = [
    "soccer_fifa_world_cup",
    "soccer_fifa_world_cup_2026",
    "soccer_international_friendlies",
]

MARKETS = ["h2h", "totals"]
REGIONS = "us,uk,eu"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def create_odds_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds (
            event_id TEXT NOT NULL,
            sport_key TEXT,
            commence_time TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            bookmaker TEXT NOT NULL,
            market TEXT NOT NULL,
            outcome_name TEXT NOT NULL,
            outcome_price REAL NOT NULL,
            outcome_point REAL NOT NULL DEFAULT -1,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (event_id, bookmaker, market, outcome_name, outcome_point)
        )
    """)
    conn.commit()


def discover_sport_key() -> str:
    """Find the correct sport key for World Cup odds."""
    resp = requests.get(
        f"{BASE_URL}/sports",
        params={"apiKey": API_KEY},
        timeout=15,
    )
    resp.raise_for_status()
    sports = resp.json()

    # Look for World Cup related sport keys
    wc_sports = [
        s for s in sports
        if "world_cup" in s["key"] or "world cup" in s.get("title", "").lower()
    ]
    if wc_sports:
        # Prefer the one that has active events
        for s in wc_sports:
            if s.get("active", False):
                print(f"  Found active World Cup sport: {s['key']} ({s['title']})")
                return s["key"]
        # Fall back to first WC sport
        key = wc_sports[0]["key"]
        print(f"  Found World Cup sport (not active): {key} ({wc_sports[0]['title']})")
        return key

    # Try candidates
    for candidate in SPORT_KEY_CANDIDATES:
        for s in sports:
            if s["key"] == candidate:
                print(f"  Using candidate sport key: {candidate}")
                return candidate

    # Print all soccer sports for debugging
    soccer = [s for s in sports if "soccer" in s["key"]]
    print("  Available soccer sports:")
    for s in soccer:
        active = "ACTIVE" if s.get("active") else "inactive"
        print(f"    {s['key']} - {s['title']} [{active}]")

    raise ValueError(
        "Could not find a World Cup sport key. "
        "Check the available soccer sports listed above."
    )


def fetch_odds(sport_key: str) -> list[dict]:
    """Fetch odds for all upcoming events in the given sport."""
    all_events = []
    for market in MARKETS:
        resp = requests.get(
            f"{BASE_URL}/sports/{sport_key}/odds",
            params={
                "apiKey": API_KEY,
                "regions": REGIONS,
                "markets": market,
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
            timeout=15,
        )
        resp.raise_for_status()

        remaining = resp.headers.get("x-requests-remaining", "?")
        print(f"  Fetched {market} odds: {len(resp.json())} events "
              f"(API requests remaining: {remaining})")

        for event in resp.json():
            event["_market_type"] = market
            all_events.append(event)

    return all_events


def upsert_odds(conn: sqlite3.Connection, events: list[dict]):
    """Parse API response and upsert into odds table."""
    now = datetime.now(timezone.utc).isoformat()
    row_count = 0

    for event in events:
        event_id = event["id"]
        sport_key = event.get("sport_key", "")
        commence = event.get("commence_time", "")
        home = normalize_team(event.get("home_team", ""))
        away = normalize_team(event.get("away_team", ""))

        for bookmaker in event.get("bookmakers", []):
            bk_name = bookmaker["key"]
            for mkt in bookmaker.get("markets", []):
                market_key = mkt["key"]
                for outcome in mkt.get("outcomes", []):
                    name = outcome["name"]
                    price = outcome["price"]
                    point = outcome.get("point")
                    if point is None:
                        point = -1

                    conn.execute("""
                        INSERT INTO odds
                            (event_id, sport_key, commence_time,
                             home_team, away_team, bookmaker, market,
                             outcome_name, outcome_price, outcome_point,
                             fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(event_id, bookmaker, market, outcome_name, outcome_point)
                        DO UPDATE SET
                            outcome_price = excluded.outcome_price,
                            fetched_at = excluded.fetched_at
                    """, (
                        event_id, sport_key, commence,
                        home, away, bk_name, market_key,
                        name, price, point, now,
                    ))
                    row_count += 1

    conn.commit()
    return row_count


def main():
    if not API_KEY:
        print("ERROR: THE_ODDS_API_KEY not set in .env")
        return

    print("Ingesting odds into", DB_PATH)
    conn = get_db()
    create_odds_table(conn)

    print("\n--- Discovering sport key ---")
    sport_key = discover_sport_key()

    print(f"\n--- Fetching odds for {sport_key} ---")
    events = fetch_odds(sport_key)

    if not events:
        print("  No events with odds found.")
        conn.close()
        return

    print(f"\n--- Upserting odds ---")
    count = upsert_odds(conn, events)
    print(f"  Upserted {count} odds rows")

    # Summary
    print("\n--- Summary ---")
    (total,) = conn.execute("SELECT COUNT(*) FROM odds").fetchone()
    print(f"  Total odds rows in DB: {total}")

    (books,) = conn.execute(
        "SELECT COUNT(DISTINCT bookmaker) FROM odds"
    ).fetchone()
    print(f"  Bookmakers: {books}")

    (events_count,) = conn.execute(
        "SELECT COUNT(DISTINCT event_id) FROM odds"
    ).fetchone()
    print(f"  Events with odds: {events_count}")

    print("\n  Upcoming matches with odds:")
    rows = conn.execute("""
        SELECT DISTINCT commence_time, home_team, away_team
        FROM odds
        ORDER BY commence_time
        LIMIT 10
    """).fetchall()
    for r in rows:
        print(f"    {r[0][:10]}  {r[1]} vs {r[2]}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
