"""
Brick 5: Edge finder.

Joins model probabilities with market odds, converts decimal odds to
implied probabilities (removing vig), computes expected value, and
flags bets where model EV exceeds a configurable threshold.

Usage:
    python edges.py
"""

import sqlite3
from pathlib import Path

from model import predict_match, load_ratings, load_global_avg

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "worldcup.db"

# Minimum EV percentage to flag as an edge
MIN_EV_PCT = 3.0


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_upcoming_matches_with_odds(conn: sqlite3.Connection) -> list[dict]:
    """
    Get distinct upcoming matches that have h2h odds.
    Returns list of dicts with home_team, away_team, commence_time.
    """
    rows = conn.execute("""
        SELECT DISTINCT home_team, away_team, commence_time
        FROM odds
        WHERE market = 'h2h'
        ORDER BY commence_time
    """).fetchall()
    return [
        {"home_team": r[0], "away_team": r[1], "commence_time": r[2]}
        for r in rows
    ]


def get_best_odds(conn: sqlite3.Connection, home: str, away: str) -> dict:
    """
    Get the best available odds for each outcome across all bookmakers.

    Returns dict with keys:
        h2h: {home: best_price, draw: best_price, away: best_price}
        totals_2_5: {over: best_price, under: best_price} (for 2.5 line)
    Also returns the average odds for vig removal.
    """
    result = {"h2h": {}, "h2h_avg": {}, "totals_2_5": {}}

    # H2H best and average odds
    h2h_rows = conn.execute("""
        SELECT outcome_name, MAX(outcome_price) as best, AVG(outcome_price) as avg_price
        FROM odds
        WHERE home_team = ? AND away_team = ?
          AND market = 'h2h'
        GROUP BY outcome_name
    """, (home, away)).fetchall()

    for name, best, avg in h2h_rows:
        if name == home:
            result["h2h"]["home"] = best
            result["h2h_avg"]["home"] = avg
        elif name == away:
            result["h2h"]["away"] = best
            result["h2h_avg"]["away"] = avg
        elif name == "Draw":
            result["h2h"]["draw"] = best
            result["h2h_avg"]["draw"] = avg

    # Totals (over/under 2.5)
    totals_rows = conn.execute("""
        SELECT outcome_name, MAX(outcome_price) as best
        FROM odds
        WHERE home_team = ? AND away_team = ?
          AND market = 'totals'
          AND outcome_point = 2.5
        GROUP BY outcome_name
    """, (home, away)).fetchall()

    for name, best in totals_rows:
        if name == "Over":
            result["totals_2_5"]["over"] = best
        elif name == "Under":
            result["totals_2_5"]["under"] = best

    return result


def implied_prob_no_vig(avg_odds: dict) -> dict:
    """
    Convert average decimal odds to implied probabilities with vig removed.
    Uses the multiplicative method: divide each raw implied prob by the total overround.
    """
    if not avg_odds:
        return {}

    raw = {}
    for key, price in avg_odds.items():
        if price and price > 0:
            raw[key] = 1.0 / price

    total = sum(raw.values())
    if total == 0:
        return {}

    return {key: p / total for key, p in raw.items()}


def compute_ev(model_prob: float, decimal_odds: float) -> float:
    """Compute expected value percentage: (model_prob * odds - 1) * 100."""
    return (model_prob * decimal_odds - 1.0) * 100.0


def find_edges(
    conn: sqlite3.Connection,
    min_ev_pct: float = MIN_EV_PCT,
    ratings: dict = None,
    global_avg: float = None,
) -> list[dict]:
    """
    Find all edges across upcoming matches.
    Returns a list of edge dicts sorted by EV descending.
    """
    if ratings is None:
        ratings = load_ratings(conn)
    if global_avg is None:
        global_avg = load_global_avg(conn)

    matches = get_upcoming_matches_with_odds(conn)
    edges = []

    for match in matches:
        home = match["home_team"]
        away = match["away_team"]

        try:
            pred = predict_match(home, away, neutral=True, ratings=ratings,
                                 global_avg=global_avg, conn=conn)
        except ValueError:
            continue

        odds = get_best_odds(conn, home, away)
        market_implied = implied_prob_no_vig(odds.get("h2h_avg", {}))

        # Check h2h markets
        bets = [
            ("1X2", home, "home_win", "home"),
            ("1X2", "Draw", "draw", "draw"),
            ("1X2", away, "away_win", "away"),
        ]

        for market_name, label, pred_key, odds_key in bets:
            best_price = odds.get("h2h", {}).get(odds_key)
            if best_price is None:
                continue
            model_p = pred[pred_key]
            ev = compute_ev(model_p, best_price)
            impl_p = market_implied.get(odds_key, 0)

            if ev >= min_ev_pct:
                edges.append({
                    "match": f"{home} vs {away}",
                    "date": match["commence_time"][:10],
                    "market": market_name,
                    "selection": label,
                    "model_prob": model_p,
                    "implied_prob": impl_p,
                    "best_odds": best_price,
                    "ev_pct": ev,
                })

        # Check over 2.5
        over_price = odds.get("totals_2_5", {}).get("over")
        if over_price:
            ev_over = compute_ev(pred["over_2_5"], over_price)
            if ev_over >= min_ev_pct:
                edges.append({
                    "match": f"{home} vs {away}",
                    "date": match["commence_time"][:10],
                    "market": "O/U 2.5",
                    "selection": "Over 2.5",
                    "model_prob": pred["over_2_5"],
                    "implied_prob": 1.0 / over_price if over_price else 0,
                    "best_odds": over_price,
                    "ev_pct": ev_over,
                })

        # Check under 2.5
        under_price = odds.get("totals_2_5", {}).get("under")
        if under_price:
            under_prob = 1.0 - pred["over_2_5"]
            ev_under = compute_ev(under_prob, under_price)
            if ev_under >= min_ev_pct:
                edges.append({
                    "match": f"{home} vs {away}",
                    "date": match["commence_time"][:10],
                    "market": "O/U 2.5",
                    "selection": "Under 2.5",
                    "model_prob": under_prob,
                    "implied_prob": 1.0 / under_price if under_price else 0,
                    "best_odds": under_price,
                    "ev_pct": ev_under,
                })

    edges.sort(key=lambda e: e["ev_pct"], reverse=True)
    return edges


def main():
    conn = get_db()
    ratings = load_ratings(conn)
    global_avg = load_global_avg(conn)

    print(f"Finding edges (min EV: {MIN_EV_PCT}%)...\n")

    edges = find_edges(conn, min_ev_pct=MIN_EV_PCT, ratings=ratings, global_avg=global_avg)

    if not edges:
        print("No edges found above threshold.")
    else:
        print(f"{'Match':<35} {'Market':<10} {'Selection':<18} "
              f"{'Model':>6} {'Impl':>6} {'Odds':>6} {'EV%':>7}")
        print("-" * 95)
        for e in edges:
            print(f"  {e['match']:<33} {e['market']:<10} {e['selection']:<18} "
                  f"{e['model_prob']:>5.1%} {e['implied_prob']:>5.1%} "
                  f"{e['best_odds']:>6.2f} {e['ev_pct']:>+6.1f}%")

    print(f"\nTotal edges: {len(edges)}")
    conn.close()


if __name__ == "__main__":
    main()
