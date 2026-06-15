"""
Brick 5: Edge finder.

Joins model probabilities with market odds, converts decimal odds to
implied probabilities (removing vig), computes expected value, and
flags bets where model EV exceeds a configurable threshold.

Markets supported: 1X2, Over/Under (1.5, 2.5, 3.5), BTTS, Double Chance,
Asian Handicap, Correct Score.

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
    result = {"h2h": {}, "h2h_avg": {}, "totals": {}}

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

    # All totals lines (1.5, 2.5, 3.5, etc.)
    totals_rows = conn.execute("""
        SELECT outcome_name, outcome_point, MAX(outcome_price) as best
        FROM odds
        WHERE home_team = ? AND away_team = ?
          AND market = 'totals'
        GROUP BY outcome_name, outcome_point
    """, (home, away)).fetchall()

    for name, point, best in totals_rows:
        key = f"{name.lower()}_{point}"
        result["totals"][key] = best

    # Keep backward compat
    result["totals_2_5"] = {}
    if "over_2.5" in result["totals"]:
        result["totals_2_5"]["over"] = result["totals"]["over_2.5"]
    if "under_2.5" in result["totals"]:
        result["totals_2_5"]["under"] = result["totals"]["under_2.5"]

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
    Find all edges across upcoming matches and all supported markets.
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
        match_label = f"{home} vs {away}"
        match_date = match["commence_time"][:10]

        def add_edge(market, selection, model_p, best_price, impl_p=None):
            if best_price is None:
                return
            ev = compute_ev(model_p, best_price)
            if ev >= min_ev_pct:
                edges.append({
                    "match": match_label,
                    "date": match_date,
                    "market": market,
                    "selection": selection,
                    "model_prob": model_p,
                    "implied_prob": impl_p if impl_p else (1.0 / best_price if best_price else 0),
                    "best_odds": best_price,
                    "ev_pct": ev,
                })

        # 1X2
        for label, pred_key, odds_key in [
            (home, "home_win", "home"),
            ("Draw", "draw", "draw"),
            (away, "away_win", "away"),
        ]:
            best_price = odds.get("h2h", {}).get(odds_key)
            impl_p = market_implied.get(odds_key, 0)
            add_edge("1X2", label, pred[pred_key], best_price, impl_p)

        # Over/Under 1.5, 2.5, 3.5
        for line, pred_key in [
            (1.5, "over_1_5"), (2.5, "over_2_5"), (3.5, "over_3_5"),
        ]:
            over_price = odds.get("totals", {}).get(f"over_{line}")
            under_price = odds.get("totals", {}).get(f"under_{line}")
            add_edge(f"O/U {line}", f"Over {line}", pred[pred_key], over_price)
            add_edge(f"O/U {line}", f"Under {line}", 1.0 - pred[pred_key], under_price)

        # BTTS
        # Note: BTTS odds may come from some bookmakers under a different market key.
        # For now we compute model prob and check if any btts odds exist.
        btts_yes_price = odds.get("totals", {}).get("btts_yes")
        btts_no_price = odds.get("totals", {}).get("btts_no")
        if btts_yes_price:
            add_edge("BTTS", "Yes", pred["btts"], btts_yes_price)
        if btts_no_price:
            add_edge("BTTS", "No", 1.0 - pred["btts"], btts_no_price)

        # Double Chance (if we can derive from h2h odds or separate market)
        # Model probabilities are always available
        add_edge("DC", f"1X ({home}/Draw)", pred["dc_1x"],
                 odds.get("h2h", {}).get("dc_1x"))
        add_edge("DC", f"X2 (Draw/{away})", pred["dc_x2"],
                 odds.get("h2h", {}).get("dc_x2"))
        add_edge("DC", f"12 ({home}/{away})", pred["dc_12"],
                 odds.get("h2h", {}).get("dc_12"))

        # Asian Handicap edges (if odds are in DB)
        ah = pred.get("asian_handicap", {})
        for line, probs in ah.items():
            ah_home_key = f"home_{line}"
            ah_away_key = f"away_{line}"
            ah_home_price = odds.get("totals", {}).get(ah_home_key)
            ah_away_price = odds.get("totals", {}).get(ah_away_key)
            if ah_home_price:
                add_edge(f"AH {line:+.1f}", f"{home} {line:+.1f}",
                         probs["home"], ah_home_price)
            if ah_away_price:
                add_edge(f"AH {line:+.1f}", f"{away} {-line:+.1f}",
                         probs["away"], ah_away_price)

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
        print(f"{'Match':<35} {'Market':<12} {'Selection':<20} "
              f"{'Model':>6} {'Impl':>6} {'Odds':>6} {'EV%':>7}")
        print("-" * 100)
        for e in edges:
            print(f"  {e['match']:<33} {e['market']:<12} {e['selection']:<20} "
                  f"{e['model_prob']:>5.1%} {e['implied_prob']:>5.1%} "
                  f"{e['best_odds']:>6.2f} {e['ev_pct']:>+6.1f}%")

    print(f"\nTotal edges: {len(edges)}")

    # Summary by market type
    from collections import Counter
    mkt_counts = Counter(e["market"] for e in edges)
    print("\nEdges by market:")
    for mkt, count in mkt_counts.most_common():
        print(f"  {mkt}: {count}")

    conn.close()


if __name__ == "__main__":
    main()
