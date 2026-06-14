"""
Brick 4: Poisson match model with Dixon-Coles low-score correction.

predict_match(home, away, neutral=True) returns:
  - P(home win), P(draw), P(away win)
  - P(over 2.5 goals), P(BTTS)
  - Scoreline probability grid (up to 10x10)

Uses attack/defense strengths from ratings.py and a global average
goals rate. Applies the Dixon-Coles tau correction for correlated
low-scoring outcomes (0-0, 1-0, 0-1, 1-1).
"""

import sqlite3
from pathlib import Path

import numpy as np
from scipy.stats import poisson

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "worldcup.db"

MAX_GOALS = 10  # Grid goes from 0 to MAX_GOALS-1
HOME_ADVANTAGE = 1.10  # Multiplier on home team's expected goals (non-neutral)

# Dixon-Coles correlation parameter.
# Negative rho reduces P(0-0) and P(1-1), increases P(1-0) and P(0-1).
# Typical fitted values are around -0.05 to -0.15.
DC_RHO = -0.08


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def load_ratings(conn: sqlite3.Connection) -> dict:
    """Load team ratings into a dict of {team: (elo, attack, defense)}."""
    rows = conn.execute(
        "SELECT team, elo, attack, defense FROM team_ratings"
    ).fetchall()
    return {r[0]: {"elo": r[1], "attack": r[2], "defense": r[3]} for r in rows}


def load_global_avg(conn: sqlite3.Connection) -> float:
    """Estimate global avg goals per team per match from recent international data."""
    row = conn.execute("""
        SELECT
            CAST(SUM(home_goals + away_goals) AS REAL) / (2.0 * COUNT(*))
        FROM international_matches
        WHERE date >= '2018-01-01'
          AND home_goals IS NOT NULL
    """).fetchone()
    return row[0] if row[0] else 1.3


def dixon_coles_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """
    Dixon-Coles correction factor for low-score outcomes.
    x = home goals, y = away goals, lam = home rate, mu = away rate.
    Only adjusts (0,0), (1,0), (0,1), (1,1). Returns 1.0 otherwise.
    """
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    elif x == 1 and y == 0:
        return 1.0 + mu * rho
    elif x == 0 and y == 1:
        return 1.0 + lam * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0


def predict_match(
    home_team: str,
    away_team: str,
    neutral: bool = True,
    ratings: dict | None = None,
    global_avg: float | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict:
    """
    Predict a match between home_team and away_team.

    Returns a dict with:
        home_win, draw, away_win: probabilities
        over_2_5: P(total goals > 2.5)
        btts: P(both teams score)
        expected_home: expected goals for home
        expected_away: expected goals for away
        scoreline_grid: numpy array of shape (MAX_GOALS, MAX_GOALS)
    """
    close_conn = False
    if conn is None:
        conn = get_db()
        close_conn = True

    if ratings is None:
        ratings = load_ratings(conn)
    if global_avg is None:
        global_avg = load_global_avg(conn)

    home_r = ratings.get(home_team)
    away_r = ratings.get(away_team)

    if home_r is None:
        raise ValueError(f"No ratings found for '{home_team}'")
    if away_r is None:
        raise ValueError(f"No ratings found for '{away_team}'")

    # Expected goals
    home_adv = 1.0 if neutral else HOME_ADVANTAGE
    lam = global_avg * home_r["attack"] * away_r["defense"] * home_adv
    mu = global_avg * away_r["attack"] * home_r["defense"]

    # Clamp to reasonable range
    lam = max(0.1, min(lam, 6.0))
    mu = max(0.1, min(mu, 6.0))

    # Build scoreline probability grid with Dixon-Coles correction
    grid = np.zeros((MAX_GOALS, MAX_GOALS))
    home_pmf = poisson.pmf(np.arange(MAX_GOALS), lam)
    away_pmf = poisson.pmf(np.arange(MAX_GOALS), mu)

    for i in range(MAX_GOALS):
        for j in range(MAX_GOALS):
            tau = dixon_coles_tau(i, j, lam, mu, DC_RHO)
            grid[i, j] = home_pmf[i] * away_pmf[j] * tau

    # Renormalize so probabilities sum to 1
    grid /= grid.sum()

    # Derive match outcome probabilities
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    over_2_5 = 0.0
    btts = 0.0

    for i in range(MAX_GOALS):
        for j in range(MAX_GOALS):
            p = grid[i, j]
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p
            if i + j > 2:
                over_2_5 += p
            if i >= 1 and j >= 1:
                btts += p

    if close_conn:
        conn.close()

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over_2_5": over_2_5,
        "btts": btts,
        "expected_home": lam,
        "expected_away": mu,
        "scoreline_grid": grid,
    }


def format_prediction(pred: dict) -> str:
    """Format a prediction dict as a readable string."""
    lines = []
    lines.append(f"{pred['home_team']} vs {pred['away_team']}")
    lines.append(f"  Expected goals: {pred['expected_home']:.2f} - {pred['expected_away']:.2f}")
    lines.append(f"  Home win: {pred['home_win']:.1%}")
    lines.append(f"  Draw:     {pred['draw']:.1%}")
    lines.append(f"  Away win: {pred['away_win']:.1%}")
    lines.append(f"  Over 2.5: {pred['over_2_5']:.1%}")
    lines.append(f"  BTTS:     {pred['btts']:.1%}")

    # Most likely scorelines
    grid = pred["scoreline_grid"]
    flat = [(grid[i, j], i, j) for i in range(MAX_GOALS) for j in range(MAX_GOALS)]
    flat.sort(reverse=True)
    lines.append("  Top scorelines:")
    for p, i, j in flat[:5]:
        lines.append(f"    {i}-{j}: {p:.1%}")

    return "\n".join(lines)


def main():
    conn = get_db()
    ratings = load_ratings(conn)
    global_avg = load_global_avg(conn)

    test_matches = [
        ("Spain", "Brazil"),
        ("Argentina", "Germany"),
        ("France", "England"),
        ("USA", "Mexico"),
        ("Japan", "South Korea"),
        ("Qatar", "Cape Verde"),
    ]

    print(f"Global avg goals per team per match: {global_avg:.3f}\n")

    for home, away in test_matches:
        pred = predict_match(home, away, neutral=True, ratings=ratings, global_avg=global_avg, conn=conn)
        print(format_prediction(pred))
        total = pred["home_win"] + pred["draw"] + pred["away_win"]
        print(f"  Sum check: {total:.6f}")
        print()

    conn.close()


if __name__ == "__main__":
    main()
