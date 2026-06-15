"""
Brick 4: Poisson match model with Dixon-Coles low-score correction.

predict_match(home, away, neutral=True) returns:
  - P(home win), P(draw), P(away win)
  - P(over 2.5 goals), P(BTTS)
  - Asian handicap probabilities for common lines
  - Double chance probabilities (1X, X2, 12)
  - Correct score probabilities (top N)
  - Scoreline probability grid (up to 10x10)

Uses attack/defense strengths and form factor from ratings.py and a global
average goals rate. Applies the Dixon-Coles tau correction for correlated
low-scoring outcomes (0-0, 1-0, 0-1, 1-1).

The rho parameter can be fitted from historical data using fit_dc_rho().
"""

import sqlite3
from pathlib import Path

import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize_scalar

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "worldcup.db"

MAX_GOALS = 10  # Grid goes from 0 to MAX_GOALS-1
HOME_ADVANTAGE = 1.10  # Multiplier on home team's expected goals (non-neutral)

# Dixon-Coles correlation parameter (default; can be fitted with fit_dc_rho).
DC_RHO = -0.08


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def load_ratings(conn: sqlite3.Connection) -> dict:
    """Load team ratings into a dict of {team: {elo, attack, defense, form}}."""
    rows = conn.execute(
        "SELECT team, elo, attack, defense, form FROM team_ratings"
    ).fetchall()
    return {
        r[0]: {"elo": r[1], "attack": r[2], "defense": r[3],
               "form": r[4] if r[4] else 1.0}
        for r in rows
    }


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


def build_scoreline_grid(lam: float, mu: float, rho: float = DC_RHO) -> np.ndarray:
    """Build a normalized scoreline probability grid with Dixon-Coles correction."""
    grid = np.zeros((MAX_GOALS, MAX_GOALS))
    home_pmf = poisson.pmf(np.arange(MAX_GOALS), lam)
    away_pmf = poisson.pmf(np.arange(MAX_GOALS), mu)

    for i in range(MAX_GOALS):
        for j in range(MAX_GOALS):
            tau = dixon_coles_tau(i, j, lam, mu, rho)
            grid[i, j] = home_pmf[i] * away_pmf[j] * tau

    grid /= grid.sum()
    return grid


def grid_to_markets(grid: np.ndarray) -> dict:
    """Extract all market probabilities from a scoreline grid."""
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    over_2_5 = 0.0
    over_1_5 = 0.0
    over_3_5 = 0.0
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
            if i + j > 1:
                over_1_5 += p
            if i + j > 3:
                over_3_5 += p
            if i >= 1 and j >= 1:
                btts += p

    # Double chance
    dc_1x = home_win + draw
    dc_x2 = draw + away_win
    dc_12 = home_win + away_win

    # Asian handicap: P(team covers the line)
    # Common lines: -0.5, -1, -1.5, +0.5, +1, +1.5
    asian_handicap = {}
    for line in [-2.5, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.5]:
        home_covers = 0.0
        away_covers = 0.0
        push = 0.0
        for i in range(MAX_GOALS):
            for j in range(MAX_GOALS):
                margin = i - j  # home perspective
                adjusted = margin + line
                if adjusted > 0:
                    home_covers += grid[i, j]
                elif adjusted < 0:
                    away_covers += grid[i, j]
                else:
                    push += grid[i, j]
        asian_handicap[line] = {
            "home": home_covers,
            "away": away_covers,
            "push": push,
        }

    # Top correct scores
    flat = [(grid[i, j], i, j) for i in range(MAX_GOALS) for j in range(MAX_GOALS)]
    flat.sort(reverse=True)
    correct_scores = [(i, j, p) for p, i, j in flat[:15]]

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over_1_5": over_1_5,
        "over_2_5": over_2_5,
        "over_3_5": over_3_5,
        "btts": btts,
        "dc_1x": dc_1x,
        "dc_x2": dc_x2,
        "dc_12": dc_12,
        "asian_handicap": asian_handicap,
        "correct_scores": correct_scores,
    }


def predict_match(
    home_team: str,
    away_team: str,
    neutral: bool = True,
    ratings: dict | None = None,
    global_avg: float | None = None,
    conn: sqlite3.Connection | None = None,
    rho: float = DC_RHO,
) -> dict:
    """
    Predict a match between home_team and away_team.

    Returns a dict with all market probabilities, expected goals, and the
    scoreline grid.
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

    # Expected goals with form factor
    home_adv = 1.0 if neutral else HOME_ADVANTAGE
    home_form = home_r.get("form", 1.0)
    away_form = away_r.get("form", 1.0)

    lam = global_avg * home_r["attack"] * away_r["defense"] * home_adv * home_form
    mu = global_avg * away_r["attack"] * home_r["defense"] * away_form

    # Clamp to reasonable range
    lam = max(0.1, min(lam, 6.0))
    mu = max(0.1, min(mu, 6.0))

    # Build scoreline probability grid with Dixon-Coles correction
    grid = build_scoreline_grid(lam, mu, rho)

    # Derive all market probabilities
    markets = grid_to_markets(grid)

    if close_conn:
        conn.close()

    return {
        "home_team": home_team,
        "away_team": away_team,
        "expected_home": lam,
        "expected_away": mu,
        "scoreline_grid": grid,
        # Core 1X2
        "home_win": markets["home_win"],
        "draw": markets["draw"],
        "away_win": markets["away_win"],
        # Totals
        "over_1_5": markets["over_1_5"],
        "over_2_5": markets["over_2_5"],
        "over_3_5": markets["over_3_5"],
        # BTTS
        "btts": markets["btts"],
        # Double chance
        "dc_1x": markets["dc_1x"],
        "dc_x2": markets["dc_x2"],
        "dc_12": markets["dc_12"],
        # Asian handicap
        "asian_handicap": markets["asian_handicap"],
        # Correct scores
        "correct_scores": markets["correct_scores"],
    }


def fit_dc_rho(matches: list[dict], ratings: dict, global_avg: float) -> float:
    """
    Fit the Dixon-Coles rho parameter by maximizing log-likelihood on
    historical match results.

    matches: list of dicts with home, away, hg, ag keys.
    Returns the optimal rho value.
    """
    def neg_log_likelihood(rho):
        ll = 0.0
        for m in matches:
            home_r = ratings.get(m["home"])
            away_r = ratings.get(m["away"])
            if home_r is None or away_r is None:
                continue
            lam = global_avg * home_r["attack"] * away_r["defense"]
            mu = global_avg * away_r["attack"] * home_r["defense"]
            lam = max(0.1, min(lam, 6.0))
            mu = max(0.1, min(mu, 6.0))

            home_pmf = poisson.pmf(np.arange(MAX_GOALS), lam)
            away_pmf = poisson.pmf(np.arange(MAX_GOALS), mu)

            hg = min(m["hg"], MAX_GOALS - 1)
            ag = min(m["ag"], MAX_GOALS - 1)
            tau = dixon_coles_tau(hg, ag, lam, mu, rho)
            p = home_pmf[hg] * away_pmf[ag] * tau
            if p > 0:
                ll += np.log(p)
            else:
                ll += -50  # penalty for zero probability
        return -ll

    result = minimize_scalar(neg_log_likelihood, bounds=(-0.3, 0.3), method="bounded")
    return result.x


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
    lines.append(f"  DC 1X:    {pred['dc_1x']:.1%}")
    lines.append(f"  DC X2:    {pred['dc_x2']:.1%}")

    # Top scorelines
    lines.append("  Top scorelines:")
    for hg, ag, p in pred["correct_scores"][:5]:
        lines.append(f"    {hg}-{ag}: {p:.1%}")

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
