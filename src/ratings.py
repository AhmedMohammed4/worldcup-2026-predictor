"""
Brick 3: Team strength ratings.

From international_matches, computes:
1. Elo ratings for every team (recency-weighted).
2. Attack and defense strength parameters for the Poisson model.
3. Recent form factor (last N matches momentum).

Attack strength = team's goal-scoring rate relative to the global average.
Defense strength = team's goals-conceded rate relative to the global average.

These are estimated from recent international matches, with exponential
time-weighting so that a 2025 match matters more than a 2019 match.
Competitive matches (World Cup, qualifiers, continental cups) are weighted
higher than friendlies.

Writes results to the `team_ratings` table.
"""

import math
import sqlite3
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np

from teams import normalize_team

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "worldcup.db"

# Elo parameters
ELO_K = 40          # K-factor (higher = more responsive)
ELO_INIT = 1500     # Starting Elo for new teams
HOME_ADV_ELO = 100  # Elo points for home advantage (0 if neutral)

# For attack/defense estimation
HISTORY_START = "2018-01-01"  # Use matches from this date onward
HALF_LIFE_DAYS = 365          # Exponential decay half-life in days

# Tournament importance weights for attack/defense estimation
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 2.0,
    "FIFA World Cup qualification": 1.5,
    "UEFA Euro": 1.8,
    "UEFA Euro qualification": 1.3,
    "Copa America": 1.6,
    "African Cup of Nations": 1.5,
    "AFC Asian Cup": 1.4,
    "CONCACAF Gold Cup": 1.3,
    "UEFA Nations League": 1.3,
    "Confederations Cup": 1.4,
    "Friendly": 0.6,
}
DEFAULT_TOURNAMENT_WEIGHT = 1.0

# Form calculation
FORM_WINDOW = 10  # Number of recent matches for form factor


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def create_ratings_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_ratings (
            team TEXT PRIMARY KEY,
            elo REAL NOT NULL,
            attack REAL NOT NULL,
            defense REAL NOT NULL,
            matches_used INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()


def expected_score(rating_a: float, rating_b: float) -> float:
    """Elo expected score for player A against player B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def actual_score(home_goals: int, away_goals: int) -> tuple[float, float]:
    """Return (home_score, away_score) for Elo: 1=win, 0.5=draw, 0=loss."""
    if home_goals > away_goals:
        return 1.0, 0.0
    elif home_goals < away_goals:
        return 0.0, 1.0
    else:
        return 0.5, 0.5


def compute_elo(matches: list[tuple]) -> dict[str, float]:
    """
    Compute Elo ratings from all international matches.

    matches: list of (date, home_team, away_team, home_goals, away_goals, neutral)
    sorted by date ascending.

    Returns dict of team -> Elo rating.
    """
    elo = defaultdict(lambda: ELO_INIT)

    for _, home, away, hg, ag, neutral in matches:
        home_adv = 0 if neutral else HOME_ADV_ELO
        e_home = expected_score(elo[home] + home_adv, elo[away])
        e_away = 1.0 - e_home

        s_home, s_away = actual_score(hg, ag)

        # Goal difference multiplier (FIFA-style)
        gd = abs(hg - ag)
        if gd <= 1:
            g = 1.0
        elif gd == 2:
            g = 1.5
        else:
            g = (11.0 + gd) / 8.0

        elo[home] += ELO_K * g * (s_home - e_home)
        elo[away] += ELO_K * g * (s_away - e_away)

    return dict(elo)


def get_tournament_weight(tournament: str) -> float:
    """Get importance weight for a tournament type."""
    if tournament is None:
        return DEFAULT_TOURNAMENT_WEIGHT
    t = str(tournament).strip()
    for key, w in TOURNAMENT_WEIGHTS.items():
        if key.lower() in t.lower():
            return w
    if "friendly" in t.lower():
        return TOURNAMENT_WEIGHTS["Friendly"]
    return DEFAULT_TOURNAMENT_WEIGHT


def compute_attack_defense(
    matches: list[tuple],
    reference_date: date | None = None,
) -> tuple[dict[str, float], dict[str, float], float]:
    """
    Compute attack and defense strengths from recent matches.

    Uses exponential time-weighting and tournament importance weighting.
    Returns (attack_dict, defense_dict, global_avg).

    attack[team] = weighted goals scored per match / global avg
    defense[team] = weighted goals conceded per match / global avg
    """
    if reference_date is None:
        reference_date = date.today()

    decay = math.log(2) / HALF_LIFE_DAYS

    # Accumulate weighted goals scored/conceded and match weights
    goals_scored = defaultdict(float)
    goals_conceded = defaultdict(float)
    weight_sum = defaultdict(float)

    # matches can be 6-tuple (no tournament) or 7-tuple (with tournament)
    for row in matches:
        if len(row) >= 7:
            match_date_str, home, away, hg, ag, neutral, tournament = row[:7]
        else:
            match_date_str, home, away, hg, ag, neutral = row[:6]
            tournament = None

        try:
            match_date = datetime.strptime(match_date_str, "%Y-%m-%d").date()
        except ValueError:
            try:
                match_date = datetime.strptime(match_date_str, "%m/%d/%Y").date()
            except ValueError:
                continue
        days_ago = (reference_date - match_date).days
        if days_ago < 0:
            continue
        time_w = math.exp(-decay * days_ago)
        tourn_w = get_tournament_weight(tournament)
        w = time_w * tourn_w

        goals_scored[home] += w * hg
        goals_conceded[home] += w * ag
        weight_sum[home] += w

        goals_scored[away] += w * ag
        goals_conceded[away] += w * hg
        weight_sum[away] += w

    # Global average goals per team per match (weighted)
    total_goals = sum(goals_scored.values())
    total_weight = sum(weight_sum.values())
    global_avg = total_goals / total_weight if total_weight > 0 else 1.3

    attack = {}
    defense = {}
    for team in weight_sum:
        w = weight_sum[team]
        if w > 0:
            scored_per_match = goals_scored[team] / w
            conceded_per_match = goals_conceded[team] / w
            attack[team] = scored_per_match / global_avg if global_avg > 0 else 1.0
            defense[team] = conceded_per_match / global_avg if global_avg > 0 else 1.0
        else:
            attack[team] = 1.0
            defense[team] = 1.0

    return attack, defense, global_avg


def compute_form(
    matches: list[tuple],
    reference_date: date | None = None,
    window: int = FORM_WINDOW,
) -> dict[str, float]:
    """
    Compute recent form factor for each team based on last N matches.

    Returns dict of team -> form_factor (1.0 = average, >1 = good form, <1 = poor).
    Based on points per match relative to expected points from Elo.
    """
    if reference_date is None:
        reference_date = date.today()

    # Collect recent matches per team (already sorted by date)
    team_recent = defaultdict(list)
    for row in matches:
        match_date_str = row[0]
        home, away, hg, ag = row[1], row[2], row[3], row[4]
        try:
            match_date = datetime.strptime(match_date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if match_date >= reference_date:
            continue
        # Store (date, goals_for, goals_against) for each team
        team_recent[home].append((match_date, hg, ag))
        team_recent[away].append((match_date, ag, hg))

    form = {}
    for team, results in team_recent.items():
        # Sort by date descending, take last N
        results.sort(key=lambda x: x[0], reverse=True)
        recent = results[:window]
        if len(recent) < 3:
            form[team] = 1.0
            continue

        # Points per match: win=3, draw=1, loss=0, normalized to [0, 3]
        points = 0.0
        for _, gf, ga in recent:
            if gf > ga:
                points += 3.0
            elif gf == ga:
                points += 1.0
        ppg = points / len(recent)
        # Average PPG is roughly 1.3-1.5; normalize so 1.5 -> 1.0
        # Form factor: mild adjustment (0.9 to 1.1 range)
        form[team] = 0.9 + 0.2 * (ppg / 3.0)

    return form


def get_wc2026_teams(conn: sqlite3.Connection) -> set[str]:
    """Get the set of real team names in the 2026 World Cup."""
    rows = conn.execute("""
        SELECT DISTINCT team FROM (
            SELECT home_team AS team FROM matches WHERE tournament_year=2026
            UNION
            SELECT away_team AS team FROM matches WHERE tournament_year=2026
        )
        WHERE team NOT GLOB '[0-9]*'
          AND team NOT GLOB 'W[0-9]*'
          AND team NOT GLOB 'L[0-9]*'
          AND team NOT LIKE '%/%'
    """).fetchall()
    return {r[0] for r in rows}


def main():
    conn = get_db()
    create_ratings_table(conn)

    # Add form column if it doesn't exist
    try:
        conn.execute("ALTER TABLE team_ratings ADD COLUMN form REAL NOT NULL DEFAULT 1.0")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Load all international matches sorted by date (with tournament for weighting)
    print("Loading international matches...")
    all_matches = conn.execute("""
        SELECT date, home_team, away_team, home_goals, away_goals, neutral
        FROM international_matches
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
        ORDER BY date
    """).fetchall()
    all_matches_with_tourn = conn.execute("""
        SELECT date, home_team, away_team, home_goals, away_goals, neutral, tournament
        FROM international_matches
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
        ORDER BY date
    """).fetchall()
    print(f"  Total: {len(all_matches)} matches")

    # Compute Elo on full history
    print("\nComputing Elo ratings...")
    elo = compute_elo(all_matches)

    # Compute attack/defense from recent matches with tournament weighting
    print("Computing attack/defense strengths (tournament-weighted)...")
    recent = [m for m in all_matches_with_tourn if m[0] >= HISTORY_START]
    print(f"  Using {len(recent)} matches since {HISTORY_START}")
    attack, defense, global_avg = compute_attack_defense(recent)
    print(f"  Global average goals per team per match: {global_avg:.3f}")

    # Compute form
    print("Computing recent form...")
    form = compute_form(all_matches_with_tourn)
    form_teams = [t for t in form if abs(form[t] - 1.0) > 0.01]
    print(f"  {len(form_teams)} teams with notable form deviation")

    # Get 2026 teams
    wc_teams = get_wc2026_teams(conn)
    print(f"\n2026 World Cup teams: {len(wc_teams)}")

    # Check coverage
    missing_elo = wc_teams - set(elo.keys())
    missing_ad = wc_teams - set(attack.keys())
    if missing_elo:
        print(f"  WARNING: missing Elo for: {missing_elo}")
    if missing_ad:
        print(f"  WARNING: missing attack/defense for: {missing_ad}")

    # Upsert ratings
    now = datetime.now().isoformat()
    for team in elo:
        att = attack.get(team, 1.0)
        dfe = defense.get(team, 1.0)
        frm = form.get(team, 1.0)
        matches_used = sum(
            1 for m in recent
            if m[1] == team or m[2] == team
        )
        conn.execute("""
            INSERT INTO team_ratings (team, elo, attack, defense, matches_used, updated_at, form)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(team) DO UPDATE SET
                elo = excluded.elo,
                attack = excluded.attack,
                defense = excluded.defense,
                matches_used = excluded.matches_used,
                updated_at = excluded.updated_at,
                form = excluded.form
        """, (team, elo[team], att, dfe, matches_used, now, frm))
    conn.commit()

    # Print 2026 team ratings sorted by Elo
    print(f"\n{'Team':<25} {'Elo':>6} {'Atk':>6} {'Def':>6} {'Matches':>7}")
    print("-" * 55)
    rated = []
    for team in sorted(wc_teams):
        e = elo.get(team, ELO_INIT)
        a = attack.get(team, 1.0)
        d = defense.get(team, 1.0)
        rated.append((team, e, a, d))

    rated.sort(key=lambda x: x[1], reverse=True)
    for team, e, a, d in rated:
        m_used = sum(1 for m in recent if m[1] == team or m[2] == team)
        print(f"  {team:<23} {e:>6.0f} {a:>6.2f} {d:>6.2f} {m_used:>7}")

    (total,) = conn.execute("SELECT COUNT(*) FROM team_ratings").fetchone()
    print(f"\nTotal teams in team_ratings: {total}")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
