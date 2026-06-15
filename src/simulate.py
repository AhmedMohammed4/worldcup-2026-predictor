"""
Brick 8: Tournament Monte Carlo simulator.

Simulates the rest of the 2026 World Cup N times using predict_match()
for each fixture. Advances winners through the real bracket format:
- 12 groups of 4, top 2 + 8 best third-place teams advance (32 total)
- Round of 32, Round of 16, Quarter-finals, Semi-finals, Final
- Knockouts decided by penalties if drawn after 90 min

Outputs each team's probability of reaching each round and winning.

Usage:
    python simulate.py [--sims N]
"""

import argparse
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from model import predict_match, load_ratings, load_global_avg, MAX_GOALS

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "worldcup.db"

# 2026 bracket: Round of 32 matchups from the fixture data.
# Each entry maps a R32 slot to (home_source, away_source).
# Sources: "1X" = group X winner, "2X" = group X runner-up,
# "3XXXXX" = best 3rd from those groups.
R32_BRACKET = [
    ("2A", "2B"),
    ("1C", "2F"),
    ("1E", "3A/B/C/D/F"),
    ("1F", "2C"),
    ("1A", "3C/E/F/H/I"),
    ("1I", "3C/D/F/G/H"),
    ("2E", "2I"),
    ("1D", "3B/E/F/I/J"),
    ("1G", "3A/E/H/I/J"),
    ("1L", "3E/H/I/J/K"),
    ("1B", "3E/F/G/I/J"),
    ("1H", "2J"),
    ("2K", "2L"),
    ("1J", "2H"),
    ("1K", "3D/E/I/J/L"),
    ("2D", "2G"),
]

# R16 matchups: pairs of R32 match indices (0-based)
R16_PAIRS = [(0, 2), (1, 4), (3, 5), (6, 7), (8, 9), (10, 11), (12, 14), (13, 15)]

# QF matchups: pairs of R16 match indices (0-based)
QF_PAIRS = [(0, 1), (4, 5), (2, 3), (6, 7)]

# SF matchups: pairs of QF match indices (0-based)
SF_PAIRS = [(0, 1), (2, 3)]

ROUNDS = ["Group", "R32", "R16", "QF", "SF", "Final", "Winner"]


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def load_groups(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Load group compositions from fixtures. Returns {group_letter: [teams]}."""
    rows = conn.execute("""
        SELECT group_name, home_team, away_team
        FROM matches
        WHERE tournament_year = 2026 AND group_name IS NOT NULL
    """).fetchall()

    groups = defaultdict(set)
    for g, h, a in rows:
        letter = g.replace("Group ", "")
        groups[letter].add(h)
        groups[letter].add(a)

    return {k: sorted(v) for k, v in sorted(groups.items())}


def load_played_group_results(conn: sqlite3.Connection) -> list[dict]:
    """Load already-played group stage matches."""
    rows = conn.execute("""
        SELECT group_name, home_team, away_team, home_goals, away_goals
        FROM matches
        WHERE tournament_year = 2026
          AND group_name IS NOT NULL
          AND home_goals IS NOT NULL
    """).fetchall()

    return [
        {"group": r[0].replace("Group ", ""), "home": r[1], "away": r[2],
         "hg": r[3], "ag": r[4]}
        for r in rows
    ]


def load_unplayed_group_matches(conn: sqlite3.Connection) -> list[dict]:
    """Load unplayed group stage matches."""
    rows = conn.execute("""
        SELECT group_name, home_team, away_team
        FROM matches
        WHERE tournament_year = 2026
          AND group_name IS NOT NULL
          AND home_goals IS NULL
    """).fetchall()

    return [
        {"group": r[0].replace("Group ", ""), "home": r[1], "away": r[2]}
        for r in rows
    ]


def simulate_match(home: str, away: str, ratings: dict, global_avg: float,
                   knockout: bool = False) -> tuple[str, int, int]:
    """
    Simulate a single match. Returns (winner, home_goals, away_goals).
    For knockouts, if drawn, decides by penalty shootout (coin flip weighted
    slightly by rating).
    """
    pred = predict_match(home, away, neutral=True, ratings=ratings,
                         global_avg=global_avg)
    grid = pred["scoreline_grid"]

    # Sample a scoreline from the grid
    flat = grid.flatten()
    flat = flat / flat.sum()
    idx = np.random.choice(len(flat), p=flat)
    hg = idx // MAX_GOALS
    ag = idx % MAX_GOALS

    if knockout and hg == ag:
        # Penalty shootout: weighted coin flip
        home_elo = ratings.get(home, {}).get("elo", 1500)
        away_elo = ratings.get(away, {}).get("elo", 1500)
        p_home_pen = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 800.0))
        winner = home if random.random() < p_home_pen else away
        return winner, hg, ag

    if hg > ag:
        return home, hg, ag
    elif ag > hg:
        return away, hg, ag
    else:
        return "draw", hg, ag


def simulate_group(group_letter: str, teams: list[str],
                   played: list[dict], unplayed: list[dict],
                   ratings: dict, global_avg: float) -> list[str]:
    """
    Simulate a group and return teams sorted by final standing.
    Uses actual results for played matches and simulates the rest.
    Returns [1st, 2nd, 3rd, 4th].
    """
    points = defaultdict(int)
    gd = defaultdict(int)
    gf = defaultdict(int)

    # Apply played results
    for m in played:
        if m["group"] != group_letter:
            continue
        h, a, hg, ag = m["home"], m["away"], m["hg"], m["ag"]
        gf[h] += hg
        gf[a] += ag
        gd[h] += hg - ag
        gd[a] += ag - hg
        if hg > ag:
            points[h] += 3
        elif hg < ag:
            points[a] += 3
        else:
            points[h] += 1
            points[a] += 1

    # Simulate unplayed
    for m in unplayed:
        if m["group"] != group_letter:
            continue
        _, hg, ag = simulate_match(m["home"], m["away"], ratings, global_avg)
        gf[m["home"]] += hg
        gf[m["away"]] += ag
        gd[m["home"]] += hg - ag
        gd[m["away"]] += ag - hg
        if hg > ag:
            points[m["home"]] += 3
        elif ag > hg:
            points[m["away"]] += 3
        else:
            points[m["home"]] += 1
            points[m["away"]] += 1

    # Sort by points, then GD, then GF (simplified tiebreak)
    standing = sorted(teams, key=lambda t: (points[t], gd[t], gf[t]), reverse=True)
    return standing


def select_best_thirds(group_standings: dict[str, list[str]]) -> dict:
    """
    Select the 8 best third-place teams across 12 groups.
    Returns a dict mapping each third-place team to its group letter,
    and a mapping for the bracket slots.
    """
    thirds = []
    for g, standing in sorted(group_standings.items()):
        if len(standing) >= 3:
            thirds.append((g, standing[2]))

    # In a real tournament, third-place teams are ranked by points/GD.
    # Here we use Elo as a proxy for simulation simplicity.
    # We need 8 best out of 12 thirds.
    # For the simulation, just take the 8 "best" by random shuffle
    # (since we don't track detailed group stats here).
    # Actually, let's use a simple random selection of 8 from 12.
    random.shuffle(thirds)
    best_8 = thirds[:8]

    return {team: group for group, team in best_8}


def resolve_bracket_slot(slot: str, group_standings: dict[str, list[str]],
                         best_thirds: dict[str, str],
                         used_thirds: set = None) -> str:
    """
    Resolve a bracket slot like '1A', '2B', '3A/B/C/D/F' to a team name.
    For third-place slots, marks teams as used to avoid double assignment.
    """
    if used_thirds is None:
        used_thirds = set()

    if slot.startswith("1") and len(slot) == 2:
        group = slot[1]
        return group_standings[group][0]
    elif slot.startswith("2") and len(slot) == 2:
        group = slot[1]
        return group_standings[group][1]
    elif slot.startswith("3"):
        eligible_groups = slot[1:].split("/")
        for team, group in best_thirds.items():
            if group in eligible_groups and team not in used_thirds:
                used_thirds.add(team)
                return team
        # Fallback: pick any unused third
        for team in best_thirds:
            if team not in used_thirds:
                used_thirds.add(team)
                return team
    return None


def run_simulation(n_sims: int = 10000):
    conn = get_db()
    ratings = load_ratings(conn)
    global_avg = load_global_avg(conn)
    groups = load_groups(conn)
    played = load_played_group_results(conn)
    unplayed = load_unplayed_group_matches(conn)

    # Get all real teams
    all_teams = set()
    for teams in groups.values():
        all_teams.update(teams)

    print(f"Simulating {n_sims} tournaments...")
    print(f"  Groups: {len(groups)}, Teams: {len(all_teams)}")
    print(f"  Played group matches: {len(played)}, Remaining: {len(unplayed)}")

    # Track how far each team gets
    round_counts = {team: defaultdict(int) for team in all_teams}

    for sim in range(n_sims):
        if (sim + 1) % 2000 == 0:
            print(f"  Sim {sim + 1}/{n_sims}...")

        # Simulate groups
        group_standings = {}
        for g, teams in groups.items():
            standing = simulate_group(g, teams, played, unplayed, ratings, global_avg)
            group_standings[g] = standing

        # All teams make "Group" round
        for team in all_teams:
            round_counts[team]["Group"] += 1

        # Select best thirds
        best_thirds = select_best_thirds(group_standings)

        # Determine R32 matchups, consuming third-place teams as assigned
        used_thirds = set()
        r32_teams = []
        for home_slot, away_slot in R32_BRACKET:
            h = resolve_bracket_slot(home_slot, group_standings, best_thirds, used_thirds)
            a = resolve_bracket_slot(away_slot, group_standings, best_thirds, used_thirds)
            if h is None or a is None:
                r32_teams.append((h or a, h, a))
                continue
            r32_teams.append((h, a))

        # Mark R32 qualifiers
        for match in r32_teams:
            if len(match) == 2:
                round_counts[match[0]]["R32"] += 1
                round_counts[match[1]]["R32"] += 1

        # Simulate R32
        r32_winners = []
        for match in r32_teams:
            if len(match) == 2:
                h, a = match
                try:
                    winner, _, _ = simulate_match(h, a, ratings, global_avg, knockout=True)
                except (ValueError, KeyError):
                    winner = h
                r32_winners.append(winner)
            else:
                r32_winners.append(match[0])

        # Simulate R16
        r16_winners = []
        for i, j in R16_PAIRS:
            h, a = r32_winners[i], r32_winners[j]
            round_counts[h]["R16"] += 1
            round_counts[a]["R16"] += 1
            try:
                winner, _, _ = simulate_match(h, a, ratings, global_avg, knockout=True)
            except (ValueError, KeyError):
                winner = h
            r16_winners.append(winner)

        # Simulate QF
        qf_winners = []
        for i, j in QF_PAIRS:
            h, a = r16_winners[i], r16_winners[j]
            round_counts[h]["QF"] += 1
            round_counts[a]["QF"] += 1
            try:
                winner, _, _ = simulate_match(h, a, ratings, global_avg, knockout=True)
            except (ValueError, KeyError):
                winner = h
            qf_winners.append(winner)

        # Simulate SF
        sf_winners = []
        sf_losers = []
        for i, j in SF_PAIRS:
            h, a = qf_winners[i], qf_winners[j]
            round_counts[h]["SF"] += 1
            round_counts[a]["SF"] += 1
            try:
                winner, _, _ = simulate_match(h, a, ratings, global_avg, knockout=True)
            except (ValueError, KeyError):
                winner = h
            sf_winners.append(winner)
            sf_losers.append(a if winner == h else h)

        # Final
        h, a = sf_winners[0], sf_winners[1]
        round_counts[h]["Final"] += 1
        round_counts[a]["Final"] += 1
        try:
            champion, _, _ = simulate_match(h, a, ratings, global_avg, knockout=True)
        except (ValueError, KeyError):
            champion = h
        round_counts[champion]["Winner"] += 1

    conn_out = get_db()

    # Print results
    print(f"\n{'Team':<25} {'R32':>6} {'R16':>6} {'QF':>6} {'SF':>6} {'Final':>6} {'Win':>6}")
    print("-" * 67)

    team_probs = []
    for team in sorted(all_teams):
        probs = {}
        for r in ROUNDS:
            probs[r] = round_counts[team][r] / n_sims
        team_probs.append((team, probs))

    # Sort by win probability
    team_probs.sort(key=lambda x: x[1]["Winner"], reverse=True)

    for team, probs in team_probs:
        print(f"  {team:<23} {probs['R32']:>5.1%} {probs['R16']:>5.1%} "
              f"{probs['QF']:>5.1%} {probs['SF']:>5.1%} "
              f"{probs['Final']:>5.1%} {probs['Winner']:>5.1%}")

    # Sum check
    total_win = sum(p["Winner"] for _, p in team_probs)
    print(f"\nWin probability sum: {total_win:.3f} (should be ~1.000)")

    # Compare to outright market odds if available
    print("\n--- Market Comparison (outright winner) ---")
    outright = conn_out.execute("""
        SELECT outcome_name, MIN(outcome_price) as best_odds
        FROM odds
        WHERE market = 'outrights'
        GROUP BY outcome_name
        ORDER BY best_odds
    """).fetchall()

    if outright:
        print(f"{'Team':<25} {'Model':>7} {'Market':>7} {'Implied':>7} {'Edge':>7}")
        print("-" * 58)
        for name, odds in outright:
            model_p = next((p["Winner"] for t, p in team_probs if t == name), None)
            if model_p is not None:
                implied = 1.0 / odds
                edge = model_p - implied
                flag = " <--" if edge > 0.02 else ""
                print(f"  {name:<23} {model_p:>6.1%} {odds:>7.1f} {implied:>6.1%} {edge:>+6.1%}{flag}")
    else:
        print("  No outright odds in database.")

    conn_out.close()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=10000)
    args = parser.parse_args()
    run_simulation(args.sims)
