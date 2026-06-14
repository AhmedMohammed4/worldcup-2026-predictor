"""
Brick 6: Walk-forward backtester.

For each played World Cup match (2018, 2022, 2026), fits ratings using only
data before that match, predicts the outcome, and evaluates against actuals.

Reports: accuracy, log loss, and simulated ROI from betting edges.

Usage:
    python backtest.py
"""

import math
import sqlite3
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path

import numpy as np

from ratings import (
    compute_elo,
    compute_attack_defense,
    ELO_INIT,
    HISTORY_START,
)
from model import predict_match, MAX_GOALS

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "worldcup.db"

# Edge threshold for simulated betting
SIM_EV_THRESHOLD = 5.0  # percent
# Simulated fair odds margin (typical book overround ~5%)
BOOK_MARGIN = 0.05


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def load_international_matches(conn: sqlite3.Connection) -> list[tuple]:
    """Load all international matches sorted by date."""
    return conn.execute("""
        SELECT date, home_team, away_team, home_goals, away_goals, neutral
        FROM international_matches
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
        ORDER BY date
    """).fetchall()


def load_wc_matches(conn: sqlite3.Connection, years: list[int]) -> list[dict]:
    """Load played World Cup matches sorted by date."""
    placeholders = ",".join("?" * len(years))
    rows = conn.execute(f"""
        SELECT tournament_year, date, round, group_name,
               home_team, away_team, home_goals, away_goals
        FROM matches
        WHERE tournament_year IN ({placeholders})
          AND home_goals IS NOT NULL
        ORDER BY date, home_team
    """, years).fetchall()

    return [
        {
            "year": r[0], "date": r[1], "round": r[2], "group": r[3],
            "home": r[4], "away": r[5], "hg": r[6], "ag": r[7],
        }
        for r in rows
    ]


def actual_outcome(hg: int, ag: int) -> str:
    if hg > ag:
        return "home"
    elif hg < ag:
        return "away"
    return "draw"


def log_loss_single(prob: float) -> float:
    """Log loss for a single prediction. Clamp to avoid log(0)."""
    prob = max(min(prob, 1.0 - 1e-15), 1e-15)
    return -math.log(prob)


def fair_odds_from_prob(prob: float, margin: float = BOOK_MARGIN) -> float:
    """Simulate bookmaker odds: fair odds with a margin applied."""
    if prob <= 0:
        return 100.0
    fair = 1.0 / prob
    # Apply margin: odds are slightly lower than fair
    return fair * (1.0 - margin)


def run_backtest(years: list[int] = None):
    if years is None:
        years = [2018, 2022, 2026]

    conn = get_db()
    all_intl = load_international_matches(conn)
    wc_matches = load_wc_matches(conn, years)

    print(f"Backtesting {len(wc_matches)} played World Cup matches across {years}")
    print(f"International match pool: {len(all_intl)} matches\n")

    # Pre-compute Elo snapshots by processing all matches and storing
    # the rating state at each unique date boundary.
    print("Computing Elo snapshots...")
    elo_snapshots = {}
    elo_state = defaultdict(lambda: ELO_INIT)
    current_date = None

    for m_date, home, away, hg, ag, neutral in all_intl:
        if m_date != current_date:
            # Save snapshot at previous date
            if current_date is not None:
                elo_snapshots[current_date] = dict(elo_state)
            current_date = m_date

        # Update Elo
        home_adv = 0 if neutral else 100
        dr = elo_state[away] - (elo_state[home] + home_adv)
        e_home = 1.0 / (1.0 + 10 ** (dr / 400.0))
        e_away = 1.0 - e_home

        if hg > ag:
            s_home, s_away = 1.0, 0.0
        elif hg < ag:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        gd = abs(hg - ag)
        g = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11.0 + gd) / 8.0)

        elo_state[home] += 40 * g * (s_home - e_home)
        elo_state[away] += 40 * g * (s_away - e_away)

    # Save final snapshot
    if current_date is not None:
        elo_snapshots[current_date] = dict(elo_state)

    # Get sorted snapshot dates for binary search
    snapshot_dates = sorted(elo_snapshots.keys())
    print(f"  {len(snapshot_dates)} date snapshots created\n")

    def get_elo_before(match_date: str) -> dict:
        """Get the most recent Elo snapshot before match_date."""
        import bisect
        idx = bisect.bisect_left(snapshot_dates, match_date)
        if idx == 0:
            return {}
        return elo_snapshots[snapshot_dates[idx - 1]]

    # Run walk-forward predictions
    results = []
    for i, m in enumerate(wc_matches):
        match_date = m["date"]

        # Get Elo ratings as of before this match
        elo = get_elo_before(match_date)

        # Get attack/defense from international matches before this date
        prior_intl = [x for x in all_intl if x[0] < match_date and x[0] >= HISTORY_START]

        if len(prior_intl) < 100:
            # Not enough data, skip
            continue

        attack, defense, global_avg = compute_attack_defense(
            prior_intl,
            reference_date=datetime.strptime(match_date, "%Y-%m-%d").date(),
        )

        # Build ratings dict for model
        ratings = {}
        for team in set(list(elo.keys()) + list(attack.keys())):
            ratings[team] = {
                "elo": elo.get(team, ELO_INIT),
                "attack": attack.get(team, 1.0),
                "defense": defense.get(team, 1.0),
            }

        try:
            pred = predict_match(
                m["home"], m["away"], neutral=True,
                ratings=ratings, global_avg=global_avg, conn=conn,
            )
        except ValueError:
            continue

        outcome = actual_outcome(m["hg"], m["ag"])
        pred_outcome = max(
            [("home", pred["home_win"]), ("draw", pred["draw"]), ("away", pred["away_win"])],
            key=lambda x: x[1],
        )[0]

        # Probability assigned to actual outcome
        outcome_prob = {
            "home": pred["home_win"],
            "draw": pred["draw"],
            "away": pred["away_win"],
        }[outcome]

        results.append({
            "year": m["year"],
            "date": m["date"],
            "home": m["home"],
            "away": m["away"],
            "hg": m["hg"],
            "ag": m["ag"],
            "elo_home": elo.get(m["home"], ELO_INIT),
            "elo_away": elo.get(m["away"], ELO_INIT),
            "outcome": outcome,
            "pred_outcome": pred_outcome,
            "correct": outcome == pred_outcome,
            "p_home": pred["home_win"],
            "p_draw": pred["draw"],
            "p_away": pred["away_win"],
            "p_actual": outcome_prob,
            "log_loss": log_loss_single(outcome_prob),
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(wc_matches)} matches...")

    print(f"\n  Completed: {len(results)} predictions\n")

    if not results:
        print("No results to report.")
        conn.close()
        return

    # Overall metrics
    correct = sum(r["correct"] for r in results)
    total = len(results)
    accuracy = correct / total
    avg_log_loss = sum(r["log_loss"] for r in results) / total

    print("=" * 70)
    print("BACKTEST REPORT")
    print("=" * 70)
    print(f"\nMatches predicted: {total}")
    print(f"Accuracy (1X2):   {correct}/{total} = {accuracy:.1%}")
    print(f"Avg log loss:     {avg_log_loss:.4f}")

    # Per-year breakdown
    for yr in sorted(set(r["year"] for r in results)):
        yr_results = [r for r in results if r["year"] == yr]
        yr_correct = sum(r["correct"] for r in yr_results)
        yr_total = len(yr_results)
        yr_ll = sum(r["log_loss"] for r in yr_results) / yr_total
        print(f"  {yr}: {yr_correct}/{yr_total} = {yr_correct/yr_total:.1%} accuracy, "
              f"log loss = {yr_ll:.4f}")

    # Simulated ROI: use Elo-based "market" odds as a proxy for real market.
    # We simulate what a bookmaker might offer using Elo win expectancy
    # (independent of our attack/defense model), plus a vig.
    # This tests whether our Poisson model adds value over simple Elo pricing.
    print(f"\n--- Simulated Betting (EV threshold: {SIM_EV_THRESHOLD}%) ---")
    print("  (Market odds simulated from Elo win expectancy + vig)")
    bankroll = 1000.0
    starting = bankroll
    bets_made = 0
    bets_won = 0
    total_staked = 0.0

    for r in results:
        # Simulate market odds from Elo-based probabilities
        elo_h = r.get("elo_home", ELO_INIT)
        elo_a = r.get("elo_away", ELO_INIT)
        # Simple Elo win expectancy (no draw)
        e_h = 1.0 / (1.0 + 10 ** ((elo_a - elo_h) / 400.0))
        e_a = 1.0 - e_h
        # Allocate draw probability based on closeness
        draw_base = 0.26
        p_market = {
            "home": e_h * (1.0 - draw_base),
            "draw": draw_base,
            "away": e_a * (1.0 - draw_base),
        }

        for side, p_model in [("home", r["p_home"]), ("draw", r["p_draw"]), ("away", r["p_away"])]:
            p_mkt = p_market[side]
            if p_mkt <= 0.01:
                continue
            # Bookmaker odds = fair odds with margin
            odds = (1.0 / p_mkt) * (1.0 - BOOK_MARGIN)
            ev = (p_model * odds - 1.0) * 100.0

            if ev >= SIM_EV_THRESHOLD:
                stake = min(bankroll * 0.02, bankroll)
                total_staked += stake
                bets_made += 1

                if r["outcome"] == side:
                    profit = stake * (odds - 1.0)
                    bankroll += profit
                    bets_won += 1
                else:
                    bankroll -= stake

    if bets_made > 0:
        roi = (bankroll - starting) / total_staked * 100.0
        print(f"  Bets placed: {bets_made}")
        print(f"  Bets won:    {bets_won} ({bets_won/bets_made:.1%})")
        print(f"  Total staked: ${total_staked:.0f}")
        print(f"  Final bankroll: ${bankroll:.0f} (started ${starting:.0f})")
        print(f"  P&L: ${bankroll - starting:+.0f}")
        print(f"  ROI: {roi:+.1f}%")
    else:
        print("  No simulated bets placed.")

    # Show some individual predictions
    print(f"\n--- Sample Predictions (last 15) ---")
    print(f"{'Date':<12} {'Match':<35} {'Score':<6} {'Pred':<6} "
          f"{'pH':>5} {'pD':>5} {'pA':>5} {'pAct':>5} {'OK?'}")
    print("-" * 95)
    for r in results[-15:]:
        ok = "Y" if r["correct"] else ""
        print(f"  {r['date']:<10} {r['home']+' v '+r['away']:<33} "
              f"{r['hg']}-{r['ag']:<4} {r['pred_outcome']:<6} "
              f"{r['p_home']:>4.0%} {r['p_draw']:>4.0%} {r['p_away']:>4.0%} "
              f"{r['p_actual']:>4.0%} {ok}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    run_backtest()
