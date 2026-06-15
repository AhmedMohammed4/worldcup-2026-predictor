"""
Brick 10: Streamlit dashboard.

Shows: upcoming matches with model vs market, flagged edges with Kelly sizing,
a bet log with running PnL, and tournament simulation win probabilities.

Usage:
    streamlit run dashboard/app.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# Add src to path so we can import project modules
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import sqlite3
from model import predict_match, load_ratings, load_global_avg
from edges import get_upcoming_matches_with_odds, get_best_odds, implied_prob_no_vig, compute_ev
from sizing import kelly_fraction, kelly_stake

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "worldcup.db"


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def create_bets_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            match TEXT NOT NULL,
            market TEXT NOT NULL,
            selection TEXT NOT NULL,
            odds REAL NOT NULL,
            stake REAL NOT NULL,
            model_prob REAL,
            result TEXT DEFAULT 'pending',
            pnl REAL DEFAULT 0.0
        )
    """)
    conn.commit()


# --- Page config ---
st.set_page_config(page_title="World Cup Edge", page_icon="&#9917;", layout="wide")
st.title("World Cup 2026 - Edge Finder")

conn = get_db()
create_bets_table(conn)
ratings = load_ratings(conn)
global_avg = load_global_avg(conn)

# --- Sidebar ---
st.sidebar.header("Settings")
bankroll = st.sidebar.number_input("Bankroll ($)", value=1000.0, step=100.0)
kelly_cap = st.sidebar.slider("Kelly fraction", 0.05, 1.0, 0.25, 0.05)
min_ev = st.sidebar.slider("Min EV threshold (%)", 0.0, 20.0, 3.0, 0.5)

# --- Tabs ---
tab_edges, tab_matches, tab_bets, tab_sim, tab_ratings = st.tabs(
    ["Edges", "Upcoming Matches", "Bet Log & PnL", "Tournament Sim", "Team Ratings"]
)

# ============================================================
# TAB 1: Edges
# ============================================================
with tab_edges:
    st.header("Flagged Edges")

    matches = get_upcoming_matches_with_odds(conn)
    edges = []

    for match in matches:
        home, away = match["home_team"], match["away_team"]
        try:
            pred = predict_match(home, away, neutral=True, ratings=ratings,
                                 global_avg=global_avg, conn=conn)
        except ValueError:
            continue

        odds_data = get_best_odds(conn, home, away)
        market_implied = implied_prob_no_vig(odds_data.get("h2h_avg", {}))

        bets_check = [
            ("1X2", home, "home_win", "home"),
            ("1X2", "Draw", "draw", "draw"),
            ("1X2", away, "away_win", "away"),
        ]

        for mkt, label, pred_key, odds_key in bets_check:
            best_price = odds_data.get("h2h", {}).get(odds_key)
            if best_price is None:
                continue
            model_p = pred[pred_key]
            ev = compute_ev(model_p, best_price)
            impl_p = market_implied.get(odds_key, 0)
            kf = kelly_fraction(model_p, best_price, kelly_cap)
            stake = kelly_stake(model_p, best_price, bankroll, kelly_cap)

            if ev >= min_ev:
                edges.append({
                    "Date": match["commence_time"][:10],
                    "Match": f"{home} vs {away}",
                    "Market": mkt,
                    "Selection": label,
                    "Model": f"{model_p:.1%}",
                    "Implied": f"{impl_p:.1%}",
                    "Best Odds": f"{best_price:.2f}",
                    "EV%": f"{ev:+.1f}%",
                    "Kelly%": f"{kf:.2%}",
                    "Stake": f"${stake:.2f}",
                    "_ev_sort": ev,
                })

        # Over/Under 2.5
        over_price = odds_data.get("totals_2_5", {}).get("over")
        if over_price:
            ev_o = compute_ev(pred["over_2_5"], over_price)
            if ev_o >= min_ev:
                kf = kelly_fraction(pred["over_2_5"], over_price, kelly_cap)
                stake = kelly_stake(pred["over_2_5"], over_price, bankroll, kelly_cap)
                edges.append({
                    "Date": match["commence_time"][:10],
                    "Match": f"{home} vs {away}",
                    "Market": "O/U 2.5",
                    "Selection": "Over 2.5",
                    "Model": f"{pred['over_2_5']:.1%}",
                    "Implied": f"{1/over_price:.1%}",
                    "Best Odds": f"{over_price:.2f}",
                    "EV%": f"{ev_o:+.1f}%",
                    "Kelly%": f"{kf:.2%}",
                    "Stake": f"${stake:.2f}",
                    "_ev_sort": ev_o,
                })

        under_price = odds_data.get("totals_2_5", {}).get("under")
        if under_price:
            under_p = 1.0 - pred["over_2_5"]
            ev_u = compute_ev(under_p, under_price)
            if ev_u >= min_ev:
                kf = kelly_fraction(under_p, under_price, kelly_cap)
                stake = kelly_stake(under_p, under_price, bankroll, kelly_cap)
                edges.append({
                    "Date": match["commence_time"][:10],
                    "Match": f"{home} vs {away}",
                    "Market": "O/U 2.5",
                    "Selection": "Under 2.5",
                    "Model": f"{under_p:.1%}",
                    "Implied": f"{1/under_price:.1%}",
                    "Best Odds": f"{under_price:.2f}",
                    "EV%": f"{ev_u:+.1f}%",
                    "Kelly%": f"{kf:.2%}",
                    "Stake": f"${stake:.2f}",
                    "_ev_sort": ev_u,
                })

    if edges:
        edges.sort(key=lambda x: x["_ev_sort"], reverse=True)
        df_edges = pd.DataFrame(edges).drop(columns=["_ev_sort"])
        st.dataframe(df_edges, use_container_width=True, hide_index=True)
        st.caption(f"{len(edges)} edges found above {min_ev}% EV threshold")
    else:
        st.info("No edges found above threshold.")


# ============================================================
# TAB 2: Upcoming Matches
# ============================================================
with tab_matches:
    st.header("Upcoming Matches - Model Predictions")

    match_rows = []
    for match in matches:
        home, away = match["home_team"], match["away_team"]
        try:
            pred = predict_match(home, away, neutral=True, ratings=ratings,
                                 global_avg=global_avg, conn=conn)
        except ValueError:
            continue

        odds_data = get_best_odds(conn, home, away)

        match_rows.append({
            "Date": match["commence_time"][:10],
            "Home": home,
            "Away": away,
            "P(H)": f"{pred['home_win']:.0%}",
            "P(D)": f"{pred['draw']:.0%}",
            "P(A)": f"{pred['away_win']:.0%}",
            "xG H": f"{pred['expected_home']:.2f}",
            "xG A": f"{pred['expected_away']:.2f}",
            "O2.5": f"{pred['over_2_5']:.0%}",
            "BTTS": f"{pred['btts']:.0%}",
            "Odds H": f"{odds_data.get('h2h', {}).get('home', '-')}",
            "Odds D": f"{odds_data.get('h2h', {}).get('draw', '-')}",
            "Odds A": f"{odds_data.get('h2h', {}).get('away', '-')}",
        })

    if match_rows:
        st.dataframe(pd.DataFrame(match_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No upcoming matches with odds found.")


# ============================================================
# TAB 3: Bet Log & PnL
# ============================================================
with tab_bets:
    st.header("Bet Log")

    col_log, col_add = st.columns([3, 2])

    with col_add:
        st.subheader("Add Bet")
        with st.form("add_bet", clear_on_submit=True):
            bet_match = st.text_input("Match (e.g. Spain vs Brazil)")
            bet_market = st.selectbox("Market", ["1X2", "Over 2.5", "Under 2.5", "Other"])
            bet_selection = st.text_input("Selection (e.g. Spain, Draw)")
            bet_odds = st.number_input("Decimal Odds", min_value=1.01, value=2.00, step=0.01)
            bet_stake = st.number_input("Stake ($)", min_value=0.0, value=10.0, step=1.0)
            bet_model_prob = st.number_input("Model Prob", min_value=0.0, max_value=1.0,
                                              value=0.5, step=0.01)
            submitted = st.form_submit_button("Log Bet")

            if submitted and bet_match and bet_selection:
                now = datetime.now(timezone.utc).isoformat()
                conn.execute("""
                    INSERT INTO bets (timestamp, match, market, selection, odds, stake, model_prob)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (now, bet_match, bet_market, bet_selection,
                      bet_odds, bet_stake, bet_model_prob))
                conn.commit()
                st.success("Bet logged.")
                st.rerun()

    with col_log:
        st.subheader("All Bets")
        bets_df = pd.read_sql("SELECT * FROM bets ORDER BY timestamp DESC", conn)

        if not bets_df.empty:
            st.dataframe(bets_df, use_container_width=True, hide_index=True)

            # Settle bets
            st.subheader("Settle Bet")
            pending = bets_df[bets_df["result"] == "pending"]
            if not pending.empty:
                bet_id = st.selectbox("Bet ID to settle",
                                      pending["id"].tolist())
                result = st.selectbox("Result", ["won", "lost", "void"])
                if st.button("Settle"):
                    row = bets_df[bets_df["id"] == bet_id].iloc[0]
                    if result == "won":
                        pnl = row["stake"] * (row["odds"] - 1)
                    elif result == "lost":
                        pnl = -row["stake"]
                    else:
                        pnl = 0.0
                    conn.execute(
                        "UPDATE bets SET result=?, pnl=? WHERE id=?",
                        (result, pnl, int(bet_id))
                    )
                    conn.commit()
                    st.success(f"Bet {bet_id} settled: {result}, PnL: ${pnl:+.2f}")
                    st.rerun()

            # PnL summary
            st.subheader("PnL Summary")
            settled = bets_df[bets_df["result"] != "pending"]
            if not settled.empty:
                total_staked = settled["stake"].sum()
                total_pnl = settled["pnl"].sum()
                wins = len(settled[settled["result"] == "won"])
                losses = len(settled[settled["result"] == "lost"])
                roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Staked", f"${total_staked:.2f}")
                c2.metric("Total PnL", f"${total_pnl:+.2f}")
                c3.metric("ROI", f"{roi:+.1f}%")
                c4.metric("Record", f"{wins}W - {losses}L")
            else:
                st.info("No settled bets yet.")
        else:
            st.info("No bets logged yet. Use the form to add your first bet.")


# ============================================================
# TAB 4: Tournament Simulation
# ============================================================
with tab_sim:
    st.header("Tournament Win Probabilities")
    st.caption("Pre-computed from Monte Carlo simulation. Click button to re-run.")

    if st.button("Run Simulation (10,000 sims)", type="primary"):
        with st.spinner("Simulating..."):
            from simulate import run_simulation
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                run_simulation(n_sims=10000)
            output = f.getvalue()
            st.session_state["sim_output"] = output

    if "sim_output" in st.session_state:
        st.code(st.session_state["sim_output"], language=None)
    else:
        # Show team ratings as a proxy
        st.info("Click the button above to run a fresh simulation. "
                "Showing current team ratings below.")

    # Always show team ratings table
    ratings_df = pd.read_sql("""
        SELECT team, elo, attack, defense, matches_used
        FROM team_ratings
        WHERE team IN (
            SELECT DISTINCT home_team FROM matches WHERE tournament_year=2026
            UNION
            SELECT DISTINCT away_team FROM matches WHERE tournament_year=2026
        )
        AND team NOT GLOB '[0-9]*'
        AND team NOT GLOB 'W[0-9]*'
        AND team NOT GLOB 'L[0-9]*'
        AND team NOT LIKE '%/%'
        ORDER BY elo DESC
    """, conn)

    if not ratings_df.empty:
        ratings_df.columns = ["Team", "Elo", "Attack", "Defense", "Matches"]
        ratings_df["Elo"] = ratings_df["Elo"].round(0).astype(int)
        ratings_df["Attack"] = ratings_df["Attack"].round(2)
        ratings_df["Defense"] = ratings_df["Defense"].round(2)


# ============================================================
# TAB 5: Team Ratings
# ============================================================
with tab_ratings:
    st.header("Team Ratings (2026 Field)")

    if not ratings_df.empty:
        st.dataframe(ratings_df, use_container_width=True, hide_index=True)

        # Quick match predictor
        st.subheader("Quick Match Predictor")
        teams_list = ratings_df["Team"].tolist()
        c1, c2 = st.columns(2)
        home_pick = c1.selectbox("Home team", teams_list, index=0)
        away_pick = c2.selectbox("Away team", teams_list, index=1)
        neutral = st.checkbox("Neutral venue", value=True)

        if home_pick != away_pick:
            pred = predict_match(home_pick, away_pick, neutral=neutral,
                                 ratings=ratings, global_avg=global_avg, conn=conn)
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{home_pick} Win", f"{pred['home_win']:.1%}")
            c2.metric("Draw", f"{pred['draw']:.1%}")
            c3.metric(f"{away_pick} Win", f"{pred['away_win']:.1%}")

            c4, c5 = st.columns(2)
            c4.metric("Over 2.5", f"{pred['over_2_5']:.1%}")
            c5.metric("BTTS", f"{pred['btts']:.1%}")

            st.caption(f"Expected goals: {home_pick} {pred['expected_home']:.2f} - "
                       f"{pred['expected_away']:.2f} {away_pick}")

conn.close()
