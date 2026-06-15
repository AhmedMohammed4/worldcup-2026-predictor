"""
World Cup 2026 Prediction Dashboard

Usage:
    streamlit run dashboard/app.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import sqlite3
from model import predict_match, load_ratings, load_global_avg, MAX_GOALS

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "worldcup.db"


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ===================================================================
# Page config
# ===================================================================
st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

conn = get_db()
ratings = load_ratings(conn)
global_avg = load_global_avg(conn)

# Get WC teams
wc_teams_df = pd.read_sql("""
    SELECT team, elo, attack, defense, form, matches_used
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
teams_list = wc_teams_df["team"].tolist()


# ===================================================================
# Header
# ===================================================================
st.title("World Cup 2026 Predictor")
st.caption(
    f"Poisson + Dixon-Coles model with Elo ratings, tournament weighting & form  -  "
    f"{len(teams_list)} teams  -  "
    f"Updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
)

# ===================================================================
# Tabs
# ===================================================================
tab_predictor, tab_fixtures, tab_sim, tab_ratings = st.tabs(
    ["Match Predictor", "Fixtures", "Tournament Sim", "Ratings"]
)


# -------------------------------------------------------------------
# TAB 1: Match Predictor
# -------------------------------------------------------------------
with tab_predictor:
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        home_pick = st.selectbox("Home team", teams_list, index=0)
    with col2:
        away_pick = st.selectbox("Away team", teams_list, index=min(1, len(teams_list) - 1))
    with col3:
        st.write("")  # spacer
        neutral = st.checkbox("Neutral venue", value=True)

    if home_pick == away_pick:
        st.warning("Pick two different teams.")
    else:
        pred = predict_match(home_pick, away_pick, neutral=neutral,
                             ratings=ratings, global_avg=global_avg, conn=conn)

        # --- Result probabilities ---
        st.subheader("Match Probabilities")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home_pick} Win", f"{pred['home_win']:.1%}")
        c2.metric("Draw", f"{pred['draw']:.1%}")
        c3.metric(f"{away_pick} Win", f"{pred['away_win']:.1%}")

        # --- Expected goals ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"xG {home_pick}", f"{pred['expected_home']:.2f}")
        c2.metric(f"xG {away_pick}", f"{pred['expected_away']:.2f}")
        c3.metric("Over 2.5", f"{pred['over_2_5']:.1%}")
        c4.metric("BTTS", f"{pred['btts']:.1%}")

        # --- Extra markets ---
        with st.expander("More markets"):
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Over 1.5", f"{pred['over_1_5']:.1%}")
            m2.metric("Over 3.5", f"{pred['over_3_5']:.1%}")
            m3.metric("DC 1X", f"{pred['dc_1x']:.1%}")
            m4.metric("DC X2", f"{pred['dc_x2']:.1%}")
            m5.metric("DC 12", f"{pred['dc_12']:.1%}")
            home_form = ratings.get(home_pick, {}).get("form", 1.0)
            away_form = ratings.get(away_pick, {}).get("form", 1.0)
            m6.metric("Form gap", f"{home_form - away_form:+.3f}")

        # --- Scoreline heatmap ---
        st.subheader("Scoreline Heatmap")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grid = pred["scoreline_grid"]
        n = 6  # show 0-5 goals
        grid_show = grid[:n, :n]

        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(grid_show, cmap="YlOrRd", aspect="equal", origin="lower")
        for i in range(n):
            for j in range(n):
                v = grid_show[i, j]
                ax.text(j, i, f"{v:.0%}" if v >= 0.01 else "",
                        ha="center", va="center", fontsize=9,
                        color="white" if v > 0.08 else "black")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel(f"{away_pick} goals")
        ax.set_ylabel(f"{home_pick} goals")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Probability")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # --- Scorelines + AH side by side ---
        left, right = st.columns(2)

        with left:
            st.subheader("Most Likely Scores")
            cs = [{"Score": f"{h}-{a}", "Prob": f"{p:.1%}"}
                  for h, a, p in pred["correct_scores"][:10]]
            st.dataframe(pd.DataFrame(cs), use_container_width=True, hide_index=True)

        with right:
            st.subheader("Asian Handicap")
            ah = []
            for line in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
                d = pred["asian_handicap"][line]
                ah.append({
                    "Line": f"{home_pick} {line:+.1f}",
                    home_pick: f"{d['home']:.1%}",
                    "Push": f"{d['push']:.1%}",
                    away_pick: f"{d['away']:.1%}",
                })
            st.dataframe(pd.DataFrame(ah), use_container_width=True, hide_index=True)


# -------------------------------------------------------------------
# TAB 2: Fixtures
# -------------------------------------------------------------------
with tab_fixtures:
    st.subheader("2026 World Cup Fixtures & Predictions")

    fixtures = pd.read_sql("""
        SELECT date, round, group_name, home_team, away_team,
               home_goals, away_goals
        FROM matches
        WHERE tournament_year = 2026
        ORDER BY date, home_team
    """, conn)

    rows = []
    for _, r in fixtures.iterrows():
        home, away = r["home_team"], r["away_team"]
        played = r["home_goals"] is not None
        score = f"{int(r['home_goals'])}-{int(r['away_goals'])}" if played else ""

        try:
            p = predict_match(home, away, neutral=True, ratings=ratings,
                              global_avg=global_avg, conn=conn)
            rows.append({
                "Date": r["date"] or "",
                "Round": r["group_name"] or r["round"] or "",
                "Home": home,
                "Away": away,
                "Result": score,
                "P(H)": f"{p['home_win']:.0%}",
                "P(D)": f"{p['draw']:.0%}",
                "P(A)": f"{p['away_win']:.0%}",
                "xG": f"{p['expected_home']:.1f} - {p['expected_away']:.1f}",
            })
        except ValueError:
            rows.append({
                "Date": r["date"] or "",
                "Round": r["group_name"] or r["round"] or "",
                "Home": home,
                "Away": away,
                "Result": score,
                "P(H)": "", "P(D)": "", "P(A)": "", "xG": "",
            })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=700)


# -------------------------------------------------------------------
# TAB 3: Tournament Simulation
# -------------------------------------------------------------------
with tab_sim:
    st.subheader("Monte Carlo Tournament Simulation")
    st.caption("Simulates the bracket thousands of times to estimate each team's chances.")

    sim_count = st.selectbox("Simulations", [1000, 5000, 10000, 25000], index=2)

    if st.button("Run Simulation", type="primary"):
        with st.spinner(f"Simulating {sim_count:,} tournaments..."):
            from simulate import (
                get_db as sim_db, load_groups, load_played_group_results,
                load_unplayed_group_matches, simulate_group, select_best_thirds,
                resolve_bracket_slot, simulate_match,
                R32_BRACKET, R16_PAIRS, QF_PAIRS, SF_PAIRS,
            )

            sim_conn = sim_db()
            sim_ratings = load_ratings(sim_conn)
            sim_avg = load_global_avg(sim_conn)
            groups = load_groups(sim_conn)
            played = load_played_group_results(sim_conn)
            unplayed = load_unplayed_group_matches(sim_conn)

            all_teams = set()
            for t in groups.values():
                all_teams.update(t)

            rc = {team: defaultdict(int) for team in all_teams}
            bar = st.progress(0)

            for s in range(sim_count):
                if (s + 1) % max(1, sim_count // 50) == 0:
                    bar.progress((s + 1) / sim_count)

                gs = {}
                for g, t in groups.items():
                    gs[g] = simulate_group(g, t, played, unplayed, sim_ratings, sim_avg)

                for team in all_teams:
                    rc[team]["Group"] += 1

                bt = select_best_thirds(gs)
                ut = set()
                r32 = []
                for hs, aws in R32_BRACKET:
                    h = resolve_bracket_slot(hs, gs, bt, ut)
                    a = resolve_bracket_slot(aws, gs, bt, ut)
                    if h is None or a is None:
                        r32.append((h or a, h, a))
                    else:
                        r32.append((h, a))

                for m in r32:
                    if len(m) == 2:
                        rc[m[0]]["R32"] += 1
                        rc[m[1]]["R32"] += 1

                def sim_ko(h, a):
                    try:
                        w, _, _ = simulate_match(h, a, sim_ratings, sim_avg, knockout=True)
                        return w
                    except (ValueError, KeyError):
                        return h

                r32w = []
                for m in r32:
                    r32w.append(sim_ko(m[0], m[1]) if len(m) == 2 else m[0])

                r16w = []
                for i, j in R16_PAIRS:
                    h, a = r32w[i], r32w[j]
                    rc[h]["R16"] += 1
                    rc[a]["R16"] += 1
                    r16w.append(sim_ko(h, a))

                qfw = []
                for i, j in QF_PAIRS:
                    h, a = r16w[i], r16w[j]
                    rc[h]["QF"] += 1
                    rc[a]["QF"] += 1
                    qfw.append(sim_ko(h, a))

                sfw = []
                for i, j in SF_PAIRS:
                    h, a = qfw[i], qfw[j]
                    rc[h]["SF"] += 1
                    rc[a]["SF"] += 1
                    sfw.append(sim_ko(h, a))

                h, a = sfw[0], sfw[1]
                rc[h]["Final"] += 1
                rc[a]["Final"] += 1
                rc[sim_ko(h, a)]["Winner"] += 1

            bar.empty()
            sim_conn.close()

            rows = []
            for team in sorted(all_teams):
                row = {"Team": team}
                for r in ["R32", "R16", "QF", "SF", "Final", "Winner"]:
                    row[r] = rc[team][r] / sim_count
                rows.append(row)

            df = pd.DataFrame(rows).sort_values("Winner", ascending=False).reset_index(drop=True)
            st.session_state["sim_df"] = df

    if "sim_df" in st.session_state:
        df = st.session_state["sim_df"]

        # Top 10 chart
        st.subheader("Top 10 - Win Probability")
        top = df.head(10).set_index("Team")["Winner"].sort_values()
        st.bar_chart(top, horizontal=True)

        # Full table
        st.subheader("All Teams")
        show = df.copy()
        for c in ["R32", "R16", "QF", "SF", "Final", "Winner"]:
            show[c] = show[c].apply(lambda x: f"{x:.1%}")
        st.dataframe(show, use_container_width=True, hide_index=True, height=600)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            f"wc_sim_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
        )
    else:
        st.info("Click **Run Simulation** to generate results.")


# -------------------------------------------------------------------
# TAB 4: Ratings
# -------------------------------------------------------------------
with tab_ratings:
    st.subheader("Team Ratings - 2026 World Cup")
    st.caption("Elo from full match history. Attack/Defense/Form from recent internationals (tournament-weighted).")

    show_r = wc_teams_df.copy()
    show_r.columns = ["Team", "Elo", "Attack", "Defense", "Form", "Matches"]
    show_r["Elo"] = show_r["Elo"].round(0).astype(int)
    show_r["Attack"] = show_r["Attack"].round(2)
    show_r["Defense"] = show_r["Defense"].round(2)
    show_r["Form"] = show_r["Form"].round(3)
    st.dataframe(show_r, use_container_width=True, hide_index=True, height=600)

    st.subheader("Model Calibration")
    rel_path = Path(__file__).resolve().parent.parent / "data" / "reliability.png"
    if rel_path.exists():
        st.image(str(rel_path))
    else:
        st.info("Run `python src/calibration.py` to generate the calibration plot.")

conn.close()
