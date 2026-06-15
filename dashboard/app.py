"""
World Cup 2026 Prediction Dashboard

Pure prediction model - match probabilities, tournament simulation,
team ratings, and scoreline analysis.

Usage:
    streamlit run dashboard/app.py
"""

import io
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import sqlite3
from model import predict_match, load_ratings, load_global_avg, MAX_GOALS

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "worldcup.db"

# ---------------------------------------------------------------------------
# Brand constants
# ---------------------------------------------------------------------------
BG = "#0E1117"
CARD = "#161A23"
ACCENT = "#3B82F6"
GREEN = "#22C55E"
RED = "#EF4444"
MUTED = "#64748B"
TEXT = "#E2E8F0"
BORDER = "#1E293B"

# ---------------------------------------------------------------------------
# SVG logo
# ---------------------------------------------------------------------------
LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="38" height="38" viewBox="0 0 100 100">
  <defs>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="2.5" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <polygon points="50,15 85,33 50,51 15,33" fill="#1a1f2b" stroke="#334155" stroke-width="1.2"/>
  <polygon points="15,33 50,51 50,85 15,67" fill="#12161f" stroke="#334155" stroke-width="1.2"/>
  <polygon points="85,33 50,51 50,85 85,67" fill="#0f1219" stroke="#334155" stroke-width="1.2"/>
  <line x1="50" y1="51" x2="50" y2="85" stroke="#3B82F6" stroke-width="2.4"
        stroke-linecap="round" filter="url(#glow)"/>
</svg>
"""

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="st-"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }}
  body {{ font-variant-numeric: tabular-nums; }}
  #MainMenu, footer, header {{visibility: hidden;}}
  .block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
  }}
  .kpi-row {{
    display: flex;
    gap: 14px;
    margin-bottom: 22px;
  }}
  .kpi-card {{
    flex: 1;
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 16px 20px;
    min-width: 0;
  }}
  .kpi-label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 4px;
  }}
  .kpi-value {{
    font-size: 26px;
    font-weight: 700;
    color: {TEXT};
    font-variant-numeric: tabular-nums;
    line-height: 1.15;
  }}
  .kpi-value.green {{ color: {GREEN}; }}
  .kpi-value.red   {{ color: {RED}; }}
  .kpi-value.accent {{ color: {ACCENT}; }}
  .header-block {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 6px;
  }}
  .header-block svg {{ flex-shrink: 0; }}
  .header-titles {{
    display: flex;
    flex-direction: column;
    gap: 0;
  }}
  .header-title {{
    font-size: 24px;
    font-weight: 700;
    color: {TEXT};
    line-height: 1.2;
  }}
  .header-sub {{
    font-size: 12px;
    color: {MUTED};
    margin-top: 2px;
  }}
  section[data-testid="stSidebar"] h2 {{
    font-size: 14px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 8px;
  }}
  button[data-baseweb="tab"] {{
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
  }}
  .stDataFrame table {{
    font-variant-numeric: tabular-nums;
    font-size: 13px;
  }}
</style>
"""


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def kpi_card(label: str, value: str, css_class: str = "") -> str:
    cls = f' {css_class}' if css_class else ''
    return f"""<div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value{cls}">{value}</div>
    </div>"""


# ===================================================================
# Page setup
# ===================================================================
st.set_page_config(page_title="WC 2026 Predictor", page_icon=None, layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

conn = get_db()
ratings = load_ratings(conn)
global_avg = load_global_avg(conn)

# Get WC teams list
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
now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
st.markdown(f"""
<div class="header-block">
    {LOGO_SVG}
    <div class="header-titles">
        <div class="header-title">World Cup 2026 Predictor</div>
        <div class="header-sub">Poisson + Dixon-Coles model with Elo ratings, tournament weighting, and form
            &nbsp;&middot;&nbsp; {len(teams_list)} teams &nbsp;&middot;&nbsp; Updated {now_str}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===================================================================
# Tabs
# ===================================================================
tab_predictor, tab_fixtures, tab_sim, tab_ratings = st.tabs(
    ["MATCH PREDICTOR", "FIXTURES", "TOURNAMENT SIM", "RATINGS"]
)


# -------------------------------------------------------------------
# TAB 1: Match Predictor
# -------------------------------------------------------------------
with tab_predictor:
    c1, c2, c3 = st.columns([2, 2, 1])
    home_pick = c1.selectbox("Home team", teams_list, index=0, key="pred_home")
    away_idx = min(1, len(teams_list) - 1)
    away_pick = c2.selectbox("Away team", teams_list, index=away_idx, key="pred_away")
    neutral = c3.checkbox("Neutral venue", value=True, key="pred_neutral")

    if home_pick != away_pick:
        pred = predict_match(home_pick, away_pick, neutral=neutral,
                             ratings=ratings, global_avg=global_avg, conn=conn)

        # Main probabilities
        st.markdown(f"""<div class="kpi-row">
            {kpi_card(f"{home_pick} Win", f"{pred['home_win']:.1%}")}
            {kpi_card("Draw", f"{pred['draw']:.1%}")}
            {kpi_card(f"{away_pick} Win", f"{pred['away_win']:.1%}")}
            {kpi_card("xG " + home_pick, f"{pred['expected_home']:.2f}", "accent")}
            {kpi_card("xG " + away_pick, f"{pred['expected_away']:.2f}", "accent")}
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="kpi-row">
            {kpi_card("Over 1.5", f"{pred['over_1_5']:.1%}")}
            {kpi_card("Over 2.5", f"{pred['over_2_5']:.1%}")}
            {kpi_card("Over 3.5", f"{pred['over_3_5']:.1%}")}
            {kpi_card("BTTS", f"{pred['btts']:.1%}")}
            {kpi_card("DC 1X", f"{pred['dc_1x']:.1%}")}
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div style="font-size:12px; color:{MUTED}; margin-bottom:12px;">
            Form: {home_pick} {ratings.get(home_pick, {}).get('form', 1.0):.2f},
            {away_pick} {ratings.get(away_pick, {}).get('form', 1.0):.2f}
            &nbsp;&middot;&nbsp;
            Elo: {home_pick} {ratings.get(home_pick, {}).get('elo', 1500):.0f},
            {away_pick} {ratings.get(away_pick, {}).get('elo', 1500):.0f}
        </div>""", unsafe_allow_html=True)

        # Scoreline heatmap
        st.markdown(f'<div style="font-size:13px; font-weight:600; color:{TEXT}; '
                    f'margin:12px 0 6px;">Scoreline Probability Heatmap</div>',
                    unsafe_allow_html=True)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grid = pred["scoreline_grid"]
        display_max = 6
        grid_display = grid[:display_max, :display_max]

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#161A23')

        im = ax.imshow(grid_display, cmap='Blues', aspect='auto', origin='lower')
        for i in range(display_max):
            for j in range(display_max):
                val = grid_display[i, j]
                color = 'white' if val > 0.06 else '#E2E8F0'
                ax.text(j, i, f"{val:.1%}", ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold' if val > 0.06 else 'normal')

        ax.set_xticks(range(display_max))
        ax.set_yticks(range(display_max))
        ax.set_xlabel(f"{away_pick} goals", fontsize=11, color='#E2E8F0')
        ax.set_ylabel(f"{home_pick} goals", fontsize=11, color='#E2E8F0')
        ax.tick_params(colors='#E2E8F0')
        plt.colorbar(im, ax=ax, label='Probability', shrink=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Most likely scorelines + Asian handicap side by side
        col_cs, col_ah = st.columns(2)

        with col_cs:
            st.markdown(f'<div style="font-size:13px; font-weight:600; color:{TEXT}; '
                        f'margin:12px 0 6px;">Most Likely Scorelines</div>',
                        unsafe_allow_html=True)
            cs_rows = []
            for hg, ag, p in pred["correct_scores"][:10]:
                cs_rows.append({"Score": f"{hg}-{ag}", "Probability": f"{p:.1%}"})
            st.dataframe(pd.DataFrame(cs_rows), use_container_width=True, hide_index=True)

        with col_ah:
            st.markdown(f'<div style="font-size:13px; font-weight:600; color:{TEXT}; '
                        f'margin:12px 0 6px;">Asian Handicap Lines</div>',
                        unsafe_allow_html=True)
            ah_rows = []
            for line in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
                ah = pred["asian_handicap"][line]
                ah_rows.append({
                    "Line": f"{home_pick} {line:+.1f}",
                    f"P({home_pick})": f"{ah['home']:.1%}",
                    "Push": f"{ah['push']:.1%}",
                    f"P({away_pick})": f"{ah['away']:.1%}",
                })
            st.dataframe(pd.DataFrame(ah_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("Select two different teams.")


# -------------------------------------------------------------------
# TAB 2: Fixtures with predictions
# -------------------------------------------------------------------
with tab_fixtures:
    st.markdown(f'<div style="font-size:14px; font-weight:600; color:{TEXT}; '
                f'margin-bottom:8px;">2026 World Cup Fixtures</div>',
                unsafe_allow_html=True)

    fixture_rows = pd.read_sql("""
        SELECT date, round, group_name, home_team, away_team,
               home_goals, away_goals
        FROM matches
        WHERE tournament_year = 2026
        ORDER BY date, home_team
    """, conn)

    display_rows = []
    for _, row in fixture_rows.iterrows():
        home, away = row["home_team"], row["away_team"]
        played = row["home_goals"] is not None

        entry = {
            "Date": row["date"] or "",
            "Round": row["group_name"] or row["round"] or "",
            "Home": home,
            "Away": away,
            "Score": f"{int(row['home_goals'])}-{int(row['away_goals'])}" if played else "-",
        }

        try:
            pred = predict_match(home, away, neutral=True, ratings=ratings,
                                 global_avg=global_avg, conn=conn)
            entry["P(H)"] = f"{pred['home_win']:.0%}"
            entry["P(D)"] = f"{pred['draw']:.0%}"
            entry["P(A)"] = f"{pred['away_win']:.0%}"
            entry["xG"] = f"{pred['expected_home']:.1f}-{pred['expected_away']:.1f}"
            entry["O2.5"] = f"{pred['over_2_5']:.0%}"
        except ValueError:
            entry["P(H)"] = "-"
            entry["P(D)"] = "-"
            entry["P(A)"] = "-"
            entry["xG"] = "-"
            entry["O2.5"] = "-"

        display_rows.append(entry)

    if display_rows:
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True,
                      height=600)


# -------------------------------------------------------------------
# TAB 3: Tournament Simulation
# -------------------------------------------------------------------
with tab_sim:
    st.markdown(f"""<div style="font-size:14px; font-weight:600; color:{TEXT};
                margin-bottom:6px;">Monte Carlo Tournament Simulation</div>""",
                unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px; color:{MUTED}; margin-bottom:12px;">'
                'Simulates the rest of the bracket N times using the Poisson model. '
                'Shows each team\'s probability of reaching each round.</div>',
                unsafe_allow_html=True)

    sim_count = st.selectbox("Number of simulations", [1000, 5000, 10000, 25000], index=2)

    if st.button("Run Simulation", type="primary"):
        with st.spinner(f"Running {sim_count:,} simulations..."):
            from simulate import (
                get_db as sim_db, load_groups, load_played_group_results,
                load_unplayed_group_matches, simulate_group, select_best_thirds,
                resolve_bracket_slot, simulate_match,
                R32_BRACKET, R16_PAIRS, QF_PAIRS, SF_PAIRS, ROUNDS,
            )
            from collections import defaultdict

            sim_conn = sim_db()
            sim_ratings = load_ratings(sim_conn)
            sim_global_avg = load_global_avg(sim_conn)
            groups = load_groups(sim_conn)
            played = load_played_group_results(sim_conn)
            unplayed = load_unplayed_group_matches(sim_conn)

            all_teams = set()
            for teams in groups.values():
                all_teams.update(teams)

            round_counts = {team: defaultdict(int) for team in all_teams}
            progress = st.progress(0)

            for sim in range(sim_count):
                if (sim + 1) % max(1, sim_count // 100) == 0:
                    progress.progress((sim + 1) / sim_count)

                group_standings = {}
                for g, teams in groups.items():
                    standing = simulate_group(g, teams, played, unplayed,
                                             sim_ratings, sim_global_avg)
                    group_standings[g] = standing

                for team in all_teams:
                    round_counts[team]["Group"] += 1

                best_thirds = select_best_thirds(group_standings)
                used_thirds = set()
                r32_teams = []
                for home_slot, away_slot in R32_BRACKET:
                    h = resolve_bracket_slot(home_slot, group_standings, best_thirds, used_thirds)
                    a = resolve_bracket_slot(away_slot, group_standings, best_thirds, used_thirds)
                    if h is None or a is None:
                        r32_teams.append((h or a, h, a))
                        continue
                    r32_teams.append((h, a))

                for m in r32_teams:
                    if len(m) == 2:
                        round_counts[m[0]]["R32"] += 1
                        round_counts[m[1]]["R32"] += 1

                r32_winners = []
                for m in r32_teams:
                    if len(m) == 2:
                        try:
                            w, _, _ = simulate_match(m[0], m[1], sim_ratings, sim_global_avg, knockout=True)
                        except (ValueError, KeyError):
                            w = m[0]
                        r32_winners.append(w)
                    else:
                        r32_winners.append(m[0])

                r16_winners = []
                for i, j in R16_PAIRS:
                    h, a = r32_winners[i], r32_winners[j]
                    round_counts[h]["R16"] += 1
                    round_counts[a]["R16"] += 1
                    try:
                        w, _, _ = simulate_match(h, a, sim_ratings, sim_global_avg, knockout=True)
                    except (ValueError, KeyError):
                        w = h
                    r16_winners.append(w)

                qf_winners = []
                for i, j in QF_PAIRS:
                    h, a = r16_winners[i], r16_winners[j]
                    round_counts[h]["QF"] += 1
                    round_counts[a]["QF"] += 1
                    try:
                        w, _, _ = simulate_match(h, a, sim_ratings, sim_global_avg, knockout=True)
                    except (ValueError, KeyError):
                        w = h
                    qf_winners.append(w)

                sf_winners = []
                for i, j in SF_PAIRS:
                    h, a = qf_winners[i], qf_winners[j]
                    round_counts[h]["SF"] += 1
                    round_counts[a]["SF"] += 1
                    try:
                        w, _, _ = simulate_match(h, a, sim_ratings, sim_global_avg, knockout=True)
                    except (ValueError, KeyError):
                        w = h
                    sf_winners.append(w)

                h, a = sf_winners[0], sf_winners[1]
                round_counts[h]["Final"] += 1
                round_counts[a]["Final"] += 1
                try:
                    champion, _, _ = simulate_match(h, a, sim_ratings, sim_global_avg, knockout=True)
                except (ValueError, KeyError):
                    champion = h
                round_counts[champion]["Winner"] += 1

            progress.empty()
            sim_conn.close()

            sim_rows = []
            for team in sorted(all_teams):
                row = {"Team": team}
                for r in ["R32", "R16", "QF", "SF", "Final", "Winner"]:
                    row[r] = round_counts[team][r] / sim_count
                sim_rows.append(row)

            sim_df = pd.DataFrame(sim_rows)
            sim_df = sim_df.sort_values("Winner", ascending=False).reset_index(drop=True)
            st.session_state["sim_df"] = sim_df

    if "sim_df" in st.session_state:
        sim_df = st.session_state["sim_df"]

        # Top 10 bar chart
        top10 = sim_df.head(10)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#161A23')
        bars = ax.barh(top10["Team"][::-1], top10["Winner"][::-1], color='#3B82F6', edgecolor='none')
        ax.set_xlabel("Win Probability", color='#E2E8F0', fontsize=11)
        ax.tick_params(colors='#E2E8F0')
        for bar, val in zip(bars, top10["Winner"][::-1]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.1%}", va='center', color='#E2E8F0', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Full table
        display_df = sim_df.copy()
        for col in ["R32", "R16", "QF", "SF", "Final", "Winner"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        total_win = sim_df["Winner"].sum()
        st.markdown(f'<div style="font-size:12px; color:{MUTED};">Win probability sum: '
                    f'{total_win:.3f}</div>', unsafe_allow_html=True)

        csv = sim_df.to_csv(index=False)
        st.download_button("Export to CSV", csv,
                           f"wc_sim_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           "text/csv")
    else:
        st.info("Click the button to run a fresh simulation.")


# -------------------------------------------------------------------
# TAB 4: Team Ratings
# -------------------------------------------------------------------
with tab_ratings:
    if not wc_teams_df.empty:
        display_ratings = wc_teams_df.copy()
        display_ratings.columns = ["Team", "Elo", "Attack", "Defense", "Form", "Matches"]
        display_ratings["Elo"] = display_ratings["Elo"].round(0).astype(int)
        display_ratings["Attack"] = display_ratings["Attack"].round(2)
        display_ratings["Defense"] = display_ratings["Defense"].round(2)
        display_ratings["Form"] = display_ratings["Form"].round(3)
        st.dataframe(display_ratings, use_container_width=True, hide_index=True)

        # Calibration chart
        st.markdown(f'<div style="font-size:14px; font-weight:600; color:{TEXT}; '
                    f'margin:16px 0 8px;">Model Calibration (backtest)</div>',
                    unsafe_allow_html=True)
        reliability_path = Path(__file__).resolve().parent.parent / "data" / "reliability.png"
        if reliability_path.exists():
            st.image(str(reliability_path), use_container_width=True)
        else:
            st.info("Run calibration.py to generate the reliability plot.")

conn.close()
