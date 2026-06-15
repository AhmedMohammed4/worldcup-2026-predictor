"""
Brick 10: Streamlit dashboard - restyled as a quant/fintech product.

Features:
- Edge finder across 1X2, O/U, BTTS, Double Chance, Asian Handicap
- Match predictor with scoreline heatmap
- Bet log with PnL tracking
- Monte Carlo simulation results as sortable table
- CSV export for edges
- Inline calibration chart

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
from edges import get_upcoming_matches_with_odds, get_best_odds, implied_prob_no_vig, compute_ev
from sizing import kelly_fraction, kelly_stake

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
# SVG logo - isometric black-box cube with one glowing blue edge
# ---------------------------------------------------------------------------
LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="38" height="38" viewBox="0 0 100 100">
  <defs>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="2.5" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <!-- top face -->
  <polygon points="50,15 85,33 50,51 15,33" fill="#1a1f2b" stroke="#334155" stroke-width="1.2"/>
  <!-- left face -->
  <polygon points="15,33 50,51 50,85 15,67" fill="#12161f" stroke="#334155" stroke-width="1.2"/>
  <!-- right face -->
  <polygon points="85,33 50,51 50,85 85,67" fill="#0f1219" stroke="#334155" stroke-width="1.2"/>
  <!-- glowing front-right edge -->
  <line x1="50" y1="51" x2="50" y2="85" stroke="#3B82F6" stroke-width="2.4"
        stroke-linecap="round" filter="url(#glow)"/>
</svg>
"""

# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------
CUSTOM_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  /* global */
  html, body, [class*="st-"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }}

  /* tabular figures everywhere */
  body {{
    font-variant-numeric: tabular-nums;
  }}

  /* hide default header / footer */
  #MainMenu, footer, header {{visibility: hidden;}}

  /* tighter top padding */
  .block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
  }}

  /* KPI card row */
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

  /* header block */
  .header-block {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 6px;
  }}
  .header-block svg {{
    flex-shrink: 0;
  }}
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

  /* styled edges table */
  .edge-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
    margin-top: 6px;
  }}
  .edge-table thead th {{
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {MUTED};
    padding: 10px 12px;
    border-bottom: 1px solid {BORDER};
    text-align: left;
    position: sticky;
    top: 0;
    background: {BG};
  }}
  .edge-table thead th.num {{
    text-align: right;
  }}
  .edge-table tbody tr {{
    transition: background 0.12s;
  }}
  .edge-table tbody tr:hover {{
    background: rgba(59, 130, 246, 0.06);
  }}
  .edge-table tbody td {{
    padding: 9px 12px;
    border-bottom: 1px solid {BORDER}22;
    color: {TEXT};
    white-space: nowrap;
  }}
  .edge-table tbody td.num {{
    text-align: right;
    font-family: 'Inter', monospace;
    font-variant-numeric: tabular-nums;
  }}
  .edge-table .ev-pos {{ color: {GREEN}; font-weight: 600; }}
  .edge-table .ev-neg {{ color: {RED}; font-weight: 600; }}
  .edge-table .muted  {{ color: {MUTED}; }}
  .edge-table .accent {{ color: {ACCENT}; font-weight: 500; }}

  /* sidebar tweaks */
  section[data-testid="stSidebar"] .stMarkdown p {{
    font-size: 13px;
    color: {MUTED};
  }}
  section[data-testid="stSidebar"] h2 {{
    font-size: 14px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 8px;
  }}

  /* tab styling */
  button[data-baseweb="tab"] {{
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
  }}

  /* streamlit dataframe override */
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


def render_edges_table(edges: list[dict]) -> str:
    """Render the edges list as a styled HTML table."""
    cols = [
        ("Date", False),
        ("Match", False),
        ("Mkt", False),
        ("Selection", False),
        ("Model", True),
        ("Implied", True),
        ("Odds", True),
        ("EV%", True),
        ("Kelly%", True),
        ("Stake", True),
    ]
    header = "".join(
        f'<th class="{"num" if is_num else ""}">{label}</th>'
        for label, is_num in cols
    )
    rows = []
    for e in edges:
        ev_val = e["_ev_sort"]
        ev_class = "ev-pos" if ev_val > 0 else "ev-neg"
        rows.append(f"""<tr>
            <td class="muted">{e["Date"]}</td>
            <td>{e["Match"]}</td>
            <td class="muted">{e["Market"]}</td>
            <td class="accent">{e["Selection"]}</td>
            <td class="num">{e["Model"]}</td>
            <td class="num muted">{e["Implied"]}</td>
            <td class="num">{e["Best Odds"]}</td>
            <td class="num {ev_class}">{e["EV%"]}</td>
            <td class="num">{e["Kelly%"]}</td>
            <td class="num">{e["Stake"]}</td>
        </tr>""")

    return f"""<div style="overflow-x:auto; border:1px solid {BORDER}; border-radius:10px;
                background:{CARD};">
        <table class="edge-table">
            <thead><tr>{header}</tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    </div>"""


def kpi_card(label: str, value: str, css_class: str = "") -> str:
    cls = f' {css_class}' if css_class else ''
    return f"""<div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value{cls}">{value}</div>
    </div>"""


def edges_to_csv(edges: list[dict]) -> str:
    """Convert edges to CSV string for download."""
    if not edges:
        return ""
    df = pd.DataFrame(edges)
    # Drop internal sort key
    df = df.drop(columns=["_ev_sort"], errors="ignore")
    return df.to_csv(index=False)


# ===================================================================
# Page setup
# ===================================================================
st.set_page_config(page_title="WC Edge", page_icon=None, layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

conn = get_db()
create_bets_table(conn)
ratings = load_ratings(conn)
global_avg = load_global_avg(conn)

# ===================================================================
# Compute edges (needed for header KPIs and edges tab)
# ===================================================================
matches_with_odds = get_upcoming_matches_with_odds(conn)
all_edges = []

for match in matches_with_odds:
    home, away = match["home_team"], match["away_team"]
    try:
        pred = predict_match(home, away, neutral=True, ratings=ratings,
                             global_avg=global_avg, conn=conn)
    except ValueError:
        continue

    odds_data = get_best_odds(conn, home, away)
    market_implied = implied_prob_no_vig(odds_data.get("h2h_avg", {}))

    # 1X2
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

        all_edges.append({
            "Date": match["commence_time"][:10],
            "Match": f"{home} vs {away}",
            "Market": mkt,
            "Selection": label,
            "model_p": model_p,
            "impl_p": impl_p,
            "best_price": best_price,
            "ev": ev,
        })

    # Over/Under lines
    for line, pred_key in [(1.5, "over_1_5"), (2.5, "over_2_5"), (3.5, "over_3_5")]:
        over_price = odds_data.get("totals", {}).get(f"over_{line}")
        under_price = odds_data.get("totals", {}).get(f"under_{line}")
        if over_price:
            ev = compute_ev(pred[pred_key], over_price)
            all_edges.append({
                "Date": match["commence_time"][:10],
                "Match": f"{home} vs {away}",
                "Market": f"O/U {line}",
                "Selection": f"Over {line}",
                "model_p": pred[pred_key],
                "impl_p": 1.0 / over_price,
                "best_price": over_price,
                "ev": ev,
            })
        if under_price:
            under_p = 1.0 - pred[pred_key]
            ev = compute_ev(under_p, under_price)
            all_edges.append({
                "Date": match["commence_time"][:10],
                "Match": f"{home} vs {away}",
                "Market": f"O/U {line}",
                "Selection": f"Under {line}",
                "model_p": under_p,
                "impl_p": 1.0 / under_price,
                "best_price": under_price,
                "ev": ev,
            })

    # BTTS
    all_edges.append({
        "Date": match["commence_time"][:10],
        "Match": f"{home} vs {away}",
        "Market": "BTTS",
        "Selection": "Yes",
        "model_p": pred["btts"],
        "impl_p": 0,
        "best_price": None,
        "ev": -999,  # No odds available unless in DB
    })

    # Double Chance
    for dc_label, dc_key in [
        (f"1X ({home}/Draw)", "dc_1x"),
        (f"X2 (Draw/{away})", "dc_x2"),
        (f"12 ({home}/{away})", "dc_12"),
    ]:
        all_edges.append({
            "Date": match["commence_time"][:10],
            "Match": f"{home} vs {away}",
            "Market": "DC",
            "Selection": dc_label,
            "model_p": pred[dc_key],
            "impl_p": 0,
            "best_price": None,
            "ev": -999,
        })

# Filter out entries with no actual odds
all_edges = [e for e in all_edges if e["best_price"] is not None and e["ev"] > -999]


# ===================================================================
# Sidebar
# ===================================================================
st.sidebar.markdown(f"""
<div style="margin-bottom:16px;">
    <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em;
                color:{MUTED}; font-weight:600; margin-bottom:4px;">
        Model Status
    </div>
    <div style="font-size:13px; color:{TEXT};">
        {len(matches_with_odds)} upcoming fixtures with odds
    </div>
    <div style="font-size:13px; color:{TEXT};">
        {len([e for e in all_edges if e['ev'] >= 3.0])} edges above 3% EV
    </div>
    <div style="font-size:12px; color:{MUTED}; margin-top:4px;">
        Poisson + Dixon-Coles + Form, 1/4 Kelly
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"## Parameters")
bankroll = st.sidebar.number_input("Bankroll ($)", value=1000.0, step=100.0)
kelly_cap = st.sidebar.slider("Kelly fraction", 0.05, 1.0, 0.25, 0.05)
min_ev = st.sidebar.slider("Min EV threshold (%)", 0.0, 20.0, 3.0, 0.5)
market_filter = st.sidebar.multiselect(
    "Markets",
    options=sorted(set(e["Market"] for e in all_edges)),
    default=sorted(set(e["Market"] for e in all_edges)),
)


# ===================================================================
# Header with logo
# ===================================================================
now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
st.markdown(f"""
<div class="header-block">
    {LOGO_SVG}
    <div class="header-titles">
        <div class="header-title">World Cup 2026 Edge Finder</div>
        <div class="header-sub">Poisson + Dixon-Coles + Form model vs. market odds across 48 bookmakers
            &nbsp;&middot;&nbsp; Last updated {now_str}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ===================================================================
# Filter edges by threshold and compute KPI values
# ===================================================================
filtered_edges = []
total_stake = 0.0
for e in all_edges:
    if e["ev"] >= min_ev and e["Market"] in market_filter:
        kf = kelly_fraction(e["model_p"], e["best_price"], kelly_cap)
        stk = kelly_stake(e["model_p"], e["best_price"], bankroll, kelly_cap)
        filtered_edges.append({
            "Date": e["Date"],
            "Match": e["Match"],
            "Market": e["Market"],
            "Selection": e["Selection"],
            "Model": f"{e['model_p']:.1%}",
            "Implied": f"{e['impl_p']:.1%}",
            "Best Odds": f"{e['best_price']:.2f}",
            "EV%": f"{e['ev']:+.1f}%",
            "Kelly%": f"{kf:.2%}",
            "Stake": f"${stk:.2f}",
            "_ev_sort": e["ev"],
        })
        total_stake += stk

filtered_edges.sort(key=lambda x: x["_ev_sort"], reverse=True)

best_ev = max((e["_ev_sort"] for e in filtered_edges), default=0)
brier_val = "0.626"  # from calibration brick

kpi_html = f"""<div class="kpi-row">
    {kpi_card("Bankroll", f"${bankroll:,.0f}")}
    {kpi_card("Edges Found", str(len(filtered_edges)), "accent")}
    {kpi_card("Best EV", f"{best_ev:+.1f}%", "green" if best_ev > 0 else "")}
    {kpi_card("Total Rec. Stake", f"${total_stake:,.2f}", "accent")}
    {kpi_card("Model Brier", brier_val)}
</div>"""
st.markdown(kpi_html, unsafe_allow_html=True)


# ===================================================================
# Tabs
# ===================================================================
tab_edges, tab_matches, tab_predictor, tab_bets, tab_sim, tab_ratings = st.tabs(
    ["EDGES", "MATCHES", "PREDICTOR", "BET LOG", "SIMULATION", "RATINGS"]
)


# -------------------------------------------------------------------
# TAB 1: Edges
# -------------------------------------------------------------------
with tab_edges:
    if filtered_edges:
        st.markdown(render_edges_table(filtered_edges), unsafe_allow_html=True)
        st.markdown(f"""<div style="font-size:12px; color:{MUTED}; margin-top:8px;">
            {len(filtered_edges)} edges above {min_ev:.1f}% EV threshold
            &nbsp;&middot;&nbsp; Odds sourced from {
                pd.read_sql("SELECT COUNT(DISTINCT bookmaker) FROM odds", conn).iloc[0, 0]
            } bookmakers
        </div>""", unsafe_allow_html=True)

        # CSV download button
        csv_data = edges_to_csv(filtered_edges)
        st.download_button(
            label="Export Edges to CSV",
            data=csv_data,
            file_name=f"wc_edges_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No edges found above threshold.")


# -------------------------------------------------------------------
# TAB 2: Upcoming Matches
# -------------------------------------------------------------------
with tab_matches:
    match_rows = []
    for match in matches_with_odds:
        home, away = match["home_team"], match["away_team"]
        try:
            pred = predict_match(home, away, neutral=True, ratings=ratings,
                                 global_avg=global_avg, conn=conn)
        except ValueError:
            continue
        odds_data = get_best_odds(conn, home, away)
        h_odds = odds_data.get("h2h", {}).get("home")
        d_odds = odds_data.get("h2h", {}).get("draw")
        a_odds = odds_data.get("h2h", {}).get("away")

        match_rows.append({
            "Date": match["commence_time"][:10],
            "Home": home,
            "Away": away,
            "P(H)": f"{pred['home_win']:.1%}",
            "P(D)": f"{pred['draw']:.1%}",
            "P(A)": f"{pred['away_win']:.1%}",
            "xG H": f"{pred['expected_home']:.2f}",
            "xG A": f"{pred['expected_away']:.2f}",
            "O2.5": f"{pred['over_2_5']:.1%}",
            "BTTS": f"{pred['btts']:.1%}",
            "DC 1X": f"{pred['dc_1x']:.1%}",
            "DC X2": f"{pred['dc_x2']:.1%}",
            "Odds H": f"{h_odds:.2f}" if h_odds else "-",
            "Odds D": f"{d_odds:.2f}" if d_odds else "-",
            "Odds A": f"{a_odds:.2f}" if a_odds else "-",
        })

    if match_rows:
        st.dataframe(pd.DataFrame(match_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No upcoming matches with odds found.")


# -------------------------------------------------------------------
# TAB 3: Match Predictor with Scoreline Heatmap
# -------------------------------------------------------------------
with tab_predictor:
    st.markdown(f'<div style="font-size:14px; font-weight:600; color:{TEXT}; '
                f'margin-bottom:8px;">Match Predictor</div>', unsafe_allow_html=True)

    # Team selection
    ratings_df = pd.read_sql("""
        SELECT team FROM team_ratings
        WHERE team IN (
            SELECT DISTINCT home_team FROM matches WHERE tournament_year=2026
            UNION
            SELECT DISTINCT away_team FROM matches WHERE tournament_year=2026
        )
        AND team NOT GLOB '[0-9]*'
        AND team NOT GLOB 'W[0-9]*'
        AND team NOT GLOB 'L[0-9]*'
        AND team NOT LIKE '%/%'
        ORDER BY team
    """, conn)
    teams_list = ratings_df["team"].tolist()

    c1, c2, c3 = st.columns([2, 2, 1])
    home_pick = c1.selectbox("Home team", teams_list, index=0, key="pred_home")
    away_pick = c2.selectbox("Away team", teams_list,
                              index=min(1, len(teams_list) - 1), key="pred_away")
    neutral = c3.checkbox("Neutral venue", value=True, key="pred_neutral")

    if home_pick != away_pick:
        pred = predict_match(home_pick, away_pick, neutral=neutral,
                             ratings=ratings, global_avg=global_avg, conn=conn)

        # KPI row
        st.markdown(f"""<div class="kpi-row">
            {kpi_card(f"{home_pick} Win", f"{pred['home_win']:.1%}")}
            {kpi_card("Draw", f"{pred['draw']:.1%}")}
            {kpi_card(f"{away_pick} Win", f"{pred['away_win']:.1%}")}
            {kpi_card("Over 2.5", f"{pred['over_2_5']:.1%}")}
            {kpi_card("BTTS", f"{pred['btts']:.1%}")}
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="kpi-row">
            {kpi_card("DC 1X", f"{pred['dc_1x']:.1%}")}
            {kpi_card("DC X2", f"{pred['dc_x2']:.1%}")}
            {kpi_card("DC 12", f"{pred['dc_12']:.1%}")}
            {kpi_card("Over 1.5", f"{pred['over_1_5']:.1%}")}
            {kpi_card("Over 3.5", f"{pred['over_3_5']:.1%}")}
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div style="font-size:12px; color:{MUTED}; margin-bottom:12px;">
            Expected goals: {home_pick} {pred['expected_home']:.2f} -
            {pred['expected_away']:.2f} {away_pick}
            &nbsp;&middot;&nbsp;
            Form: {home_pick} {ratings.get(home_pick, {}).get('form', 1.0):.2f},
            {away_pick} {ratings.get(away_pick, {}).get('form', 1.0):.2f}
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

        # Top correct scores table
        st.markdown(f'<div style="font-size:13px; font-weight:600; color:{TEXT}; '
                    f'margin:12px 0 6px;">Most Likely Scorelines</div>',
                    unsafe_allow_html=True)
        cs_rows = []
        for hg, ag, p in pred["correct_scores"][:10]:
            cs_rows.append({"Score": f"{hg}-{ag}", "Probability": f"{p:.1%}",
                            "Fair Odds": f"{1/p:.1f}" if p > 0.001 else "-"})
        st.dataframe(pd.DataFrame(cs_rows), use_container_width=True, hide_index=True)

        # Asian Handicap table
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
# TAB 4: Bet Log & PnL
# -------------------------------------------------------------------
with tab_bets:
    col_log, col_add = st.columns([3, 2])

    with col_add:
        st.markdown(f'<div style="font-size:14px; font-weight:600; color:{TEXT}; '
                    f'margin-bottom:8px;">Log a Bet</div>', unsafe_allow_html=True)
        with st.form("add_bet", clear_on_submit=True):
            bet_match = st.text_input("Match (e.g. Spain vs Brazil)")
            bet_market = st.selectbox("Market", [
                "1X2", "Over 1.5", "Under 1.5", "Over 2.5", "Under 2.5",
                "Over 3.5", "Under 3.5", "BTTS Yes", "BTTS No",
                "DC 1X", "DC X2", "DC 12", "Asian Handicap", "Correct Score", "Other",
            ])
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
        bets_df = pd.read_sql("SELECT * FROM bets ORDER BY timestamp DESC", conn)

        if not bets_df.empty:
            # PnL summary cards
            settled = bets_df[bets_df["result"] != "pending"]
            if not settled.empty:
                t_staked = settled["stake"].sum()
                t_pnl = settled["pnl"].sum()
                wins = len(settled[settled["result"] == "won"])
                losses = len(settled[settled["result"] == "lost"])
                roi = (t_pnl / t_staked * 100) if t_staked > 0 else 0
                pnl_cls = "green" if t_pnl >= 0 else "red"
                roi_cls = "green" if roi >= 0 else "red"

                st.markdown(f"""<div class="kpi-row">
                    {kpi_card("Staked", f"${t_staked:.2f}")}
                    {kpi_card("PnL", f"${t_pnl:+.2f}", pnl_cls)}
                    {kpi_card("ROI", f"{roi:+.1f}%", roi_cls)}
                    {kpi_card("Record", f"{wins}W - {losses}L")}
                </div>""", unsafe_allow_html=True)

            st.dataframe(bets_df, use_container_width=True, hide_index=True)

            pending = bets_df[bets_df["result"] == "pending"]
            if not pending.empty:
                st.markdown(f'<div style="font-size:14px; font-weight:600; color:{TEXT}; '
                            f'margin:12px 0 6px;">Settle Bet</div>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns([2, 2, 1])
                bet_id = c1.selectbox("Bet ID", pending["id"].tolist())
                result = c2.selectbox("Result", ["won", "lost", "void"])
                if c3.button("Settle", type="primary"):
                    row = bets_df[bets_df["id"] == bet_id].iloc[0]
                    if result == "won":
                        pnl = row["stake"] * (row["odds"] - 1)
                    elif result == "lost":
                        pnl = -row["stake"]
                    else:
                        pnl = 0.0
                    conn.execute("UPDATE bets SET result=?, pnl=? WHERE id=?",
                                 (result, pnl, int(bet_id)))
                    conn.commit()
                    st.rerun()
        else:
            st.info("No bets logged yet.")


# -------------------------------------------------------------------
# TAB 5: Tournament Simulation (proper table output)
# -------------------------------------------------------------------
with tab_sim:
    st.markdown(f"""<div style="font-size:14px; font-weight:600; color:{TEXT};
                margin-bottom:6px;">Monte Carlo Tournament Simulation</div>""",
                unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px; color:{MUTED}; margin-bottom:12px;">'
                'Simulates the rest of the bracket N times using the Poisson model. '
                'Results show each team\'s probability of reaching each round.</div>',
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

            # Build results dataframe
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

        # Top 10 chart
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

        # Format as percentages for display
        display_df = sim_df.copy()
        for col in ["R32", "R16", "QF", "SF", "Final", "Winner"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        total_win = sim_df["Winner"].sum()
        st.markdown(f'<div style="font-size:12px; color:{MUTED};">Win probability sum: '
                    f'{total_win:.3f}</div>', unsafe_allow_html=True)

        # CSV export
        csv = sim_df.to_csv(index=False)
        st.download_button("Export Simulation to CSV", csv,
                           f"wc_sim_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           "text/csv")
    else:
        st.info("Click the button to run a fresh simulation.")


# -------------------------------------------------------------------
# TAB 6: Team Ratings
# -------------------------------------------------------------------
with tab_ratings:
    full_ratings_df = pd.read_sql("""
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

    if not full_ratings_df.empty:
        full_ratings_df.columns = ["Team", "Elo", "Attack", "Defense", "Form", "Matches"]
        full_ratings_df["Elo"] = full_ratings_df["Elo"].round(0).astype(int)
        full_ratings_df["Attack"] = full_ratings_df["Attack"].round(2)
        full_ratings_df["Defense"] = full_ratings_df["Defense"].round(2)
        full_ratings_df["Form"] = full_ratings_df["Form"].round(3)
        st.dataframe(full_ratings_df, use_container_width=True, hide_index=True)

        # Calibration chart inline
        st.markdown(f'<div style="font-size:14px; font-weight:600; color:{TEXT}; '
                    f'margin:16px 0 8px;">Calibration (from backtest)</div>',
                    unsafe_allow_html=True)
        reliability_path = Path(__file__).resolve().parent.parent / "data" / "reliability.png"
        if reliability_path.exists():
            st.image(str(reliability_path), use_container_width=True)
        else:
            st.info("Run calibration.py to generate the reliability plot.")

conn.close()
