import streamlit as st
import pandas as pd
import os
import sys
import json
import re
import traceback
from PIL import Image

st.set_page_config(
    page_title="ChessLens",
    page_icon="♟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton > button:hover { background-color: #388E3C; }
    .drill-box {
        background-color: #1a1f2e;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 0.4rem 0;
        border: 1px solid #2d3250;
    }
    .settings-box {
        background-color: #1a1f2e;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.3rem 0;
        border: 1px solid #2d3250;
        font-size: 13px;
        color: #9e9e9e;
    }
    .mode-fast { border-left: 4px solid #4CAF50; }
    .mode-deep { border-left: 4px solid #7C4DFF; }
</style>
""", unsafe_allow_html=True)


# ── Config ────────────────────────────────────────────────────────────────────────

CONFIG_PATH = "chesslens_config.json"

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {"max_games": 100, "eval_time": 0.02, "mode": "fast", "lichess_token": "", "num_workers": 5}

def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f)

config = load_config()


# ── File paths ────────────────────────────────────────────────────────────────────

def user_dir(u):         return f"data/analytics"
def moves_path(u):       return f"data/processed/moves.parquet"
def evaluated_path(u):   return f"data/processed/moves_evaluated.parquet"
def features_path(u):    return f"data/processed/moves_features.parquet"
def categorized_path(u): return f"data/processed/moves_categorized.parquet"
def report_path(u):      return f"data/analytics/{u}_coaching_report.txt"
def chart_path(u):       return f"data/analytics/{u}_analysis.png"

def data_exists(u):
    return u != "" and os.path.exists(categorized_path(u)) and os.path.exists(report_path(u))

def validate_username(username: str) -> str:
    """Validate Chess.com username to prevent injection."""
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        st.error("❌ Invalid username. Only letters, numbers, underscores, and hyphens allowed.")
        st.stop()
    return username

def run_phase(label, func, progress_el, status_el, pct, **kwargs):
    """Run a pipeline phase function directly (no subprocess)."""
    status_el.markdown(f"⏳ **{label}**")
    try:
        result = func(**kwargs)
        progress_el.progress(pct)
        return result
    except Exception as e:
        st.error(f"❌ {label} failed:\n\n```\n{traceback.format_exc()}\n```")
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("♟ ChessLens")
    st.markdown("---")

    username = st.text_input("Chess.com Username", placeholder="e.g. MITEX7").strip()

    st.markdown("---")
    st.markdown("### 🎛️ Analysis Mode")

    # Stockfish path — shown only if not auto-detected
    import phase2_engine_eval as _p2_check
    import shutil as _shutil
    _sf_ok = _shutil.which(_p2_check.STOCKFISH_PATH) or __import__('os').path.isfile(_p2_check.STOCKFISH_PATH)
    if not _sf_ok:
        st.warning("⚠️ Stockfish not found — set path below")
    with st.expander("⚙️ Stockfish Path" + (" ✅" if _sf_ok else " ❌"), expanded=not _sf_ok):
        sf_path_input = st.text_input(
            "Stockfish executable path",
            value=config.get("stockfish_path", _p2_check.STOCKFISH_PATH),
            placeholder=r"Windows: C:\stockfish\stockfish.exe  |  Mac: /usr/local/bin/stockfish",
            help="Download from https://stockfishchess.org/download/ and paste the path here."
        ).strip()
        if sf_path_input:
            config["stockfish_path"] = sf_path_input
            save_config(config)
            _p2_check.STOCKFISH_PATH = sf_path_input


    mode = st.radio(
        "Select Mode",
        options=["⚡ Fast Mode", "🔬 Deep Mode"],
        index=0 if config.get("mode", "fast") == "fast" else 1,
        help="Fast = local Stockfish on suspicious moves only. Deep = local Stockfish depth 18 on all moves."
    )
    mode_key = "fast" if "Fast" in mode else "deep"

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    if mode_key == "fast":
        st.markdown("""<div class="settings-box mode-fast">
        ⚡ <b>Fast Mode</b><br>
        • Local Stockfish on ~20% of moves<br>
        • Standard mistake taxonomy<br>
        • Best for quick weekly check-ins
        </div>""", unsafe_allow_html=True)

        max_games = st.slider("Recent Games", 50, 500, config.get("max_games", 100), 50,
                              help="50-150 games recommended for fast mode")
        eval_time = st.select_slider(
            "Engine Accuracy",
            options=[0.01, 0.02, 0.05],
            value=config.get("eval_time", 0.02),
            format_func=lambda x: {0.01:"⚡ Fastest (~30s)", 0.02:"✅ Balanced (~1-2min)", 0.05:"🎯 Accurate (~3min)"}[x]
        )
        lichess_token = ""
        num_workers   = 1

    else:
        st.markdown("""<div class="settings-box mode-deep">
        🔬 <b>Deep Mode</b><br>
        • Local Stockfish depth 18 on every move<br>
        • Extended mistake taxonomy (15+ types)<br>
        • Best for in-depth monthly analysis
        </div>""", unsafe_allow_html=True)

        max_games = st.slider("Recent Games", 50, 200, min(config.get("max_games", 100), 200), 50,
                              help="100 games recommended for deep mode")

        num_workers = st.slider("Parallel Workers", 1, 8,
                                value=config.get("num_workers", 5),
                                help="More workers = faster. Set to number of CPU cores for best results.")

        eval_time = 0.02

    st.markdown(f"""
    <div class="settings-box">
        📊 <b>{max_games} games</b> | Mode: {'Fast ⚡' if mode_key=='fast' else 'Deep 🔬'}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    run_button = st.button("🚀 Analyze My Games")


# ── Main ──────────────────────────────────────────────────────────────────────────

st.title("♟ ChessLens — AI Chess Coach")
st.markdown("*Stop guessing why you lose. Start knowing.*")
st.markdown("---")

if not username:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="drill-box">
        <h3>⚡ Fast Mode</h3>
        <p>Quick analysis in minutes. Local Stockfish on suspicious moves. Perfect for weekly check-ins.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="drill-box">
        <h3>🔬 Deep Mode</h3>
        <p>Full depth-20+ analysis via Lichess. 15+ mistake types including zwischenzug, overloaded pieces, sacrifices.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="drill-box">
        <h3>📅 Training Plan</h3>
        <p>Personalized weekly drill schedule built from your specific mistake patterns.</p>
        </div>""", unsafe_allow_html=True)
    st.info("👈 Enter your Chess.com username in the sidebar to get started.")
    st.stop()


# ── Run pipeline ──────────────────────────────────────────────────────────────────

if run_button:
    username = validate_username(username)

    # Merge into existing config so stockfish_path (and other persisted keys) are preserved
    config.update({
        "max_games": max_games, "eval_time": eval_time,
        "mode": mode_key, "num_workers": num_workers,
    })
    save_config(config)

    for p in [moves_path(username), evaluated_path(username), features_path(username), categorized_path(username)]:
        if os.path.exists(p): os.remove(p)

    os.makedirs("data/analytics", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/raw_pgn", exist_ok=True)

    st.markdown(f"### {'⚡ Fast' if mode_key=='fast' else '🔬 Deep'} analysis — **{max_games} games** for **{username}**")
    st.caption(f"Mode: {mode_key}")

    progress = st.progress(0)
    status   = st.empty()

    failed = False

    # Phase 1a — Fetch games
    from phase1_fetch_games import fetch_all_games
    result = run_phase("Phase 1a — Fetching games from Chess.com",
        fetch_all_games, progress, status, 12,
        username=username, output_dir='data/raw_pgn')
    if result is None:
        failed = True

    # Phase 1b — Parse PGN (cap to max_games most-recent games for speed)
    if not failed:
        from phase1_parse_pgn import parse_all_pgn_files
        result = run_phase("Phase 1b — Parsing PGN files",
            parse_all_pgn_files, progress, status, 22,
            pgn_dir='data/raw_pgn', player_username=username,
            output_path=moves_path(username), max_games=max_games)
        if result is None:
            failed = True

    # Phase 2 — Engine evaluation
    if not failed:
        import phase2_engine_eval as p2
        p2.MODE          = mode_key
        p2.MAX_GAMES     = max_games
        p2.EVAL_TIME     = eval_time
        p2.NUM_WORKERS   = num_workers
        result = run_phase(f"Phase 2 — Engine evaluation ({mode_key} mode)",
            p2.add_engine_evaluations, progress, status, 60,
            moves_path=moves_path(username),
            output_path=evaluated_path(username), resume=False)
        if result is None:
            failed = True

    # Phase 3 — Feature engineering
    if not failed:
        from phase3_feature_engineering import add_features
        result = run_phase("Phase 3 — Extracting position features",
            add_features, progress, status, 72,
            evaluated_path=evaluated_path(username),
            output_path=features_path(username))
        if result is None:
            failed = True

    # Phase 4 — Taxonomy
    if not failed:
        from phase4_taxonomy import add_detailed_taxonomy
        result = run_phase(f"Phase 4 — Classifying mistakes ({mode_key} mode)",
            add_detailed_taxonomy, progress, status, 83,
            features_path=features_path(username),
            output_path=categorized_path(username), mode=mode_key)
        if result is None:
            failed = True

    # Phase 5 — Analytics
    if not failed:
        import matplotlib
        matplotlib.use('Agg')
        from phase5_analytics import load_data, print_deep_report, plot_focused_dashboard
        try:
            status.markdown("⏳ **Phase 5 — Building analytics**")
            df_loaded = load_data(categorized_path(username))
            print_deep_report(df_loaded, username)
            plot_focused_dashboard(df_loaded, username)
            progress.progress(92)
        except Exception:
            st.error(f"❌ Phase 5 failed:\n\n```\n{traceback.format_exc()}\n```")
            failed = True

    # Phase 6 — Recommendations
    if not failed:
        from phase6_recommendations import run_recommendations
        result = run_phase("Phase 6 — Generating coaching report",
            run_recommendations, progress, status, 100,
            data_path=categorized_path(username), username=username)
        if result is None:
            failed = True

    if not failed:
        status.markdown("✅ **Analysis complete!**")
        st.success(f"Done! Scroll down to see your {'deep' if mode_key=='deep' else 'fast'} analysis for **{username}**.")
        st.rerun()


# ── Results ───────────────────────────────────────────────────────────────────────

if data_exists(username):
    df             = pd.read_parquet(categorized_path(username))
    mistakes       = df[df["mistake_category"] != "none"]
    total          = len(df)
    total_mistakes = len(mistakes)

    st.markdown(f"## 📊 Results for **{username}**")
    st.caption(f"{total} moves analyzed")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Mistakes per 100 moves", round(total_mistakes/max(total,1)*100,1))
    with c2: st.metric("Total Blunders", len(df[df["mistake_type"]=="blunder"]))
    with c3:
        avg_cp = df[df["cp_loss"]>0]["cp_loss"].mean()
        st.metric("Avg CP Loss", round(avg_cp,1) if pd.notna(avg_cp) else "N/A")
    with c4:
        wins   = len(df[df["result"]=="win"])
        losses = len(df[df["result"]=="loss"])
        st.metric("Win / Loss Moves", f"{wins} / {losses}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Tactic Blind Spots", "📈 Charts", "📅 Training Plan", "📋 Full Report"
    ])

    with tab1:
        st.markdown("### Tactics You Miss Most")
        missed = mistakes[mistakes["mistake_category"] == "missed_tactic"]

        TACTIC_INFO = {
            # Basic
            "fork":                  ("🍴 Fork",                "Attacking two pieces at once"),
            "knight_fork":           ("🐴 Knight Fork",         "Knight attacking two pieces simultaneously"),
            "pawn_fork":             ("♟ Pawn Fork",            "Pawn attacking two pieces at once"),
            "pin":                   ("📌 Pin",                  "Pinning a piece to the king or queen"),
            "skewer":                ("⚔️ Skewer",              "Attacking through a valuable piece to win what's behind"),
            "back_rank_mate":        ("🏠 Back Rank Mate",      "Checkmate on the opponent's back rank"),
            "discovered_attack":     ("💥 Discovered Attack",   "Moving one piece to reveal a hidden attacker"),
            # Deep mode extras
            "zwischenzug":           ("⚡ Zwischenzug",          "An in-between move instead of recapturing immediately"),
            "missed_sacrifice":      ("🎭 Missed Sacrifice",     "Giving up material for a decisive positional or tactical advantage"),
            "overloaded_piece":      ("🔄 Overloaded Piece",     "Opponent's piece was defending too many things at once"),
            # Phase-based calculation
            "opening_calculation":   ("📖 Opening Mistake",     "Calculation error in the opening phase"),
            "middlegame_calculation":("🧮 Middlegame Mistake",  "Calculation error in the middlegame"),
            "endgame_calculation":   ("🏁 Endgame Mistake",     "Calculation error in the endgame"),
            "calculation_error":     ("🧮 Calculation Error",   "General calculation error"),
        }

        if not missed.empty:
            for tactic, count in missed["tactic_type"].value_counts().items():
                pct         = round(count / len(missed) * 100, 1)
                label, desc = TACTIC_INFO.get(tactic, (tactic.replace("_"," ").title(), ""))
                color       = "#EF5350" if pct > 30 else "#FFA726" if pct > 15 else "#42A5F5"
                st.markdown(f"""
                <div class="drill-box">
                    <b style="color:{color}">{label}</b> — {pct}% of missed tactics ({count} times)<br>
                    <small style="color:#9e9e9e">{desc}</small>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("✅ No missed tactics detected.")

        st.markdown("### Pieces You Hang Most")
        hanging = mistakes[mistakes["mistake_category"] == "hanging_piece"]
        if not hanging.empty:
            PIECE_EMOJI = {"queen":"👑","rook":"🏰","bishop":"🗼","knight":"🐴","pawn":"♟"}
            cols = st.columns(len(hanging["piece_lost"].value_counts()))
            for i, (piece, count) in enumerate(hanging["piece_lost"].value_counts().items()):
                with cols[i]:
                    st.metric(f"{PIECE_EMOJI.get(piece,'♟')} {piece.title()}", f"{count}x")
        else:
            st.success("✅ No hanging pieces detected.")

        # Deep mode extras
        trapped = mistakes[mistakes["mistake_category"] == "trapped_piece"]
        if not trapped.empty:
            st.markdown("### Trapped Pieces")
            for t, c in trapped["tactic_type"].value_counts().items():
                st.markdown(f"""<div class="drill-box">
                <b>🪤 {t.replace('_',' ').title()}</b> — {c} times<br>
                <small style="color:#9e9e9e">You moved a piece to a square where it got trapped</small>
                </div>""", unsafe_allow_html=True)

        st.markdown("### Missed Forced Mates")
        missed_mates = mistakes[mistakes["mistake_category"] == "missed_mate"]
        if not missed_mates.empty:
            c1, c2 = st.columns(2)
            with c1: st.metric("💀 Total Missed Mates", len(missed_mates))
            with c2:
                pct = round(len(missed_mates) / max(len(mistakes), 1) * 100, 1)
                st.metric("💀 % of All Mistakes", f"{pct}%")
        else:
            st.success("✅ No missed forced mates.")

    with tab2:
        cp = chart_path(username)
        if os.path.exists(cp):
            st.image(Image.open(cp), use_container_width=True)
        else:
            st.warning("Chart not found — re-run the analysis.")

        st.markdown("### Mistake Rate by Phase")
        rows = []
        for phase in ["opening","middlegame","endgame"]:
            phase_df  = df[df["phase"]==phase]
            phase_mis = phase_df[phase_df["mistake_type"].isin(["mistake","blunder"])]
            if len(phase_df) > 0:
                rows.append({
                    "Phase": phase.title(), "Total Moves": len(phase_df),
                    "Mistakes": len(phase_mis),
                    "Mistake Rate %": round(len(phase_mis)/len(phase_df)*100,1)
                })
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Phase"), use_container_width=True)

    with tab3:
        st.markdown("### 📅 Your Personalized Weekly Training Plan")
        DRILLS = {
            "fork":                   "Solve 20 Fork puzzles — Lichess (filter: Fork theme)",
            "knight_fork":            "Solve 20 Knight Fork puzzles — chess.com/puzzles filter: Fork",
            "pawn_fork":              "Solve 15 Pawn Fork puzzles — Lichess (filter: Fork)",
            "pin":                    "Solve 20 Pin puzzles — Lichess (filter: Pin theme)",
            "skewer":                 "Solve 15 Skewer puzzles — Lichess (filter: Skewer)",
            "back_rank_mate":         "Study back rank mates — Lichess Practice → Checkmate Patterns",
            "discovered_attack":      "Solve Discovered Attack puzzles — Lichess",
            "zwischenzug":            "Study in-between moves — search 'zwischenzug puzzles' on Lichess",
            "missed_sacrifice":       "Study tactical sacrifices — search 'sacrifice' puzzles on chess.com",
            "overloaded_piece":       "Solve overloaded piece puzzles — Lichess (filter: Overloading)",
            "opening_calculation":    "Review your opening lines — use chess.com Opening Explorer",
            "middlegame_calculation":  "Play 15+10 games and calculate 3 moves ahead before every move",
            "endgame_calculation":    "Practice endgames — Lichess Practice → Endgame Training",
            "calculation_error":      "Play 15+10 games and list checks, captures, threats before every move",
            "hanging_piece":          "Puzzle Rush Survival on chess.com — 15 minutes daily",
        }
        missed         = mistakes[mistakes["mistake_category"]=="missed_tactic"]
        top_tactics    = missed["tactic_type"].value_counts().head(2).index.tolist() \
                         if not missed.empty else ["calculation_error"]
        worst_phase    = (
            df.groupby("phase", group_keys=False)
            .apply(lambda x: len(x[x["mistake_type"].isin(["mistake","blunder"])])/max(len(x),1),
                   include_groups=False)
            .idxmax()
        ) if "phase" in df.columns and not df["phase"].dropna().empty else "middlegame"
        days = {
            "Monday":    (f"🎯 {top_tactics[0].replace('_',' ').title()} Puzzles",   DRILLS.get(top_tactics[0],"Solve puzzles on Lichess")),
            "Tuesday":   ("⚠️ Hanging Piece Drills",                                  DRILLS["hanging_piece"]),
            "Wednesday": (f"🎯 {top_tactics[-1].replace('_',' ').title()} Puzzles",  DRILLS.get(top_tactics[-1],"Mixed tactics")),
            "Thursday":  (f"📖 {worst_phase.title()} Study",                         f"Focus on your weakest phase: {worst_phase}"),
            "Friday":    ("🎮 Play + Review",                                         "Play 2 rapid games (10+0), review with engine after"),
            "Saturday":  ("🔍 Game Review",                                           "Review your 3 worst games this week — find the pattern"),
            "Sunday":    ("😌 Rest",                                                  "Optional: 15 min mixed puzzle session"),
        }
        for day, (activity, detail) in days.items():
            st.markdown(f"""<div class="drill-box">
            <b>{day}</b> — {activity}<br>
            <small style="color:#9e9e9e">→ {detail}</small>
            </div>""", unsafe_allow_html=True)
        st.markdown("---")
        st.info("⚡ Fix one tactic blind spot at a time. Don't move on until your puzzle rating for that theme improves.")

    with tab4:
        rp = report_path(username)
        if os.path.exists(rp):
            with open(rp) as f: text = f.read()
            st.download_button("⬇️ Download Full Report", data=text,
                file_name=f"{username}_chesslens_report.txt", mime="text/plain")
            st.code(text, language=None)
        else:
            st.warning("Report not generated yet.")

else:
    st.info(f"No analysis found for **{username}** yet. Click **Analyze My Games** in the sidebar to start.")