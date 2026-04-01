import argparse
import os
import sys
import re


def validate_username(username: str) -> str:
    """Validate and sanitize the Chess.com username."""
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        raise ValueError(f"Invalid username: '{username}'. Only alphanumeric, underscore, and hyphen allowed.")
    return username


def run_pipeline(username: str, skip_eval: bool = False, skip_ml: bool = True):

    username = validate_username(username)

    print(f"\n{'='*60}")
    print(f"  CHESS MISTAKE ANALYZER — FULL PIPELINE")
    print(f"  Player: {username}")
    print(f"{'='*60}\n")

    # ── Path helpers ──────────────────────────────────────────────────────────
    data_dir       = f"data/analytics"
    moves_file     = f"data/processed/moves.parquet"
    eval_file      = f"data/processed/moves_evaluated.parquet"
    features_file  = f"data/processed/moves_features.parquet"
    cat_file       = f"data/processed/moves_categorized.parquet"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # ── PHASE 1a: Fetch Games ──────────────────────────────────────────────────
    print("\n[PHASE 1a] Fetching games from Chess.com...")
    from phase1_fetch_games import fetch_all_games
    fetch_all_games(username, output_dir="data/raw_pgn")

    # ── PHASE 1b: Parse PGN ────────────────────────────────────────────────────
    print("\n[PHASE 1b] Parsing PGN files (most recent games)...")
    from phase1_parse_pgn import parse_all_pgn_files
    df = parse_all_pgn_files(
        pgn_dir="data/raw_pgn",
        player_username=username,
        output_path=moves_file,
        max_games=None,  # None = all games; Phase 2 applies MAX_GAMES cap
    )
    print(f"  → {len(df)} player moves parsed")

    # ── PHASE 2: Engine Evaluation ─────────────────────────────────────────────
    if skip_eval:
        print("\n[PHASE 2] SKIPPED (--skip_eval flag set)")
        print("  → Using moves.parquet without engine evaluation")
        import shutil
        shutil.copy(moves_file, eval_file)
    else:
        print("\n[PHASE 2] Running Stockfish evaluation...")
        from phase2_engine_eval import add_engine_evaluations
        df_eval = add_engine_evaluations(
            moves_path=moves_file,
            output_path=eval_file,
            resume=True,
        )
        print(f"  → {len(df_eval)} moves evaluated")

    # ── PHASE 3: Feature Engineering ──────────────────────────────────────────
    if os.path.exists(features_file):
        print("\n[PHASE 3] SKIPPED (already exists)")
    else:
        print("\n[PHASE 3] Extracting position features...")
        from phase3_feature_engineering import add_features
        df_features = add_features(
            evaluated_path=eval_file,
            output_path=features_file,
        )
        print(f"  → Features added: {df_features.shape[1]} columns")

    # ── PHASE 4: Mistake Taxonomy ──────────────────────────────────────────────
    if os.path.exists(cat_file):
        print("\n[PHASE 4] SKIPPED (already exists)")
    else:
        print("\n[PHASE 4] Classifying mistakes...")
        from phase4_taxonomy import add_detailed_taxonomy
        add_detailed_taxonomy(
            features_path=features_file,
            output_path=cat_file,
        )

    # ── PHASE 5: Analytics ─────────────────────────────────────────────────────
    print("\n[PHASE 5] Running analytics...")
    from phase5_analytics import load_data, print_deep_report, plot_focused_dashboard
    df_final = load_data(cat_file)
    print_deep_report(df_final, username)
    plot_focused_dashboard(df_final, username)

    # ── PHASE 6: Recommendations ───────────────────────────────────────────────
    from phase6_recommendations import run_recommendations
    run_recommendations(
        data_path=cat_file,
        username=username,
    )

    # ── PHASE 7: ML (Optional) ─────────────────────────────────────────────────
    if not skip_ml:
        print("\n[PHASE 7] Training ML models...")
        from phase7_ml_models import train_blunder_predictor
        train_blunder_predictor(
            data_path=features_file,
            model_name="random_forest",
        )

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"  Outputs saved to: {data_dir}/")
    print(f"{'='*60}\n")


# ── Validation: Compare Two Players ────────────────────────────────────────────

def compare_players(username1: str, username2: str):
    """Load fingerprints and compare two players' mistake profiles."""
    import json

    username1 = validate_username(username1)
    username2 = validate_username(username2)

    def load_fp(username):
        path = f"data/analytics/{username}_fingerprint.json"
        if not os.path.exists(path):
            print(f"No fingerprint found for {username}. Run pipeline first.")
            return None
        with open(path) as f:
            return json.load(f)

    fp1 = load_fp(username1)
    fp2 = load_fp(username2)

    if not fp1 or not fp2:
        return

    print(f"\n{'='*55}")
    print(f"  PLAYER COMPARISON: {username1} vs {username2}")
    print(f"{'='*55}")

    def safe_get(fp, *keys):
        val = fp
        for k in keys:
            val = val.get(k, "N/A") if isinstance(val, dict) else "N/A"
        return val

    metrics = [
        ("Mistakes per 100 moves", ["overall", "mistakes_per_100"]),
        ("Blunders per 100 moves", ["overall", "blunders_per_100"]),
        ("Avg CP loss", ["overall", "avg_cp_loss"]),
        ("Opening mistake rate %", ["opening_mistake_rate_%"]),
        ("Middlegame mistake rate %", ["middlegame_mistake_rate_%"]),
        ("Endgame mistake rate %", ["endgame_mistake_rate_%"]),
        ("Time pressure blunder rate", ["time_pressure_stats", "blunder_rate_under_pressure_%"]),
    ]

    print(f"\n{'Metric':<35} {username1:<20} {username2:<20}")
    print("-" * 75)
    for label, keys in metrics:
        v1 = safe_get(fp1, *keys)
        v2 = safe_get(fp2, *keys)
        print(f"{label:<35} {str(v1):<20} {str(v2):<20}")

    print(f"\n{'Top Mistake Categories':<35} {username1:<20} {username2:<20}")
    print("-" * 75)
    cats1 = safe_get(fp1, "top_mistake_categories") or {}
    cats2 = safe_get(fp2, "top_mistake_categories") or {}
    all_cats = set(list(cats1.keys()) + list(cats2.keys()))
    for cat in sorted(all_cats):
        v1 = cats1.get(cat, 0)
        v2 = cats2.get(cat, 0)
        print(f"{cat:<35} {str(v1)+'%':<20} {str(v2)+'%':<20}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess Mistake Analyzer")
    parser.add_argument("--username", required=True, help="Chess.com username")
    parser.add_argument("--skip_eval", action="store_true", help="Skip Stockfish evaluation")
    parser.add_argument("--skip_ml", action="store_true", default=True, help="Skip ML training")
    parser.add_argument("--compare", help="Second username to compare against")

    args = parser.parse_args()

    if args.compare:
        compare_players(args.username, args.compare)
    else:
        run_pipeline(args.username, skip_eval=args.skip_eval, skip_ml=args.skip_ml)
