import pandas as pd
import os
import sys

from phase5_analytics import (
    tactic_breakdown, hanging_piece_breakdown, worst_phase,
    worst_move_range, time_pressure_analysis, missed_mate_analysis,
    get_output_dir
)

TACTIC_DRILLS = {
    "fork": {
        "what": "You miss moves that attack two pieces at once.",
        "drills": [
            "Solve 20 fork puzzles daily — Lichess (filter: Fork theme)",
            "Study knight fork patterns — focus on squares d5, e5, c5",
            "Before every move ask: can any piece attack two targets at once?",
            "Drill: chess.com/puzzles → filter by Fork",
        ]
    },
    "pin": {
        "what": "You miss opportunities to pin opponent pieces to their king or queen.",
        "drills": [
            "Solve 20 pin puzzles daily — Lichess (filter: Pin theme)",
            "Study absolute vs relative pins — when is a pinned piece truly stuck?",
            "Practice spotting bishop and rook alignment with the opponent king",
            "Drill: chess.com/puzzles → filter by Pin",
        ]
    },
    "skewer": {
        "what": "You miss skewers — attacking a valuable piece with another piece behind it.",
        "drills": [
            "Solve 15 skewer puzzles daily — Lichess (filter: Skewer theme)",
            "Study rook and queen skewer patterns on open files and diagonals",
            "Check if opponent king or queen shares a file or diagonal with another piece",
            "Drill: chess.com/puzzles → filter by Skewer",
        ]
    },
    "back_rank_mate": {
        "what": "You miss back rank mates — the opponent king is trapped on the first or eighth rank.",
        "drills": [
            "Solve back rank mate puzzles on Lichess daily",
            "Whenever you have a rook on the 7th rank, look for a back rank finish",
            "Study the Lucena and Philidor positions — back rank awareness matters in endgames too",
            "Drill: Lichess → Practice → Checkmate Patterns → Back Rank",
        ]
    },
    "discovered_attack": {
        "what": "You miss discovered attacks — moving one piece reveals a deadly attack by another.",
        "drills": [
            "Solve discovered attack puzzles — Lichess (filter: Discovered Attack)",
            "Before moving any piece ask: what attack am I revealing behind it?",
            "Study how bishops and rooks work together for discovered checks",
            "Drill: chess.com/puzzles → filter by Discovered Attack",
        ]
    },
    "calculation_error": {
        "what": "You go wrong in multi-move sequences — calculation errors.",
        "drills": [
            "Practice visualizing 3 moves ahead before touching any piece",
            "Use the candidate moves method — list all checks, captures, threats first",
            "Solve medium-difficulty puzzles (1500-1800 rated) on chess.com daily",
            "Play 15+10 games to practice deeper calculation with more time",
        ]
    },
    "mate_in_1": {
        "what": "You are missing checkmates in 1 move — the most basic oversight.",
        "drills": [
            "Solve 50 mate-in-1 puzzles every single day until your miss rate is zero",
            "Before ending your turn always ask: can I checkmate right now?",
            "Drill: chess.com → Puzzles → filter by Checkmate in 1",
            "This must be fixed before anything else — it is costing you easy wins",
        ]
    },
    "mate_in_2": {
        "what": "You are missing checkmates in 2 moves.",
        "drills": [
            "Solve 30 mate-in-2 puzzles daily on chess.com or Lichess",
            "Study common mating patterns: back rank, smothered mate, queen and bishop",
            "After every move check: can I force checkmate in two from here?",
            "Drill: Lichess → Practice → Checkmate Patterns",
        ]
    },
    "hanging_piece": {
        "what": "You leave pieces undefended where they can be taken for free.",
        "drills": [
            "After every move you consider, scan all your pieces — is any undefended?",
            "Remember LPDO: Loose Pieces Drop Off — any undefended piece is a target",
            "Solve Puzzle Rush Survival on chess.com daily for quick threat detection",
            "Drill: chess.com → Puzzles → filter by Hanging Piece",
        ]
    },
    "time_pressure": {
        "what": "Your move quality collapses when you have less than 60 seconds on the clock.",
        "drills": [
            "Switch from bullet (1-2 min) to rapid (10+ min) games immediately",
            "When under 60 seconds stop calculating — play the most solid defensive move",
            "Practice the 5-second rule: spend first 5 seconds looking for opponent threats only",
            "Drill: play 5|3 increment games to practice not getting into time trouble",
        ]
    },
    "king_exposed": {
        "what": "You leave your king unsafe — not castling or opening lines near your king.",
        "drills": [
            "Prioritize castling in every game — aim to castle before move 10",
            "Never open the center before your king is castled",
            "Study attacking games by Kasparov and Tal to recognize king attack patterns",
            "Review your last 10 games — in how many did you castle after move 12?",
        ]
    },
    "opening_principle": {
        "what": "You are violating opening principles in the first 12 moves.",
        "drills": [
            "Memorize the 3 opening rules: control center, develop pieces, castle early",
            "Narrow your repertoire to 2 openings per color and study them deeply",
            "Use chess.com Opening Explorer to find exactly which moves you get wrong",
            "Study: chess.com Lessons → Opening Principles",
        ]
    },
    "endgame_technique": {
        "what": "You make technique errors in the endgame — losing or drawing won positions.",
        "drills": [
            "Study King + Pawn vs King — learn opposition and key squares",
            "Study Rook endings: Lucena position (winning) and Philidor position (drawing)",
            "Practice endgames on Lichess → Practice → Endgame Training daily",
            "When queens come off, activate your king immediately — every tempo counts",
        ]
    },
    "weak_pawns": {
        "what": "You create isolated or doubled pawns that become long-term weaknesses.",
        "drills": [
            "Study pawn structure theory for your main openings",
            "Think twice before any exchange that leaves doubled or isolated pawns",
            "In your next 10 games, count your pawn weaknesses after move 15",
            "Study: Nimzowitsch My System — chapter on pawn weaknesses",
        ]
    },
}


def generate_full_report(df: pd.DataFrame, username: str) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append(f"  CHESSLENS PERSONALIZED COACHING REPORT")
    lines.append(f"  Player: {username.upper()}")
    lines.append("=" * 65)

    mistakes = df[df["mistake_category"] != "none"]
    total_mistakes = len(mistakes)
    total_moves = len(df)

    if total_mistakes == 0:
        lines.append("\nNo mistakes found. Play more games and re-run.")
        return "\n".join(lines)

    lines.append(f"\n  {total_mistakes} mistakes found across {total_moves} moves.\n")

    # ── Missed mate in 1 (most critical) ────────────────────────────────────────
    mate_data = missed_mate_analysis(df)
    if mate_data["missed_mate_in_1"] > 0:
        lines.append(f"{'─'*65}")
        lines.append(f"🔴 CRITICAL — YOU MISSED {mate_data['missed_mate_in_1']} CHECKMATES IN 1 MOVE")
        lines.append(f"{'─'*65}")
        lines.append(f"  {TACTIC_DRILLS['mate_in_1']['what']}\n")
        for d in TACTIC_DRILLS["mate_in_1"]["drills"]:
            lines.append(f"  → {d}")
        lines.append("")

    if mate_data["missed_mate_in_2"] > 0:
        lines.append(f"{'─'*65}")
        lines.append(f"🔴 HIGH PRIORITY — YOU MISSED {mate_data['missed_mate_in_2']} CHECKMATES IN 2")
        lines.append(f"{'─'*65}")
        lines.append(f"  {TACTIC_DRILLS['mate_in_2']['what']}\n")
        for d in TACTIC_DRILLS["mate_in_2"]["drills"]:
            lines.append(f"  → {d}")
        lines.append("")

    # ── Tactical blind spots ────────────────────────────────────────────────────
    tdf = tactic_breakdown(df)
    if not tdf.empty:
        lines.append(f"{'─'*65}")
        lines.append(f"🔴 YOUR TACTICAL BLIND SPOTS (ranked by frequency)")
        lines.append(f"{'─'*65}")
        for tactic, row in tdf.iterrows():
            if tactic in TACTIC_DRILLS:
                info = TACTIC_DRILLS[tactic]
                lines.append(f"\n  [{row['percent_%']}%] {tactic.upper().replace('_',' ')} — {row['count']} times")
                lines.append(f"  What: {info['what']}")
                lines.append(f"  Fix:")
                for d in info["drills"]:
                    lines.append(f"    → {d}")
        lines.append("")

    # ── Hanging pieces ──────────────────────────────────────────────────────────
    hdf = hanging_piece_breakdown(df)
    hang_total = len(mistakes[mistakes["mistake_category"] == "hanging_piece"])
    if hang_total > 0:
        hang_pct = round(hang_total / total_mistakes * 100, 1)
        lines.append(f"{'─'*65}")
        lines.append(f"🟡 HANGING PIECES — {hang_pct}% of your mistakes ({hang_total} times)")
        lines.append(f"{'─'*65}")
        if not hdf.empty:
            lines.append(f"  Pieces hung most:")
            for piece, row in hdf.iterrows():
                lines.append(f"    {piece:<10}: {row['count']} times ({row['percent_%']}%)")
        lines.append("")
        for d in TACTIC_DRILLS["hanging_piece"]["drills"]:
            lines.append(f"  → {d}")
        lines.append("")

    # ── Time pressure ───────────────────────────────────────────────────────────
    tp = time_pressure_analysis(df)
    if tp and tp.get("multiplier", 0) >= 1.5:
        lines.append(f"{'─'*65}")
        lines.append(f"🟡 TIME PRESSURE — You blunder {tp['multiplier']}x more under 60 seconds")
        lines.append(f"{'─'*65}")
        lines.append(f"  Normal blunder rate    : {tp['blunder_rate_normal_%']}%")
        lines.append(f"  Under 60s blunder rate : {tp['blunder_rate_under_pressure_%']}%\n")
        for d in TACTIC_DRILLS["time_pressure"]["drills"]:
            lines.append(f"  → {d}")
        lines.append("")

    # ── Worst phase ─────────────────────────────────────────────────────────────
    worst, phase_rates = worst_phase(df)
    lines.append(f"{'─'*65}")
    lines.append(f"🟡 WEAKEST PHASE: {worst.upper()} ({phase_rates.get(worst, 0)}% mistake rate)")
    lines.append(f"{'─'*65}")
    phase_tactic = {"opening": "opening_principle", "endgame": "endgame_technique", "middlegame": "calculation_error"}
    key = phase_tactic.get(worst, "calculation_error")
    lines.append(f"  {TACTIC_DRILLS[key]['what']}\n")
    for d in TACTIC_DRILLS[key]["drills"]:
        lines.append(f"  → {d}")
    lines.append("")

    # ── Danger move range ───────────────────────────────────────────────────────
    move_data = worst_move_range(df)
    lines.append(f"{'─'*65}")
    lines.append(f"🟢 DANGER ZONE: Moves {move_data['worst_range']} ({move_data['rate_%']}% mistake rate)")
    lines.append(f"{'─'*65}")
    lines.append(f"  You make the most mistakes in moves {move_data['worst_range']}.")
    lines.append(f"  → Slow down specifically during this move range")
    lines.append(f"  → Spend at least 2 minutes per move here")
    lines.append(f"  → Before each move in this range use CCT: Checks, Captures, Threats\n")

    # ── Weekly plan ─────────────────────────────────────────────────────────────
    top_tactics = list(tdf.index[:2]) if not tdf.empty else ["calculation_error"]
    lines.append(f"{'─'*65}")
    lines.append(f"📅 YOUR WEEKLY TRAINING PLAN")
    lines.append(f"{'─'*65}")
    lines.append(f"  Monday    : 20 min — {top_tactics[0].replace('_',' ').title()} puzzles on Lichess")
    lines.append(f"  Tuesday   : 20 min — Hanging piece drills (Puzzle Rush Survival)")
    if len(top_tactics) > 1:
        lines.append(f"  Wednesday : 20 min — {top_tactics[1].replace('_',' ').title()} puzzles on chess.com")
    else:
        lines.append(f"  Wednesday : 20 min — Mixed tactics puzzles")
    lines.append(f"  Thursday  : 20 min — {worst.title()} study")
    lines.append(f"  Friday    : Play 2 rapid games (10+0) then review with engine")
    lines.append(f"  Saturday  : Review your 3 worst games this week — find the pattern")
    lines.append(f"  Sunday    : Rest or optional 15 min light puzzle session")
    lines.append(f"\n  ⚡ Golden Rule: Fix ONE thing at a time.")
    lines.append(f"  Start with your #1 tactic blind spot. Don't move on until it improves.")
    lines.append("=" * 65)

    return "\n".join(lines)


def run_recommendations(data_path: str, username: str):
    df = pd.read_parquet(data_path)
    report = generate_full_report(df, username)
    print(report)

    output_dir = get_output_dir(username)
    out_path = os.path.join(output_dir, f"{username}_coaching_report.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    username = sys.argv[1] if len(sys.argv) > 1 else "player"
    run_recommendations(
        data_path=f"data/{username}/moves_categorized.parquet",
        username=username
    )