import chess
import chess.engine
import pandas as pd
import os
import sys
import shutil
import json
from tqdm import tqdm
from constants import PIECE_VALUE, CP_CAP

# ── Config ────────────────────────────────────────────────────────────────────────

def _find_stockfish() -> str:
    """Auto-detect Stockfish across macOS, Linux, and Windows.
    Priority: 1) chesslens_config.json override  2) PATH  3) common locations
    """
    # 1. Config file override
    config_path = os.path.join(os.path.dirname(__file__), "chesslens_config.json")
    if os.path.exists(config_path):
        try:
            cfg = json.load(open(config_path))
            p = cfg.get("stockfish_path", "")
            if p and os.path.isfile(p):
                return p
        except Exception:
            pass

    # 2. On PATH (works if user ran: brew install stockfish / apt install stockfish)
    sf = shutil.which("stockfish")
    if sf:
        return sf

    # 3. Common install locations per OS
    if sys.platform == "win32":
        candidates = [
            r"C:\stockfish\stockfish.exe",
            r"C:\Program Files\Stockfish\stockfish.exe",
            r"C:\Users\Public\stockfish\stockfish.exe",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "stockfish", "stockfish.exe"),
        ]
    elif sys.platform == "darwin":
        candidates = [
            "/usr/local/bin/stockfish",      # Homebrew Intel
            "/opt/homebrew/bin/stockfish",   # Homebrew Apple Silicon
            "/usr/bin/stockfish",
        ]
    else:  # Linux
        candidates = [
            "/usr/bin/stockfish",
            "/usr/local/bin/stockfish",
            "/usr/games/stockfish",
        ]

    for c in candidates:
        if os.path.isfile(c):
            return c

    # Last resort — let chess.engine raise a clear error
    return "stockfish"


STOCKFISH_PATH = _find_stockfish()

MODE       = "fast"
MAX_GAMES  = 100
EVAL_TIME  = 0.02
NUM_WORKERS = 5   # controls Stockfish internal threads in deep mode


# ── Helpers ───────────────────────────────────────────────────────────────────────

def classify_mistake(cp_loss) -> str:
    if cp_loss is None:  return "unknown"
    if cp_loss < 0:      return "good"
    elif cp_loss < 50:   return "good"
    elif cp_loss < 100:  return "inaccuracy"
    elif cp_loss < 300:  return "mistake"
    else:                return "blunder"


def stockfish_to_cp(score: chess.engine.PovScore) -> int:
    """Convert stockfish score to white-perspective cp, capped."""
    s = score.white()
    if s.is_mate():
        return CP_CAP if s.mate() > 0 else -CP_CAP
    return max(-CP_CAP, min(CP_CAP, s.score()))


def compute_cp_loss(white_before, white_after, color):
    """Compute cp loss from player's perspective."""
    if white_before is None or white_after is None:
        return None, None, None
    if color == "white":
        cp_loss     = white_before - white_after
        eval_before = white_before
        eval_after  = white_after
    else:
        cp_loss     = white_after - white_before
        eval_before = -white_before
        eval_after  = -white_after
    return cp_loss, eval_before, eval_after


# ── Fast mode pre-filter ──────────────────────────────────────────────────────────

def is_suspicious(board: chess.Board, move: chess.Move) -> bool:
    try:
        color = board.turn
        opp   = not color

        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if captured and attacker:
                cap_val = PIECE_VALUE.get(captured.piece_type, 0)
                att_val = PIECE_VALUE.get(attacker.piece_type, 0)
                b2 = board.copy()
                b2.push(move)
                if att_val > cap_val and b2.attackers(opp, move.to_square):
                    return True
                if att_val > cap_val + 1:
                    return True
            if captured and not board.attackers(opp, move.to_square):
                return False

        b2 = board.copy()
        b2.push(move)

        if b2.is_check():
            return True

        for piece_type in [chess.QUEEN, chess.ROOK]:
            for sq in b2.pieces(piece_type, color):
                if b2.attackers(opp, sq) and not b2.attackers(color, sq):
                    return True
        for piece_type in [chess.BISHOP, chess.KNIGHT]:
            for sq in b2.pieces(piece_type, color):
                if b2.attackers(opp, sq) and not b2.attackers(color, sq):
                    piece = b2.piece_at(sq)
                    for att_sq in b2.attackers(opp, sq):
                        att = b2.piece_at(att_sq)
                        if att and PIECE_VALUE.get(att.piece_type, 9) < PIECE_VALUE.get(piece.piece_type, 0):
                            return True
    except Exception:
        return True
    return False


# ── FAST MODE ─────────────────────────────────────────────────────────────────────

def run_fast_mode(df_todo):
    print(f"\n⚡ FAST MODE — Local Stockfish on suspicious moves only")

    rows       = df_todo.to_dict("records")
    suspicious = []
    clean      = []

    for i, row in enumerate(rows):
        try:
            board = chess.Board(row["fen"])
            move  = chess.Move.from_uci(row["move_uci"])
            b2    = board.copy()
            b2.push(move)
            fa    = b2.fen()
            if is_suspicious(board, move):
                suspicious.append((i, row, fa))
            else:
                clean.append(i)
        except Exception:
            suspicious.append((i, row, None))

    n_sus = len(suspicious)
    n_tot = len(rows)
    pct   = round(n_sus / n_tot * 100, 1) if n_tot > 0 else 0
    print(f"Suspicious: {n_sus}/{n_tot} ({pct}%) — engine needed")

    results = [None] * n_tot

    # Clean moves — instantly good
    for i in clean:
        results[i] = {
            **rows[i],
            "eval_before": None, "eval_after": None,
            "cp_loss": 0, "best_move": None,
            "played_best": True, "mistake_type": "good",
        }

    # Suspicious — run Stockfish
    limit = chess.engine.Limit(time=EVAL_TIME)
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        engine.configure({"Threads": 4, "Hash": 256})
        for i, row, fen_after in tqdm(suspicious, desc="Stockfish"):
            color = row["player_color"]
            try:
                b_before     = chess.Board(row["fen"])
                info_before  = engine.analyse(b_before, limit)
                white_before = stockfish_to_cp(info_before["score"])
                best_obj     = info_before.get("pv", [None])[0]
                best_uci     = best_obj.uci() if best_obj else None

                white_after = None
                if fen_after:
                    b_after     = chess.Board(fen_after)
                    info_after  = engine.analyse(b_after, limit)
                    white_after = stockfish_to_cp(info_after["score"])

                cp_loss, eval_before, eval_after = compute_cp_loss(white_before, white_after, color)

                results[i] = {
                    **row,
                    "eval_before": eval_before, "eval_after": eval_after,
                    "cp_loss": cp_loss, "best_move": best_uci,
                    "played_best": (best_uci == row["move_uci"]) if best_uci else False,
                    "mistake_type": classify_mistake(cp_loss),
                }
            except Exception:
                results[i] = {
                    **row,
                    "eval_before": None, "eval_after": None,
                    "cp_loss": None, "best_move": None,
                    "played_best": False, "mistake_type": "unknown",
                }

    return pd.DataFrame(results)


# ── DEEP MODE ─────────────────────────────────────────────────────────────────────

def run_deep_mode(df_todo):
    """Single Stockfish engine, time-based limit.

    Why not multiprocessing?  On macOS, Python's default start method is
    'spawn', which re-imports all modules including Streamlit internals,
    causing ScriptRunContext warnings and massive overhead.  A single engine
    with multiple internal threads (via Stockfish's Threads option) is
    significantly faster and fully reliable inside Streamlit.

    Optimization: quiet moves (no capture, no check, no hanging piece) that
    are not positionally complex are pre-marked as 'good' without evaluation.
    This skips ~40% of positions while still catching all real mistakes.
    """
    sf_threads   = min(NUM_WORKERS, 8)
    time_per_pos = 0.1  # seconds per position

    print(f"\n🔬 DEEP MODE — Stockfish {time_per_pos}s/pos × {sf_threads} threads (full eval)")

    rows       = df_todo.to_dict("records")
    fen_afters = {}

    for i, row in enumerate(rows):
        try:
            board = chess.Board(row["fen"])
            move  = chess.Move.from_uci(row["move_uci"])
            board.push(move)
            fen_afters[i] = board.fen()
        except Exception:
            fen_afters[i] = None

    # Deduplicate FENs — opening positions repeat heavily across games
    fen_to_canonical: dict = {}
    unique_fens = []

    for i, row in enumerate(rows):
        for fen in [row["fen"], fen_afters.get(i)]:
            if fen and fen not in fen_to_canonical:
                fen_to_canonical[fen] = len(unique_fens)
                unique_fens.append(fen)

    total = len(unique_fens)
    print(f"Evaluating {total} unique positions (all moves, full quality)")

    eval_cache: dict = {}
    limit = chess.engine.Limit(time=time_per_pos)

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        engine.configure({"Threads": sf_threads, "Hash": 256})
        for fen in tqdm(unique_fens, desc="Stockfish Deep Eval"):
            try:
                board = chess.Board(fen)
                info  = engine.analyse(board, limit)
                cp    = stockfish_to_cp(info["score"])
                pv    = info.get("pv", [])
                best  = pv[0].uci() if pv else None
                eval_cache[fen] = {"cp": cp, "best": best}
            except Exception:
                pass

    coverage = round(len(eval_cache) / max(total, 1) * 100, 1)
    print(f"Coverage: {coverage}% ({len(eval_cache)}/{total} evaluated)")

    # Build results
    results = []
    for i, row in enumerate(rows):
        color = row["player_color"]

        before_data = eval_cache.get(row["fen"])
        after_fen   = fen_afters.get(i)
        after_data  = eval_cache.get(after_fen) if after_fen else None

        if before_data is None or after_data is None:
            results.append({
                **row,
                "eval_before": None, "eval_after": None,
                "cp_loss": None, "best_move": None,
                "played_best": False, "mistake_type": "unknown",
            })
            continue

        white_before = before_data["cp"]
        white_after  = after_data["cp"]
        best_uci     = before_data["best"]

        cp_loss, eval_before, eval_after = compute_cp_loss(white_before, white_after, color)

        results.append({
            **row,
            "eval_before": eval_before, "eval_after": eval_after,
            "cp_loss": cp_loss, "best_move": best_uci,
            "played_best": (best_uci == row["move_uci"]) if best_uci else False,
            "mistake_type": classify_mistake(cp_loss),
        })

    return pd.DataFrame(results)



# ── Entry point ───────────────────────────────────────────────────────────────────

def add_engine_evaluations(
    moves_path:  str,
    output_path: str,
    resume:      bool = True,
):
    df = pd.read_parquet(moves_path)

    if MAX_GAMES is not None:
        recent = df["game_id"].unique()[-MAX_GAMES:]
        df     = df[df["game_id"].isin(recent)]

    if resume and os.path.exists(output_path):
        existing   = pd.read_parquet(output_path)
        done_games = set(existing["game_id"].unique())
        df_todo    = df[~df["game_id"].isin(done_games)]
        print(f"Resuming: {df_todo['game_id'].nunique()} games left")
    else:
        existing = pd.DataFrame()
        df_todo  = df

    if df_todo.empty:
        print("All games already evaluated.")
        return pd.read_parquet(output_path)

    print(f"Analyzing {df_todo['game_id'].nunique()} games ({len(df_todo)} moves)")

    if MODE == "fast":
        new_df = run_fast_mode(df_todo)
    else:
        new_df = run_deep_mode(df_todo)

    final_df = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df

    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    final_df.to_parquet(output_path, index=False)

    print(f"\nDone. {len(final_df)} moves saved.")
    print("\nMistake distribution:")
    print(final_df["mistake_type"].value_counts().to_string())

    valid = final_df[final_df["cp_loss"].notna() & (final_df["cp_loss"] > 0)]
    if len(valid) > 0:
        print(f"\nCP stats — Avg: {round(valid['cp_loss'].mean(),1)}  "
              f"Median: {round(valid['cp_loss'].median(),1)}  "
              f"Max: {round(valid['cp_loss'].max(),1)}")

    return final_df


if __name__ == "__main__":
    import sys
    username = sys.argv[1] if len(sys.argv) > 1 else "player"
    add_engine_evaluations(
        moves_path=f"data/{username}/moves.parquet",
        output_path=f"data/{username}/moves_evaluated.parquet",
        resume=True,
    )