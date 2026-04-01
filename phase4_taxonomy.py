import chess
import pandas as pd
import os
from tqdm import tqdm
from constants import PIECE_VALUE, PIECE_NAMES, CP_CAP


# ── Tactic Detectors ──────────────────────────────────────────────────────────────

def detect_fork(board, move):
    b = board.copy(); b.push(move)
    piece = b.piece_at(move.to_square)
    if not piece: return False
    attacked = 0
    for sq in chess.SQUARES:
        t = b.piece_at(sq)
        if t and t.color != piece.color:
            if t.piece_type in [chess.QUEEN, chess.ROOK, chess.KING, chess.BISHOP, chess.KNIGHT]:
                if b.is_attacked_by(piece.color, sq):
                    attacked += 1
    return attacked >= 2


def detect_pin(board, move):
    """Detect if the move creates a pin — an opponent piece is on the same ray
    between the moving piece and the opponent king."""
    b = board.copy(); b.push(move)
    piece = b.piece_at(move.to_square)
    if not piece or piece.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
        return False
    opp = b.turn
    ksq = b.king(opp)
    if ksq is None:
        return False
    # Check if the piece on to_square and the king are on the same ray
    # with exactly one opponent piece in between
    ray_squares = list(chess.SquareSet.between(move.to_square, ksq))
    if not ray_squares:
        return False
    # Verify the piece can actually attack along this ray
    if not b.is_attacked_by(piece.color, ksq) and len(ray_squares) > 0:
        # Check if removing pieces in between would create attack
        opp_pieces_between = []
        for sq in ray_squares:
            p = b.piece_at(sq)
            if p and p.color == opp:
                opp_pieces_between.append((sq, p))
            elif p and p.color == piece.color:
                return False  # friendly piece blocks the ray
        if len(opp_pieces_between) == 1:
            # Remove the piece and check if king becomes attacked
            pin_sq, pin_piece = opp_pieces_between[0]
            b.remove_piece_at(pin_sq)
            if b.is_attacked_by(piece.color, ksq):
                b.set_piece_at(pin_sq, pin_piece)
                return True
            b.set_piece_at(pin_sq, pin_piece)
    return False


def detect_skewer(board, move):
    """Detect skewer: attacking a valuable piece with a less valuable piece behind it on the same line."""
    b = board.copy(); b.push(move)
    piece = b.piece_at(move.to_square)
    if not piece or piece.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
        return False
    opp = not piece.color
    # Look for high-value targets attacked by this piece
    for sq in chess.SQUARES:
        t = b.piece_at(sq)
        if t and t.color == opp and t.piece_type in [chess.KING, chess.QUEEN]:
            if b.is_attacked_by(piece.color, sq):
                # Check if there's a piece behind the target on the same line
                behind_squares = list(chess.SquareSet.between(move.to_square, sq))
                # Extend the ray beyond the target
                df_file = chess.square_file(sq) - chess.square_file(move.to_square)
                df_rank = chess.square_rank(sq) - chess.square_rank(move.to_square)
                # Normalize direction
                d_file = (1 if df_file > 0 else -1) if df_file != 0 else 0
                d_rank = (1 if df_rank > 0 else -1) if df_rank != 0 else 0
                # Walk beyond the target
                cur_file = chess.square_file(sq) + d_file
                cur_rank = chess.square_rank(sq) + d_rank
                while 0 <= cur_file < 8 and 0 <= cur_rank < 8:
                    behind_sq = chess.square(cur_file, cur_rank)
                    behind_piece = b.piece_at(behind_sq)
                    if behind_piece:
                        if behind_piece.color == opp:
                            return True  # opponent piece behind the target
                        break  # friendly piece blocks
                    cur_file += d_file
                    cur_rank += d_rank
    return False


def detect_back_rank(board, move):
    b = board.copy(); b.push(move)
    if b.is_checkmate(): return True
    if b.is_check():
        ksq = b.king(b.turn)
        if ksq and chess.square_rank(ksq) in [0, 7]:
            return True
    return False


def detect_discovered(board, move):
    piece = board.piece_at(move.from_square)
    if not piece: return False
    b = board.copy(); b.push(move)
    for sq in chess.SQUARES:
        t = b.piece_at(sq)
        if t and t.color != piece.color:
            if t.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                if b.is_attacked_by(piece.color, sq):
                    for att_sq in b.attackers(piece.color, sq):
                        att = b.piece_at(att_sq)
                        if att and att_sq != move.to_square:
                            return True
    return False


def detect_hanging(board, move, color):
    b = board.copy(); b.push(move)
    opp = not color
    for sq in chess.SQUARES:
        piece = b.piece_at(sq)
        if piece and piece.color == color:
            if piece.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                if b.attackers(opp, sq) and not b.attackers(color, sq):
                    return True, piece.piece_type
    return False, None


def detect_trapped_piece(board, move, color):
    """Piece has no safe squares to move to after the move."""
    b = board.copy(); b.push(move)
    opp = not color
    for sq in chess.SQUARES:
        piece = b.piece_at(sq)
        if piece and piece.color == color:
            if piece.piece_type in [chess.BISHOP, chess.KNIGHT, chess.ROOK]:
                if b.attackers(opp, sq):
                    # Check if piece has any safe escape
                    safe_squares = 0
                    b2 = b.copy()
                    b2.turn = color
                    for escape in b2.generate_legal_moves():
                        if escape.from_square == sq:
                            b3 = b2.copy()
                            b3.push(escape)
                            if not b3.attackers(opp, escape.to_square):
                                safe_squares += 1
                                break
                    if safe_squares == 0:
                        return True, piece.piece_type
    return False, None


def detect_weak_back_rank(board, move, color):
    """Player's own back rank is weak after the move."""
    b = board.copy(); b.push(move)
    opp = not color
    back_rank = 0 if color == chess.WHITE else 7
    king_sq   = b.king(color)
    if not king_sq: return False
    if chess.square_rank(king_sq) != back_rank: return False
    # Check if opponent rook/queen can invade back rank
    for sq in b.pieces(chess.ROOK, opp) | b.pieces(chess.QUEEN, opp):
        if b.is_attacked_by(color, sq): continue
        if chess.square_rank(sq) == back_rank:
            return True
    return False


def detect_pawn_fork_missed(board, move):
    """Missed pawn fork opportunity."""
    b = board.copy(); b.push(move)
    color = b.turn  # opponent after move — check what player missed
    opp   = not color
    for sq in board.pieces(chess.PAWN, opp):
        pawn = chess.Move(sq, sq + (8 if opp == chess.WHITE else -8))
        if pawn in board.legal_moves:
            b2 = board.copy(); b2.push(pawn)
            attacked = sum(
                1 for s in chess.SQUARES
                if b2.piece_at(s) and b2.piece_at(s).color == color
                and b2.piece_at(s).piece_type in [chess.QUEEN, chess.ROOK, chess.KNIGHT, chess.BISHOP]
                and b2.is_attacked_by(opp, s)
            )
            if attacked >= 2:
                return True
    return False


def detect_overloaded_piece(board, move, color):
    """Opponent's piece is overloaded — defending two things at once."""
    b = board.copy(); b.push(move)
    opp = not color
    for sq in chess.SQUARES:
        piece = b.piece_at(sq)
        if piece and piece.color == opp:
            defended = []
            for sq2 in chess.SQUARES:
                t = b.piece_at(sq2)
                if t and t.color == opp and sq2 != sq:
                    if b.is_attacked_by(opp, sq2) and sq in b.attackers(opp, sq2):
                        defended.append(sq2)
            if len(defended) >= 2:
                # Check if any of those defended pieces can be captured profitably
                for def_sq in defended:
                    def_piece = b.piece_at(def_sq)
                    if def_piece:
                        attackers = b.attackers(color, def_sq)
                        if attackers:
                            return True
    return False


def detect_zwischenzug(board, move, best_move):
    """
    Missed zwischenzug (in-between move) — instead of recapturing immediately,
    there was a stronger forcing move first.
    """
    if not best_move: return False
    b = board.copy()
    if best_move not in b.legal_moves: return False
    b.push(best_move)
    # If best move gives check but player played a quiet move
    played_capture = board.is_capture(move)
    best_gives_check = b.is_check()
    if best_gives_check and not played_capture:
        return True
    return False


def detect_sacrifice_missed(board, move, best_move, cp_loss):
    """Missed sacrifice — best move gives up material for compensation."""
    if not best_move or cp_loss < 150: return False
    b = board.copy()
    if best_move not in b.legal_moves: return False
    # Best move is a capture that loses material
    if b.is_capture(best_move):
        captured = b.piece_at(best_move.to_square)
        attacker = b.piece_at(best_move.from_square)
        if captured and attacker:
            if PIECE_VALUE.get(attacker.piece_type, 0) > PIECE_VALUE.get(captured.piece_type, 0) + 1:
                return True
    return False


# ── Phase Detection ───────────────────────────────────────────────────────────────

def get_phase(board, move_number):
    total = sum(
        len(board.pieces(pt, c)) * v
        for pt, v in [(chess.PAWN,1),(chess.KNIGHT,3),(chess.BISHOP,3),(chess.ROOK,5),(chess.QUEEN,9)]
        for c in [chess.WHITE, chess.BLACK]
    )
    has_q = bool(board.pieces(chess.QUEEN, chess.WHITE)) or bool(board.pieces(chess.QUEEN, chess.BLACK))
    if move_number <= 12: return "opening"
    elif total <= 20 or not has_q: return "endgame"
    return "middlegame"


# ── Master Classifier ─────────────────────────────────────────────────────────────

def classify_move(row, mode="fast") -> dict:
    result = {
        "mistake_category": "none",
        "tactic_type":      "none",
        "piece_lost":       "none",
        "mate_missed":      0,
    }

    cp_loss      = row.get("cp_loss", 0) or 0
    mistake_type = row.get("mistake_type", "good")

    if mistake_type in ["good", "unknown"] or cp_loss < 50:
        return result

    try:
        board      = chess.Board(row["fen"])
        color      = chess.WHITE if row["player_color"] == "white" else chess.BLACK
        played     = chess.Move.from_uci(row["move_uci"])
        best_uci   = row.get("best_move")
        best_move  = chess.Move.from_uci(best_uci) if best_uci else None
        move_num   = row.get("move_number", 20)
        board_after = board.copy()
        board_after.push(played)
    except Exception:
        result["mistake_category"] = "blunder_other" if cp_loss >= 300 else "mistake_other"
        return result

    # ── 1. Missed forced mate (using eval_before from Phase 2) ────────────
    eval_before = row.get("eval_before")
    if cp_loss >= 200 and eval_before is not None:
        # eval_before at CP_CAP means engine found a forced mate
        if abs(eval_before) >= CP_CAP:
            # Player had a winning forced mate but missed it
            result["mistake_category"] = "missed_mate"
            result["tactic_type"]      = "missed_mate"
            result["mate_missed"]      = 1
            return result

    # ── 2. Hanging piece ─────────────────────────────────────────────────────
    is_hang, hung = detect_hanging(board, played, color)
    if is_hang and cp_loss >= 150:
        result["mistake_category"] = "hanging_piece"
        result["tactic_type"]      = "hanging_piece"
        result["piece_lost"]       = PIECE_NAMES.get(hung, "piece")
        return result

    # ── 3. Trapped piece ─────────────────────────────────────────────────────
    if mode == "deep":
        is_trap, trap_piece = detect_trapped_piece(board, played, color)
        if is_trap and cp_loss >= 150:
            result["mistake_category"] = "trapped_piece"
            result["tactic_type"]      = f"trapped_{PIECE_NAMES.get(trap_piece,'piece')}"
            result["piece_lost"]       = PIECE_NAMES.get(trap_piece, "piece")
            return result

    # ── 4. Missed tactic on best move ────────────────────────────────────────
    if best_move and cp_loss >= 150 and not row.get("played_best", False):

        if detect_back_rank(board, best_move):
            result["mistake_category"] = "missed_tactic"
            result["tactic_type"]      = "back_rank_mate"
            return result

        if detect_fork(board, best_move):
            # Distinguish knight fork vs pawn fork vs piece fork
            piece = board.piece_at(best_move.from_square)
            fork_type = "knight_fork" if piece and piece.piece_type == chess.KNIGHT else \
                        "pawn_fork"   if piece and piece.piece_type == chess.PAWN   else "fork"
            result["mistake_category"] = "missed_tactic"
            result["tactic_type"]      = fork_type
            return result

        if detect_pin(board, best_move):
            result["mistake_category"] = "missed_tactic"
            result["tactic_type"]      = "pin"
            return result

        if detect_skewer(board, best_move):
            result["mistake_category"] = "missed_tactic"
            result["tactic_type"]      = "skewer"
            return result

        if detect_discovered(board, best_move):
            result["mistake_category"] = "missed_tactic"
            result["tactic_type"]      = "discovered_attack"
            return result

        if mode == "deep":
            if detect_zwischenzug(board, played, best_move):
                result["mistake_category"] = "missed_tactic"
                result["tactic_type"]      = "zwischenzug"
                return result

            if detect_sacrifice_missed(board, played, best_move, cp_loss):
                result["mistake_category"] = "missed_tactic"
                result["tactic_type"]      = "missed_sacrifice"
                return result

            if detect_overloaded_piece(board, played, color):
                result["mistake_category"] = "missed_tactic"
                result["tactic_type"]      = "overloaded_piece"
                return result

        # Distinguish calculation error by game phase
        phase = get_phase(board, move_num)
        calc_type = {
            "opening":    "opening_calculation",
            "middlegame": "middlegame_calculation",
            "endgame":    "endgame_calculation",
        }.get(phase, "calculation_error")

        result["mistake_category"] = "missed_tactic"
        result["tactic_type"]      = calc_type
        return result

    # ── 5. Time pressure ─────────────────────────────────────────────────────
    if row.get("time_pressure") and cp_loss >= 300:
        result["mistake_category"] = "time_pressure_blunder"
        result["tactic_type"]      = "time_pressure"
        return result

    # ── 6. King safety ────────────────────────────────────────────────────────
    if not row.get("castled", True) and row.get("open_files_near_king", 0) >= 2 and cp_loss >= 100:
        result["mistake_category"] = "king_safety_error"
        result["tactic_type"]      = "king_exposed"
        return result

    if mode == "deep":
        if detect_weak_back_rank(board, played, color) and cp_loss >= 100:
            result["mistake_category"] = "king_safety_error"
            result["tactic_type"]      = "weak_back_rank"
            return result

    # ── 7. Phase errors ───────────────────────────────────────────────────────
    phase = get_phase(board, row.get("move_number", 20))
    if phase == "opening" and cp_loss >= 100:
        result["mistake_category"] = "opening_error"
        result["tactic_type"]      = "opening_principle"
        return result
    if phase == "endgame" and cp_loss >= 100:
        # Distinguish endgame types
        has_pawns  = bool(board.pieces(chess.PAWN, color))
        has_rooks  = bool(board.pieces(chess.ROOK, color))
        eg_type    = "rook_endgame" if has_rooks else "pawn_endgame" if has_pawns else "endgame_technique"
        result["mistake_category"] = "endgame_error"
        result["tactic_type"]      = eg_type
        return result

    if (row.get("isolated_pawns", 0) >= 2 or row.get("doubled_pawns", 0) >= 2) and cp_loss >= 100:
        result["mistake_category"] = "pawn_structure_error"
        result["tactic_type"]      = "weak_pawns"
        return result

    result["mistake_category"] = "blunder_other" if cp_loss >= 300 else "mistake_other"
    return result


# ── Main ──────────────────────────────────────────────────────────────────────────

def add_detailed_taxonomy(
    features_path: str,
    output_path:   str,
    mode:          str = "fast",
):
    df = pd.read_parquet(features_path)
    print(f"Classifying {len(df)} moves (mode: {mode})...")

    tqdm.pandas(desc=f"Classifying moves ({mode})")
    class_df = df.progress_apply(lambda row: classify_move(row, mode), axis=1, result_type='expand')

    final_df = pd.concat([df.reset_index(drop=True), class_df], axis=1)

    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    final_df.to_parquet(output_path, index=False)

    mistakes = final_df[final_df["mistake_category"] != "none"]
    print(f"\nTotal mistakes: {len(mistakes)}")
    print("\nCategory breakdown:")
    print(mistakes["mistake_category"].value_counts().to_string())
    print("\nTactic type breakdown:")
    print(mistakes["tactic_type"].value_counts().to_string())

    return final_df


if __name__ == "__main__":
    import sys
    username = sys.argv[1] if len(sys.argv) > 1 else "player"
    mode     = sys.argv[2] if len(sys.argv) > 2 else "fast"
    add_detailed_taxonomy(
        features_path=f"data/{username}/moves_features.parquet",
        output_path=f"data/{username}/moves_categorized.parquet",
        mode=mode,
    )