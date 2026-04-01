import chess
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from constants import PIECE_VALUE


# ── Game Phase Detection ────────────────────────────────────────────────────────

PIECE_VALUES = PIECE_VALUE  # alias for backward compatibility

def get_total_material(board: chess.Board) -> int:
    total = 0
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        total += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        total += len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
    return total


def get_game_phase(board: chess.Board, move_number: int) -> str:
    has_white_queen = bool(board.pieces(chess.QUEEN, chess.WHITE))
    has_black_queen = bool(board.pieces(chess.QUEEN, chess.BLACK))
    total_material = get_total_material(board)

    if move_number <= 12:
        return "opening"
    elif total_material <= 20 or (not has_white_queen and not has_black_queen):
        return "endgame"
    else:
        return "middlegame"


# ── King Safety ─────────────────────────────────────────────────────────────────

def is_castled(board: chess.Board, color: chess.Color) -> bool:
    """Heuristic: king is not on e-file and not too far from corner."""
    king_sq = board.king(color)
    if king_sq is None:
        return False
    king_file = chess.square_file(king_sq)
    # Kingside castle: file g (6), Queenside: file c (2)
    return king_file in [1, 2, 6, 7]


def count_open_files_near_king(board: chess.Board, color: chess.Color) -> int:
    """Count files adjacent to king that have no pawns (open = dangerous)."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    king_file = chess.square_file(king_sq)
    open_files = 0
    for f in range(max(0, king_file - 1), min(8, king_file + 2)):
        white_pawns_on_file = any(
            chess.square_file(sq) == f for sq in board.pieces(chess.PAWN, chess.WHITE)
        )
        black_pawns_on_file = any(
            chess.square_file(sq) == f for sq in board.pieces(chess.PAWN, chess.BLACK)
        )
        if not white_pawns_on_file and not black_pawns_on_file:
            open_files += 1
    return open_files


# ── Pawn Structure ──────────────────────────────────────────────────────────────

def count_doubled_pawns(board: chess.Board, color: chess.Color) -> int:
    doubled = 0
    for file in range(8):
        pawns_on_file = sum(
            1 for sq in board.pieces(chess.PAWN, color)
            if chess.square_file(sq) == file
        )
        if pawns_on_file > 1:
            doubled += pawns_on_file - 1
    return doubled


def count_isolated_pawns(board: chess.Board, color: chess.Color) -> int:
    isolated = 0
    pawn_files = set(chess.square_file(sq) for sq in board.pieces(chess.PAWN, color))
    for f in pawn_files:
        neighbors = {f - 1, f + 1} & pawn_files
        if not neighbors:
            isolated += 1
    return isolated


def count_passed_pawns(board: chess.Board, color: chess.Color) -> int:
    """A pawn is passed if no enemy pawn can block or capture it."""
    passed = 0
    opponent = not color
    opp_pawn_files = set(chess.square_file(sq) for sq in board.pieces(chess.PAWN, opponent))
    opp_pawns = list(board.pieces(chess.PAWN, opponent))

    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        blocking_files = {f - 1, f, f + 1} & set(range(8))
        is_passed = True
        for opp_sq in opp_pawns:
            opp_f = chess.square_file(opp_sq)
            opp_r = chess.square_rank(opp_sq)
            if opp_f in blocking_files:
                if color == chess.WHITE and opp_r > r:
                    is_passed = False
                    break
                elif color == chess.BLACK and opp_r < r:
                    is_passed = False
                    break
        if is_passed:
            passed += 1
    return passed


# ── Mobility ────────────────────────────────────────────────────────────────────

def get_mobility(board: chess.Board, color: chess.Color) -> int:
    """Count number of legal moves available for the given color."""
    board_copy = board.copy()
    board_copy.turn = color
    return board_copy.legal_moves.count()


# ── Material Count ──────────────────────────────────────────────────────────────

def get_material_features(board: chess.Board, color: chess.Color) -> dict:
    opp = not color
    return {
        "pawns": len(board.pieces(chess.PAWN, color)),
        "knights": len(board.pieces(chess.KNIGHT, color)),
        "bishops": len(board.pieces(chess.BISHOP, color)),
        "rooks": len(board.pieces(chess.ROOK, color)),
        "queens": len(board.pieces(chess.QUEEN, color)),
        "opp_pawns": len(board.pieces(chess.PAWN, opp)),
        "opp_knights": len(board.pieces(chess.KNIGHT, opp)),
        "opp_bishops": len(board.pieces(chess.BISHOP, opp)),
        "opp_rooks": len(board.pieces(chess.ROOK, opp)),
        "opp_queens": len(board.pieces(chess.QUEEN, opp)),
        "material_balance": get_total_material(board),
    }


# ── Master Feature Extractor ────────────────────────────────────────────────────

def extract_features_for_row(row: pd.Series) -> dict:
    try:
        board = chess.Board(row["fen"])
        color = chess.WHITE if row["player_color"] == "white" else chess.BLACK
        move_number = row["move_number"]

        material = get_material_features(board, color)
        phase = get_game_phase(board, move_number)
        castled = is_castled(board, color)
        open_files = count_open_files_near_king(board, color)
        doubled = count_doubled_pawns(board, color)
        isolated = count_isolated_pawns(board, color)
        passed = count_passed_pawns(board, color)
        mobility = get_mobility(board, color)

        # Time pressure
        time_left = row.get("time_left")
        time_pressure = (time_left is not None and time_left < 60)

        return {
            "phase": phase,
            "castled": castled,
            "open_files_near_king": open_files,
            "doubled_pawns": doubled,
            "isolated_pawns": isolated,
            "passed_pawns": passed,
            "mobility": mobility,
            "time_pressure": time_pressure,
            **material,
        }

    except Exception:
        return {
            "phase": "unknown", "castled": None, "open_files_near_king": None,
            "doubled_pawns": None, "isolated_pawns": None, "passed_pawns": None,
            "mobility": None, "time_pressure": None,
            "pawns": None, "knights": None, "bishops": None,
            "rooks": None, "queens": None, "opp_pawns": None,
            "opp_knights": None, "opp_bishops": None, "opp_rooks": None,
            "opp_queens": None, "material_balance": None,
        }


def add_features(
    evaluated_path: str = "data/processed/moves_evaluated.parquet",
    output_path: str = "data/processed/moves_features.parquet"
):
    df = pd.read_parquet(evaluated_path)
    print(f"Extracting features for {len(df)} moves...")

    tqdm.pandas(desc="Extracting features")
    features_df = df.progress_apply(extract_features_for_row, axis=1, result_type='expand')
    final_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    print(f"Saved {len(final_df)} rows to {output_path}")
    return final_df


if __name__ == "__main__":
    df = add_features()
    print(df[["move_san", "phase", "castled", "time_pressure", "mistake_type"]].head(20))
