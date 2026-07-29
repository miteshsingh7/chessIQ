"""Shared constants for the ChessLens pipeline."""

import chess

PIECE_VALUE = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

PIECE_NAMES = {
    chess.QUEEN: "queen", chess.ROOK: "rook",
    chess.BISHOP: "bishop", chess.KNIGHT: "knight", chess.PAWN: "pawn",
}

# Evaluation settings
CP_CAP = 900  # cap at 900 — anything above is effectively a game-ending blunder
