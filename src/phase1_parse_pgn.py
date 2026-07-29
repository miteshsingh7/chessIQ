import chess
import chess.pgn
import re
import os
import glob
import pandas as pd
import json
from io import StringIO


def parse_clock(comment: str):
    """Extract clock time in seconds from PGN comment like [%clk 0:05:23]"""
    match = re.search(r'\[%clk\s+(\d+):(\d+):(\d+)\]', comment)
    if match:
        h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return h * 3600 + m * 60 + s
    return None


def get_game_result(game, player_username: str):
    """Return 'win', 'loss', or 'draw' from the perspective of the player."""
    result = game.headers.get("Result", "*")
    white = game.headers.get("White", "").lower()
    black = game.headers.get("Black", "").lower()
    player = player_username.lower()

    if result == "1-0":
        return "win" if white == player else "loss"
    elif result == "0-1":
        return "win" if black == player else "loss"
    elif result == "1/2-1/2":
        return "draw"
    return "unknown"


def parse_pgn_file(pgn_path: str, player_username: str):
    """
    Parse a PGN file and return a list of move-level dicts.
    Each dict = one move made by the tracked player.
    """
    rows = []
    game_id = 0

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    pgn_io = StringIO(content)

    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break

        game_id += 1
        headers = game.headers
        white = headers.get("White", "").lower()
        black = headers.get("Black", "").lower()
        player = player_username.lower()

        if player not in [white, black]:
            continue  # game doesn't involve the player

        player_color = chess.WHITE if white == player else chess.BLACK
        result = get_game_result(game, player_username)
        time_control = headers.get("TimeControl", "?")
        eco = headers.get("ECO", "?")
        opening = headers.get("Opening", "?")

        board = game.board()
        move_number = 0

        for node in game.mainline():
            move = node.move
            fen_before = board.fen()
            san = board.san(move)
            color = board.turn  # who is making this move

            clock_time = parse_clock(node.comment) if node.comment else None
            move_number += 1

            if color == player_color:
                rows.append({
                    "game_id": f"{os.path.basename(pgn_path)}_{game_id}",
                    "move_number": move_number,
                    "fen": fen_before,
                    "move_san": san,
                    "move_uci": move.uci(),
                    "time_left": clock_time,
                    "player_color": "white" if player_color == chess.WHITE else "black",
                    "result": result,
                    "time_control": time_control,
                    "eco": eco,
                    "opening": opening,
                    "white_player": headers.get("White", ""),
                    "black_player": headers.get("Black", ""),
                    "white_elo": headers.get("WhiteElo", None),
                    "black_elo": headers.get("BlackElo", None),
                })

            board.push(move)

    return rows


def parse_all_pgn_files(pgn_dir: str, player_username: str, output_path: str = "data/processed/moves.parquet", max_games: int = None):
    """Parse PGN files in directory and save to parquet.

    Files are processed in chronological order (oldest first) so that
    when max_games is set the most-recent N games are kept.
    """
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    all_rows = []

    # Sort files chronologically: MITEX7_2024_03.pgn → (2024, 03) order
    pgn_files = sorted(
        glob.glob(os.path.join(pgn_dir, f"{player_username}_*.pgn")),
        key=lambda p: os.path.basename(p).replace(".pgn", "").split("_")[-2:]
    )
    print(f"Found {len(pgn_files)} PGN files to parse")

    # If max_games set, we still need to walk newest-first to know when to stop.
    # Parse newest files first, stop once we have enough games, then reverse rows.
    game_count = 0
    if max_games is not None:
        pgn_files_to_scan = list(reversed(pgn_files))  # newest first
        temp_rows = []
        for pgn_file in pgn_files_to_scan:
            print(f"  Parsing {os.path.basename(pgn_file)}...")
            rows = parse_pgn_file(pgn_file, player_username)
            # Unique game IDs in this file
            game_ids_here = list(dict.fromkeys(r["game_id"] for r in rows))
            if game_count + len(game_ids_here) >= max_games:
                # Only keep enough games to reach max_games
                needed = max_games - game_count
                keep_ids = set(game_ids_here[:needed])
                rows = [r for r in rows if r["game_id"] in keep_ids]
                temp_rows.extend(rows)
                game_count += len(keep_ids)
                print(f"    -> {len(rows)} player moves extracted (capped at {max_games} games total)")
                break
            temp_rows.extend(rows)
            game_count += len(game_ids_here)
            print(f"    -> {len(rows)} player moves extracted ({game_count} games so far)")
        # Reverse so rows are in chronological order
        # We collected newest-first, so just use as-is (Phase 2 still picks [-MAX_GAMES:])
        all_rows = temp_rows
    else:
        for pgn_file in pgn_files:
            print(f"  Parsing {os.path.basename(pgn_file)}...")
            rows = parse_pgn_file(pgn_file, player_username)
            all_rows.extend(rows)
            print(f"    -> {len(rows)} player moves extracted")

    df = pd.DataFrame(all_rows)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df)} total moves to {output_path}")
    return df


if __name__ == "__main__":
    USERNAME = "your_chess_username"  # <-- Change this
    df = parse_all_pgn_files(
        pgn_dir="data/raw_pgn",
        player_username=USERNAME,
        output_path="data/processed/moves.parquet"
    )
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(df.dtypes)