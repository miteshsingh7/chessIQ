import requests
import os
import time
import json

def fetch_all_games(username: str, output_dir: str = "data/raw_pgn"):
    """
    Fetch all PGN games for a Chess.com username.
    Saves raw PGN files and metadata JSON per month.
    """
    os.makedirs(output_dir, exist_ok=True)

    headers = {"User-Agent": "ChessMistakeAnalyzer/1.0 contact@example.com"}

    # Step 1: Get list of monthly archive URLs
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    response = requests.get(archives_url, headers=headers)
    response.raise_for_status()
    archives = response.json().get("archives", [])

    print(f"Found {len(archives)} monthly archives for '{username}'")

    all_games = []

    for archive_url in archives:
        month_label = archive_url.split("/")[-2] + "_" + archive_url.split("/")[-1]
        pgn_path = os.path.join(output_dir, f"{username}_{month_label}.pgn")
        meta_path = os.path.join(output_dir, f"{username}_{month_label}_meta.json")

        # Avoid re-downloading
        if os.path.exists(pgn_path) and os.path.exists(meta_path):
            print(f"  Skipping {month_label} (already downloaded)")
            try:
                with open(meta_path) as f:
                    all_games.extend(json.load(f))
                continue
            except (json.JSONDecodeError, IOError):
                print(f"  Warning: corrupted metadata for {month_label}, re-downloading")
                os.remove(pgn_path)
                # Fall through to re-download

        if not os.path.exists(pgn_path):
            time.sleep(0.5)  # Be polite to the API
            try:
                games_response = requests.get(archive_url, headers=headers)
                games_response.raise_for_status()
                games = games_response.json().get("games", [])
            except requests.exceptions.RequestException as e:
                print(f"  Warning: Failed to fetch games for {month_label}: {e}")
                continue

            # Filter to rapid games only
            rapid_games = [g for g in games if g.get("time_class") == "rapid"]

            # Save PGN — rapid games only
            with open(pgn_path, "w") as f:
                for game in rapid_games:
                    if "pgn" in game:
                        f.write(game["pgn"] + "\n\n")

            # Save metadata (everything except raw PGN to keep it lean)
            meta = []
            for game in rapid_games:
                meta.append({
                    "url": game.get("url"),
                    "time_control": game.get("time_control"),
                    "end_time": game.get("end_time"),
                    "rated": game.get("rated"),
                    "time_class": game.get("time_class"),
                    "rules": game.get("rules"),
                    "white": {
                        "username": game.get("white", {}).get("username"),
                        "rating": game.get("white", {}).get("rating"),
                        "result": game.get("white", {}).get("result"),
                    },
                    "black": {
                        "username": game.get("black", {}).get("username"),
                        "rating": game.get("black", {}).get("rating"),
                        "result": game.get("black", {}).get("result"),
                    },
                })

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            all_games.extend(meta)
            print(f"  Saved {len(rapid_games)} rapid games from {month_label}")

    print(f"\nTotal games fetched: {len(all_games)}")
    return all_games


if __name__ == "__main__":
    USERNAME = "MITEX7"  # <-- Change this
    fetch_all_games(USERNAME)
