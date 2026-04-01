# ♟ ChessLens

**Local chess game analyzer powered by Stockfish.** Analyzes your Chess.com games, detects mistakes and tactical patterns, and gives you a personalized weekly training plan.

---

## 🐳 Quick Start — Docker (recommended)

> No Python or Stockfish install needed. Just Docker.

```bash
# 1. Clone the repo
git clone https://github.com/your-username/chesslens.git
cd chesslens

# 2. Build the image (one time, ~2 min)
docker build -t chesslens .

# 3. Run it (data persists across restarts via a named volume)
docker run -p 8501:8501 -v chesslens_data:/app/data chesslens

# 4. Open in your browser
open http://localhost:8501   # Mac
start http://localhost:8501  # Windows
```

To stop: `Ctrl+C`  
To restart: just run step 3 again (your data persists in the `chesslens_data` volume).

---

## 🐍 Local Python Setup (alternative)

**Requirements:** Python 3.10+, Stockfish installed

```bash
# Install Stockfish
brew install stockfish          # macOS (Homebrew)
sudo apt install stockfish      # Ubuntu/Debian
# Windows: download from https://stockfishchess.org/download/

# Install Python dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

---

## 🎮 How to Use

1. Enter your **Chess.com username**
2. Choose a mode:
   - ⚡ **Fast Mode** — ~2 min for 50 games. Local Stockfish on suspicious moves only.
   - 🔬 **Deep Mode** — ~7 min for 50 games. Full Stockfish evaluation on every move.
3. Click **Analyze** and wait for the 7-phase pipeline to complete
4. Get your personalized coaching report with:
   - Your most common mistake types (blunders, inaccuracies, tactical misses)
   - Weakest opening, phase, and move range
   - A specific weekly training plan

---

## ⚙️ Configuration

Settings are saved to `chesslens_config.json` automatically.

| Setting | Description |
|---------|-------------|
| `max_games` | Number of recent games to analyze (50–200) |
| `mode` | `"fast"` or `"deep"` |
| `num_workers` | Stockfish internal threads (more = faster, up to your CPU core count) |
| `stockfish_path` | Override Stockfish path (useful on Windows) |

**Windows users:** If Stockfish isn't found automatically, paste the path to `stockfish.exe` in the **⚙️ Stockfish Path** expander in the sidebar.

---

## 🔧 Pipeline Overview

| Phase | What it does |
|-------|-------------|
| 1a | Fetches PGN games from Chess.com API |
| 1b | Parses PGN into a move-level DataFrame |
| 2 | Stockfish engine evaluation (the main analysis) |
| 3 | Feature engineering (game phase, king safety, material) |
| 4 | Mistake taxonomy (fork, pin, hanging piece, back rank…) |
| 5 | Analytics and charts |
| 6 | Personalized coaching recommendations |
| 7 | ML-based pattern detection |

---

## 📦 Requirements

- Python 3.10+
- Stockfish 15+ (installed separately or via Docker)
- See `requirements.txt` for Python packages
