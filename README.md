# ♟️ ChessLens — AI Chess Coach & Analytics

**ChessLens** is an intelligent AI chess coaching and analysis application powered by **Stockfish**, **Python**, and **Streamlit**. It automatically fetches games from Chess.com, parses PGNs, runs deep engine evaluations, classifies tactical blunder taxonomies, and generates personalized training plans.

---

## ✨ Features

- **🎨 Solid Matte UI**: Elegant dark matte finish (`#11141C`) with off-white (`#F4F2EC`) typography and subtle chess design motifs.
- **♟️ Interactive Blunder Inspector**: SVG position viewer with red/green move arrows showing **Played Move** vs. **Stockfish Recommended Move**.
- **⚡ Fast & 🔬 Deep Modes**:
  - **Fast Mode**: Evaluates suspicious moves for quick weekly check-ins.
  - **Deep Mode**: Comprehensive depth-18 Stockfish evaluation classifying 15+ blunder taxonomies (forks, pins, skewers, zwischenzugs, trapped pieces).
- **📊 Plotly Visualizations**: Dynamic tactical skill breakdown, phase mistake rates, and centipawn loss distribution graphs.
- **📅 Interactive Training Hub**: Custom weekly training calendar with checkable drills and direct links to themed Lichess puzzle filters.

---

## 📂 Repository Architecture

```text
chessIQ/
├── README.md                  # Project overview & documentation
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker container deployment config
├── .gitignore                 # Git ignore rules
│
├── src/                       # 📦 Core Stockfish Analysis Pipeline
│   ├── __init__.py
│   ├── constants.py           # Piece values, evaluation caps, taxonomy definitions
│   ├── phase1_fetch_games.py  # Chess.com API game fetcher
│   ├── phase1_parse_pgn.py    # PGN parsing & move sequence parser
│   ├── phase2_engine_eval.py  # Stockfish engine evaluation engine
│   ├── phase3_feature_engineering.py  # Position feature extraction
│   ├── phase4_taxonomy.py     # Blunder & tactical mistake classification
│   ├── phase5_analytics.py    # Data aggregations & report plotting
│   ├── phase6_recommendations.py# Coaching report & drill generator
│   └── phase7_ml_models.py    # ML evaluation models
│
└── scripts/                   # 🛠️ Utility scripts & HTML templates
    ├── apply_footers.py
    ├── apply_synopsis_footers.py
    ├── debug_report.py
    └── ChessIQ_A2_Poster.html
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/miteshsingh7/chessIQ.git
cd chessIQ
pip install -r requirements.txt
```

### 2. Install Stockfish Engine
Download Stockfish from [StockfishChess.org](https://stockfishchess.org/download/) and paste the executable path in the app settings, or ensure `stockfish` is in your `PATH`.

### 3. Launch the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🐳 Docker Deployment

Build and run using Docker:
```bash
docker build -t chesslens .
docker run -p 8501:8501 chesslens
```

---

## 📜 License
[MIT License](LICENSE)

<!-- note --> (1)

<!-- note --> (3)

<!-- note --> (4)

<!-- note --> (7)
