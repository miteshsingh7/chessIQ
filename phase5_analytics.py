import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def get_output_dir(username: str) -> str:
    path = f"data/analytics"
    os.makedirs(path, exist_ok=True)
    return path


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[df["mistake_type"] != "unknown"]


def tactic_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    missed = df[df["mistake_category"] == "missed_tactic"]
    if missed.empty:
        return pd.DataFrame()
    counts = missed["tactic_type"].value_counts()
    pct = (counts / len(missed) * 100).round(1)
    return pd.DataFrame({"count": counts, "percent_%": pct})


def hanging_piece_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    hanging = df[df["mistake_category"] == "hanging_piece"]
    if hanging.empty:
        return pd.DataFrame()
    counts = hanging["piece_lost"].value_counts()
    pct = (counts / len(hanging) * 100).round(1)
    return pd.DataFrame({"count": counts, "percent_%": pct})


def worst_phase(df: pd.DataFrame):
    mistakes = df[df["mistake_type"].isin(["mistake", "blunder"])]
    phase_rate = {}
    for phase in ["opening", "middlegame", "endgame"]:
        total = len(df[df["phase"] == phase])
        errs = len(mistakes[mistakes["phase"] == phase])
        if total > 0:
            phase_rate[phase] = round(errs / total * 100, 1)
    if not phase_rate:
        return "middlegame", {}
    return max(phase_rate, key=phase_rate.get), phase_rate


def worst_move_range(df: pd.DataFrame) -> dict:
    mistakes = df[df["mistake_type"].isin(["mistake", "blunder"])].copy()
    bins = [0, 10, 20, 30, 40, 100]
    labels = ["1-10", "11-20", "21-30", "31-40", "40+"]
    mistakes["move_bucket"] = pd.cut(mistakes["move_number"], bins=bins, labels=labels)
    df2 = df.copy()
    df2["move_bucket"] = pd.cut(df2["move_number"], bins=bins, labels=labels)
    counts = mistakes["move_bucket"].value_counts().sort_index()
    totals = df2["move_bucket"].value_counts().sort_index()
    rate = (counts / totals * 100).round(1).fillna(0)
    worst = rate.idxmax()
    return {"worst_range": worst, "rate_%": float(rate[worst]), "all_ranges": rate.to_dict()}


def time_pressure_analysis(df: pd.DataFrame) -> dict:
    if "time_pressure" not in df.columns:
        return {}
    tp = df[df["time_pressure"] == True]
    normal = df[df["time_pressure"] == False]

    def blunder_rate(s):
        return round(s[s["mistake_type"] == "blunder"].shape[0] / max(len(s), 1) * 100, 1)

    return {
        "blunder_rate_under_pressure_%": blunder_rate(tp),
        "blunder_rate_normal_%": blunder_rate(normal),
        "multiplier": round(blunder_rate(tp) / max(blunder_rate(normal), 0.1), 1),
        "moves_under_pressure": len(tp),
    }


def missed_mate_analysis(df: pd.DataFrame) -> dict:
    missed_mates = df[df["mistake_category"] == "missed_mate"]
    if missed_mates.empty:
        return {"total_missed_mates": 0, "missed_mate_in_1": 0, "missed_mate_in_2": 0, "missed_mate_in_3plus": 0}
    return {
        "total_missed_mates": len(missed_mates),
        "missed_mate_in_1": len(missed_mates[missed_mates["tactic_type"] == "mate_in_1"]),
        "missed_mate_in_2": len(missed_mates[missed_mates["tactic_type"] == "mate_in_2"]),
        "missed_mate_in_3plus": len(missed_mates[missed_mates["mate_missed"] >= 3]),
    }


def win_loss_mistake_diff(df: pd.DataFrame) -> dict:
    wins = df[df["result"] == "win"]
    losses = df[df["result"] == "loss"]
    return {
        "avg_cp_loss_in_wins": round(wins["cp_loss"].mean(), 1),
        "avg_cp_loss_in_losses": round(losses["cp_loss"].mean(), 1),
        "blunders_per_100_in_losses": round(losses[losses["mistake_type"] == "blunder"].shape[0] / max(len(losses), 1) * 100, 1),
        "blunders_per_100_in_wins": round(wins[wins["mistake_type"] == "blunder"].shape[0] / max(len(wins), 1) * 100, 1),
    }


def print_deep_report(df: pd.DataFrame, username: str):
    mistakes = df[df["mistake_category"] != "none"]
    total_mistakes = len(mistakes)
    total_moves = len(df)

    print("\n" + "=" * 65)
    print(f"  CHESSLENS DEEP ANALYSIS — {username.upper()}")
    print("=" * 65)
    print(f"\n  Total moves   : {total_moves}")
    print(f"  Total mistakes: {total_mistakes}")
    print(f"  Mistakes/100  : {round(total_mistakes/max(total_moves,1)*100,1)}")
    print(f"  Avg CP loss   : {round(df['cp_loss'].mean(),1)}")

    worst, phase_rates = worst_phase(df)
    print(f"\n  Worst phase: {worst.upper()}")
    for phase, rate in phase_rates.items():
        print(f"    {phase:<12}: {rate}%")

    tdf = tactic_breakdown(df)
    if not tdf.empty:
        print(f"\n  Missed tactics:")
        for tactic, row in tdf.iterrows():
            print(f"    {tactic:<25}: {row['count']} times ({row['percent_%']}%)")

    hdf = hanging_piece_breakdown(df)
    if not hdf.empty:
        print(f"\n  Pieces hung:")
        for piece, row in hdf.iterrows():
            print(f"    {piece:<15}: {row['count']} times")

    mate_data = missed_mate_analysis(df)
    if mate_data["total_missed_mates"] > 0:
        print(f"\n  Missed mates: {mate_data['total_missed_mates']} total")
        print(f"    Mate in 1: {mate_data['missed_mate_in_1']}")
        print(f"    Mate in 2: {mate_data['missed_mate_in_2']}")
        print(f"    Mate in 3+: {mate_data['missed_mate_in_3plus']}")

    tp = time_pressure_analysis(df)
    if tp:
        print(f"\n  Time pressure: {tp['multiplier']}x more blunders under 60 seconds")

    print("=" * 65)


def plot_focused_dashboard(df: pd.DataFrame, username: str):
    output_dir = get_output_dir(username)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"ChessLens — {username}", fontsize=16, fontweight="bold")

    # 1. Tactic types missed
    ax = axes[0][0]
    tdf = tactic_breakdown(df)
    if not tdf.empty:
        tdf["percent_%"].plot(kind="barh", ax=ax, color="#5C6BC0")
        ax.set_title("Tactics You Miss Most")
        ax.set_xlabel("% of missed tactics")
        ax.invert_yaxis()
    else:
        ax.set_title("No Missed Tactics")

    # 2. Hanging pieces
    ax = axes[0][1]
    hdf = hanging_piece_breakdown(df)
    if not hdf.empty:
        colors_map = {"queen": "#E53935", "rook": "#FB8C00", "bishop": "#43A047", "knight": "#1E88E5"}
        bar_colors = [colors_map.get(p, "#999") for p in hdf.index]
        hdf["count"].plot(kind="bar", ax=ax, color=bar_colors)
        ax.set_title("Pieces You Hang Most")
        ax.tick_params(axis="x", rotation=0)
    else:
        ax.set_title("No Hanging Pieces")

    # 3. Mistake rate by move number
    ax = axes[0][2]
    move_data = worst_move_range(df)
    ranges = list(move_data["all_ranges"].keys())
    rates = list(move_data["all_ranges"].values())
    ax.bar(ranges, rates, color="#EF5350")
    ax.set_title("Mistake Rate by Move Number")
    ax.set_ylabel("% mistake rate")

    # 4. Mistake rate by phase
    ax = axes[1][0]
    _, phase_rates = worst_phase(df)
    ax.bar(list(phase_rates.keys()), list(phase_rates.values()),
           color=["#42A5F5", "#FF7043", "#66BB6A"])
    ax.set_title("Mistake Rate by Phase (%)")
    ax.set_ylabel("% mistakes")

    # 5. Time pressure
    ax = axes[1][1]
    tp = time_pressure_analysis(df)
    if tp:
        vals = [tp["blunder_rate_normal_%"], tp["blunder_rate_under_pressure_%"]]
        bars = ax.bar(["Normal", "Under 60s"], vals, color=["#26C6DA", "#FF7043"])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{val}%", ha="center", fontweight="bold")
        ax.set_title(f"Blunder Rate: {tp['multiplier']}x Worse Under Pressure")

    # 6. Category overview
    ax = axes[1][2]
    cats = df[df["mistake_category"] != "none"]["mistake_category"].value_counts().head(6)
    if not cats.empty:
        cats.plot(kind="barh", ax=ax, color="#AB47BC")
        ax.set_title("Mistake Categories Overview")
        ax.invert_yaxis()

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{username}_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {save_path}")


if __name__ == "__main__":
    import sys
    username = sys.argv[1] if len(sys.argv) > 1 else "player"
    df = load_data(f"data/{username}/moves_categorized.parquet")
    print_deep_report(df, username)
    plot_focused_dashboard(df, username)