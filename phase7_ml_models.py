import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Feature Preparation ─────────────────────────────────────────────────────────

FEATURE_COLS = [
    "move_number",
    "time_left",
    "time_pressure",
    "pawns", "knights", "bishops", "rooks", "queens",
    "opp_pawns", "opp_knights", "opp_bishops", "opp_rooks", "opp_queens",
    "material_balance",
    "castled",
    "open_files_near_king",
    "doubled_pawns",
    "isolated_pawns",
    "passed_pawns",
    "mobility",
]

PHASE_ENCODING = {"opening": 0, "middlegame": 1, "endgame": 2, "unknown": 1}
COLOR_ENCODING = {"white": 0, "black": 1}


def prepare_features(df: pd.DataFrame):
    """Clean and encode features for ML."""
    df = df.copy()

    # Encode categoricals
    df["phase_enc"] = df["phase"].map(PHASE_ENCODING).fillna(1)
    df["color_enc"] = df["player_color"].map(COLOR_ENCODING).fillna(0)

    # Target: is this move a blunder?
    df["is_blunder"] = (df["mistake_type"] == "blunder").astype(int)

    # Boolean to int
    for col in ["castled", "time_pressure"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Fill missing
    feature_cols = FEATURE_COLS + ["phase_enc", "color_enc"]
    existing_cols = [c for c in feature_cols if c in df.columns]
    X = df[existing_cols].fillna(0)
    y = df["is_blunder"]

    return X, y, existing_cols


# ── Model Training ──────────────────────────────────────────────────────────────

def train_blunder_predictor(
    data_path: str = "data/processed/moves_features.parquet",
    model_name: str = "random_forest"
):
    print("Loading data...")
    df = pd.read_parquet(data_path)
    df = df[df["mistake_type"] != "unknown"].dropna(subset=["cp_loss"])

    X, y, feature_names = prepare_features(df)

    print(f"Dataset: {len(X)} samples | Blunder rate: {y.mean()*100:.1f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Choose model ──────────────────────────────────────────────────────────
    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",  # handles imbalanced blunder rate
            random_state=42,
            n_jobs=-1
        )
    elif model_name == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
    elif model_name == "logistic_regression":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
        ])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # ── Cross-validation ──────────────────────────────────────────────────────
    print(f"\nTraining {model_name}...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"CV AUC-ROC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Final fit ─────────────────────────────────────────────────────────────
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{'─'*50}")
    print("TEST SET RESULTS")
    print(f"{'─'*50}")
    print(classification_report(y_test, y_pred, target_names=["Not Blunder", "Blunder"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

    # ── Feature importance ────────────────────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        plot_feature_importance(model.feature_importances_, feature_names, model_name)

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, f"blunder_predictor_{model_name}.joblib")
    joblib.dump({"model": model, "features": feature_names}, model_path)
    print(f"\nModel saved to {model_path}")

    return model, feature_names


def plot_feature_importance(importances, feature_names, model_name):
    idx = np.argsort(importances)[::-1]
    top_n = min(15, len(feature_names))
    plt.figure(figsize=(10, 5))
    plt.bar(range(top_n), importances[idx[:top_n]], color="#5C6BC0")
    plt.xticks(range(top_n), [feature_names[i] for i in idx[:top_n]], rotation=45, ha="right")
    plt.title(f"Feature Importance — {model_name}")
    plt.tight_layout()
    save_dir = os.path.join("data", "analytics")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"feature_importance_{model_name}.png"), dpi=120)
    plt.close()
    print(f"Feature importance chart saved")


# ── Inference ───────────────────────────────────────────────────────────────────

def predict_blunder_probability(fen_features: dict, model_path: str) -> float:
    """
    Given a dict of position features, predict probability of a blunder.
    Useful for real-time analysis.
    """
    saved = joblib.load(model_path)
    model = saved["model"]
    features = saved["features"]
    row = pd.DataFrame([fen_features])[features].fillna(0)
    prob = model.predict_proba(row)[0][1]
    return round(prob, 3)


# ── Train mistake type classifier ───────────────────────────────────────────────

def train_mistake_type_classifier(
    data_path: str = "data/processed/moves_categorized.parquet"
):
    """
    Multi-class classifier: predict type of mistake (hanging, tactic, endgame, etc.)
    Only trained on moves that are already mistakes.
    """
    df = pd.read_parquet(data_path)
    df = df[df["mistake_category"].notna() & (df["mistake_category"] != "none")]

    X, _, feature_names = prepare_features(df)
    y = df["mistake_category"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Mistake Type Classifier Results:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    model_path = os.path.join(MODEL_DIR, "mistake_type_classifier.joblib")
    joblib.dump({"model": model, "label_encoder": le, "features": feature_names}, model_path)
    print(f"Saved to {model_path}")
    return model, le


if __name__ == "__main__":
    # Train blunder predictor
    model, features = train_blunder_predictor(
        data_path="data/processed/moves_features.parquet",
        model_name="random_forest"
    )

    # Optionally train mistake type classifier
    # train_mistake_type_classifier("data/processed/moves_categorized.parquet")
