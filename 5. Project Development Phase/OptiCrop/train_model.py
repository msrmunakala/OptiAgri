"""Train and evaluate crop recommendation models for OptiCrop."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from utils import FEATURES

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
CHART_DIR = BASE_DIR / "static" / "images" / "charts"
DATASET_PATH = DATA_DIR / "Crop_recommendation.csv"
MODEL_PATH = MODEL_DIR / "crop_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

DATASET_URLS = [
    "https://raw.githubusercontent.com/atharvaingle/crop-recommendation-dataset/master/Crop_recommendation.csv",
    "https://raw.githubusercontent.com/dphi-official/Datasets/master/crop_recommendation/Crop_recommendation.csv",
]


def ensure_directories() -> None:
    for directory in [DATA_DIR, MODEL_DIR, CHART_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def download_dataset() -> bool:
    """Download the public crop recommendation dataset when it is missing."""
    if DATASET_PATH.exists() and DATASET_PATH.stat().st_size > 1000:
        return True

    for url in DATASET_URLS:
        try:
            urlretrieve(url, DATASET_PATH)
            if DATASET_PATH.exists() and DATASET_PATH.stat().st_size > 1000:
                return True
        except (OSError, URLError):
            continue
    return False


def build_fallback_dataset() -> pd.DataFrame:
    """Create a realistic synthetic fallback if downloads are unavailable."""
    rng = np.random.default_rng(42)
    profiles = {
        "rice": (85, 45, 40, 25, 82, 6.4, 235),
        "maize": (75, 48, 22, 23, 65, 6.3, 90),
        "chickpea": (40, 68, 80, 18, 18, 7.1, 80),
        "kidneybeans": (22, 68, 20, 20, 22, 5.8, 105),
        "pigeonpeas": (20, 68, 22, 28, 48, 5.8, 150),
        "mothbeans": (22, 48, 20, 28, 53, 6.8, 50),
        "mungbean": (20, 47, 20, 28, 85, 6.7, 50),
        "blackgram": (40, 65, 20, 30, 65, 7.0, 70),
        "lentil": (25, 65, 20, 22, 64, 6.9, 45),
        "pomegranate": (20, 15, 40, 22, 90, 6.4, 105),
        "banana": (100, 80, 50, 27, 80, 6.0, 105),
        "mango": (20, 28, 30, 32, 50, 5.8, 95),
        "grapes": (24, 132, 200, 23, 82, 6.1, 70),
        "watermelon": (100, 15, 50, 26, 85, 6.5, 55),
        "muskmelon": (100, 15, 50, 28, 92, 6.4, 25),
        "apple": (20, 125, 200, 22, 92, 5.9, 110),
        "orange": (20, 15, 10, 23, 92, 7.0, 110),
        "papaya": (50, 60, 50, 33, 92, 6.7, 150),
        "coconut": (20, 15, 30, 27, 95, 5.9, 175),
        "cotton": (115, 45, 20, 24, 80, 6.9, 80),
        "jute": (75, 45, 40, 25, 80, 6.6, 175),
        "coffee": (100, 25, 30, 25, 58, 6.5, 160),
    }
    rows = []
    for crop, center in profiles.items():
        for _ in range(100):
            noise = rng.normal([0, 0, 0, 0, 0, 0, 0], [12, 10, 15, 2.5, 8, 0.35, 22])
            values = np.array(center, dtype=float) + noise
            values = np.clip(values, [0, 0, 0, 0, 0, 3, 0], [160, 160, 220, 50, 100, 10, 400])
            rows.append([*values, crop])
    data = pd.DataFrame(rows, columns=[*FEATURES, "label"])
    data.to_csv(DATASET_PATH, index=False)
    return data


def load_and_clean_data() -> pd.DataFrame:
    ensure_directories()
    downloaded = download_dataset()
    if downloaded:
        data = pd.read_csv(DATASET_PATH)
    else:
        data = build_fallback_dataset()

    expected = [*FEATURES, "label"]
    missing = set(expected).difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    data = data[expected].copy()
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data.dropna(inplace=True)
    return data


def save_visualizations(data: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", palette="viridis")

    plt.figure(figsize=(9, 7))
    corr = data[FEATURES].corr()
    sns.heatmap(corr, annot=True, cmap="YlGnBu", linewidths=0.5, fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "correlation_matrix.png", dpi=180)
    plt.close()

    subset = data[["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]]
    sns.pairplot(subset.sample(min(450, len(subset)), random_state=42), hue="label", corner=True, plot_kws={"s": 18})
    plt.savefig(CHART_DIR / "pair_plot.png", dpi=150)
    plt.close("all")

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    axes = axes.flatten()
    for idx, feature in enumerate(FEATURES):
        sns.histplot(data[feature], kde=True, ax=axes[idx], color="#2f8f46")
        axes[idx].set_title(f"{feature} Distribution")
    for idx in range(len(FEATURES), len(axes)):
        axes[idx].axis("off")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "feature_distributions.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 7))
    order = data["label"].value_counts().index
    sns.countplot(data=data, y="label", order=order, color="#3d9b55")
    plt.title("Crop Class Distribution")
    plt.xlabel("Samples")
    plt.ylabel("Crop")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "crop_distribution.png", dpi=180)
    plt.close()


def model_candidates() -> dict[str, Pipeline]:
    scaler = ColumnTransformer(
        transformers=[("scale", StandardScaler(), FEATURES)],
        remainder="drop",
    )
    return {
        "Decision Tree": Pipeline([("model", DecisionTreeClassifier(random_state=42))]),
        "Random Forest": Pipeline([("model", RandomForestClassifier(n_estimators=250, random_state=42))]),
        "SVM": Pipeline([("scale", scaler), ("model", SVC(probability=True, kernel="rbf", C=10, random_state=42))]),
        "Logistic Regression": Pipeline([
            ("scale", scaler),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "KNN": Pipeline([("scale", scaler), ("model", KNeighborsClassifier(n_neighbors=5))]),
    }


def train_and_save() -> dict[str, object]:
    data = load_and_clean_data()
    save_visualizations(data)

    x = data[FEATURES]
    y = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results: dict[str, float] = {}
    trained_models: dict[str, Pipeline] = {}
    for name, pipeline in model_candidates().items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        results[name] = float(accuracy_score(y_test, predictions))
        trained_models[name] = pipeline

    best_name = max(results, key=results.get)
    best_model = trained_models[best_name]
    best_predictions = best_model.predict(x_test)
    labels = sorted(y.unique())

    cm = confusion_matrix(y_test, best_predictions, labels=labels)
    plt.figure(figsize=(13, 10))
    sns.heatmap(cm, annot=False, cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    report = classification_report(y_test, best_predictions, output_dict=True, zero_division=0)
    bundle = {
        "model": best_model,
        "features": FEATURES,
        "labels": labels,
        "best_model": best_name,
        "accuracy": results[best_name],
    }
    joblib.dump(bundle, MODEL_PATH)

    metrics = {
        "best_model": best_name,
        "accuracy": results[best_name],
        "model_scores": results,
        "classification_report": report,
        "dataset_rows": int(len(data)),
        "dataset_path": str(DATASET_PATH.relative_to(BASE_DIR)),
        "charts": [
            "correlation_matrix.png",
            "pair_plot.png",
            "feature_distributions.png",
            "crop_distribution.png",
            "confusion_matrix.png",
        ],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def ensure_model() -> dict[str, object]:
    """Train once when artifacts are missing."""
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return train_and_save()


if __name__ == "__main__":
    metrics = train_and_save()
    print(f"Best model: {metrics['best_model']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
