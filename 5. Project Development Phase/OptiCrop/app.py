"""Flask web application for OptiCrop."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

from train_model import CHART_DIR, METRICS_PATH, MODEL_PATH, ensure_model
from utils import (
    FEATURES,
    RANGES,
    basic_recommendation,
    crop_metadata,
    soil_health_score,
    validate_input,
)

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "opticrop-local-development-secret")


def load_model_bundle() -> dict:
    ensure_model()
    return joblib.load(MODEL_PATH)


def load_metrics() -> dict:
    ensure_model()
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.context_processor
def inject_globals() -> dict:
    return {"features": FEATURES, "ranges": RANGES}


@app.route("/")
def home():
    metrics = load_metrics()
    return render_template("index.html", metrics=metrics, values={}, errors={})


@app.route("/predict", methods=["POST"])
def predict():
    validation = validate_input(request.form)
    values_for_form = {feature: request.form.get(feature, "") for feature in FEATURES}
    metrics = load_metrics()

    if validation.errors:
        for message in validation.errors.values():
            flash(message, "danger")
        return render_template("index.html", metrics=metrics, values=values_for_form, errors=validation.errors), 400

    values = validation.values or []
    bundle = load_model_bundle()
    model = bundle["model"]
    frame = pd.DataFrame([values], columns=FEATURES)
    crop = str(model.predict(frame)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        confidence = round(float(probabilities.max()) * 100, 2)
    else:
        confidence = round(float(bundle.get("accuracy", 0.0)) * 100, 2)

    score, status = soil_health_score(values)
    metadata = crop_metadata(crop)
    recommendation = basic_recommendation(crop, score, status)

    return render_template(
        "result.html",
        crop=crop,
        confidence=confidence,
        soil_score=score,
        soil_status=status,
        metadata=metadata,
        recommendation=recommendation,
        inputs=dict(zip(FEATURES, values, strict=True)),
        metrics=metrics,
    )


@app.route("/dashboard")
def dashboard():
    metrics = load_metrics()
    charts = [
        chart.name
        for chart in sorted(CHART_DIR.glob("*.png"))
        if chart.is_file()
    ]
    return render_template("dashboard.html", metrics=metrics, charts=charts)


@app.route("/about")
def about():
    metrics = load_metrics()
    return render_template("about.html", metrics=metrics)


@app.errorhandler(404)
def not_found(_error):
    return redirect(url_for("home"))


if __name__ == "__main__":
    ensure_model()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug)
