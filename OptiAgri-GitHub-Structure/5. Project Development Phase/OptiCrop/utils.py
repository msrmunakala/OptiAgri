"""Utility helpers for OptiCrop predictions, metadata, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

RANGES = {
    "N": (0, 160),
    "P": (0, 160),
    "K": (0, 220),
    "temperature": (0, 50),
    "humidity": (0, 100),
    "ph": (3.0, 10.0),
    "rainfall": (0, 400),
}

CROP_INFO: dict[str, dict[str, str]] = {
    "apple": {
        "description": "A temperate fruit crop that prefers cool conditions and balanced nutrients.",
        "season": "Winter to early spring",
        "water": "Moderate, with consistent soil moisture",
        "fertilizer": "Compost with balanced NPK and calcium support",
    },
    "banana": {
        "description": "A tropical crop that performs well in warm, humid, nutrient-rich soil.",
        "season": "Year-round in tropical climates",
        "water": "High, frequent irrigation",
        "fertilizer": "Nitrogen and potassium-rich organic fertilizer",
    },
    "blackgram": {
        "description": "A pulse crop suited to warm weather and well-drained soils.",
        "season": "Kharif and summer",
        "water": "Low to moderate",
        "fertilizer": "Phosphorus-rich starter fertilizer and rhizobium inoculation",
    },
    "chickpea": {
        "description": "A cool-season legume that tolerates dry conditions and improves soil nitrogen.",
        "season": "Rabi",
        "water": "Low, avoid waterlogging",
        "fertilizer": "Phosphorus and sulfur with organic compost",
    },
    "coconut": {
        "description": "A coastal palm crop that prefers humid weather and sandy loam soils.",
        "season": "Monsoon planting",
        "water": "Moderate to high",
        "fertilizer": "Organic manure with potassium and micronutrients",
    },
    "coffee": {
        "description": "A plantation crop that prefers mild temperatures, shade, and acidic soil.",
        "season": "Monsoon to post-monsoon",
        "water": "Moderate",
        "fertilizer": "Compost with nitrogen and potassium split applications",
    },
    "cotton": {
        "description": "A fiber crop that needs warm temperatures and well-drained black soil.",
        "season": "Kharif",
        "water": "Moderate",
        "fertilizer": "Nitrogen and phosphorus with potassium as needed",
    },
    "grapes": {
        "description": "A fruit vine crop that favors dry weather during ripening and well-drained soil.",
        "season": "Winter pruning to summer harvest",
        "water": "Moderate, drip irrigation preferred",
        "fertilizer": "Balanced NPK with organic manure",
    },
    "jute": {
        "description": "A bast fiber crop suited to hot, humid climates and alluvial soils.",
        "season": "Pre-monsoon to monsoon",
        "water": "High",
        "fertilizer": "Nitrogen with phosphorus and potassium basal dose",
    },
    "kidneybeans": {
        "description": "A legume crop that prefers cool temperatures and fertile, well-drained soil.",
        "season": "Spring or rabi in mild regions",
        "water": "Moderate",
        "fertilizer": "Phosphorus and potash with limited nitrogen",
    },
    "lentil": {
        "description": "A drought-tolerant pulse crop for cool, dry growing conditions.",
        "season": "Rabi",
        "water": "Low",
        "fertilizer": "Phosphorus and sulfur with seed inoculation",
    },
    "maize": {
        "description": "A cereal crop that grows quickly in warm weather with good fertility.",
        "season": "Kharif and spring",
        "water": "Moderate",
        "fertilizer": "Nitrogen-rich split fertilizer with phosphorus basal dose",
    },
    "mango": {
        "description": "A tropical fruit tree that prefers warm conditions and dry flowering weather.",
        "season": "Summer harvest",
        "water": "Low to moderate after establishment",
        "fertilizer": "Organic manure with balanced NPK and micronutrients",
    },
    "mothbeans": {
        "description": "A hardy pulse crop suitable for arid and semi-arid regions.",
        "season": "Kharif",
        "water": "Low",
        "fertilizer": "Low nitrogen, phosphorus support",
    },
    "mungbean": {
        "description": "A short-duration pulse crop that grows well in warm, well-drained soils.",
        "season": "Kharif and summer",
        "water": "Low to moderate",
        "fertilizer": "Phosphorus and compost with rhizobium inoculation",
    },
    "muskmelon": {
        "description": "A cucurbit fruit crop that needs warm weather and light, fertile soil.",
        "season": "Summer",
        "water": "Moderate, consistent during fruiting",
        "fertilizer": "Compost with balanced NPK and potassium during fruiting",
    },
    "orange": {
        "description": "A citrus crop that prefers subtropical weather and well-drained soil.",
        "season": "Winter harvest",
        "water": "Moderate",
        "fertilizer": "Citrus NPK blend with zinc and magnesium",
    },
    "papaya": {
        "description": "A fast-growing tropical fruit crop suited to warm and humid climates.",
        "season": "Year-round in frost-free areas",
        "water": "Moderate to high",
        "fertilizer": "Frequent organic manure and balanced NPK",
    },
    "pigeonpeas": {
        "description": "A deep-rooted pulse crop that tolerates drought and improves soil fertility.",
        "season": "Kharif",
        "water": "Low to moderate",
        "fertilizer": "Phosphorus and sulfur with seed inoculation",
    },
    "pomegranate": {
        "description": "A fruit crop that performs well in dry, warm climates with drainage.",
        "season": "Spring or monsoon bahar",
        "water": "Low to moderate",
        "fertilizer": "Organic manure with balanced NPK and potassium",
    },
    "rice": {
        "description": "A staple cereal crop best suited to high moisture and clay-rich fields.",
        "season": "Kharif",
        "water": "High",
        "fertilizer": "Nitrogen split dose with phosphorus and potassium basal dose",
    },
    "watermelon": {
        "description": "A warm-season fruit crop that prefers sandy loam and full sun.",
        "season": "Summer",
        "water": "Moderate, reduce near maturity",
        "fertilizer": "Compost with potassium and phosphorus support",
    },
}


@dataclass(frozen=True)
class ValidationResult:
    values: list[float] | None
    errors: dict[str, str]


def validate_input(form: dict[str, Any]) -> ValidationResult:
    """Validate form input and return ordered numeric feature values."""
    errors: dict[str, str] = {}
    values: list[float] = []

    for feature in FEATURES:
        raw_value = str(form.get(feature, "")).strip()
        label = feature.upper() if feature in {"N", "P", "K"} else feature.title()

        if raw_value == "":
            errors[feature] = f"{label} is required."
            continue

        try:
            value = float(raw_value)
        except ValueError:
            errors[feature] = f"{label} must be numeric."
            continue

        minimum, maximum = RANGES[feature]
        if not minimum <= value <= maximum:
            errors[feature] = f"{label} must be between {minimum} and {maximum}."
            continue

        values.append(value)

    if errors:
        return ValidationResult(values=None, errors=errors)

    return ValidationResult(values=values, errors={})


def soil_health_score(values: list[float]) -> tuple[int, str]:
    """Compute a simple agronomic soil-health status from user inputs."""
    n, p, k, _temperature, humidity, ph, rainfall = values
    nutrient_score = np.mean([
        min(n / 120, 1.0),
        min(p / 100, 1.0),
        min(k / 120, 1.0),
    ])
    ph_score = max(0.0, 1.0 - abs(ph - 6.5) / 3.5)
    moisture_score = np.mean([min(humidity / 80, 1.0), min(rainfall / 220, 1.0)])
    score = int(round((0.45 * nutrient_score + 0.35 * ph_score + 0.20 * moisture_score) * 100))

    if score >= 75:
        status = "Excellent"
    elif score >= 55:
        status = "Suitable"
    elif score >= 40:
        status = "Needs improvement"
    else:
        status = "Poor"

    return score, status


def crop_metadata(crop: str) -> dict[str, str]:
    """Return crop advisory metadata with a sensible fallback."""
    return CROP_INFO.get(
        crop.lower(),
        {
            "description": "A suitable crop for the submitted environmental profile.",
            "season": "Depends on local climate and variety",
            "water": "Follow local extension guidance",
            "fertilizer": "Use soil-test-based balanced fertilizer",
        },
    )


def basic_recommendation(crop: str, score: int, status: str) -> str:
    """Create a concise recommendation sentence."""
    crop_name = crop.title()
    if score >= 75:
        return f"{crop_name} is strongly recommended. Current soil and climate indicators are highly supportive."
    if score >= 55:
        return f"{crop_name} is recommended. Improve nutrients and irrigation scheduling for better yield stability."
    if status == "Needs improvement":
        return f"{crop_name} can be considered, but soil conditioning and pH correction should be prioritized."
    return f"{crop_name} is the best model match, but the soil profile needs improvement before production."
