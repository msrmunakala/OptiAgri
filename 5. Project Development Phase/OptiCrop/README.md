# OptiCrop - Smart Agricultural Production Optimization Engine

OptiCrop is a production-ready Flask web application that recommends the most suitable crop from soil and environmental conditions using machine learning.

## Features

- Crop prediction from Nitrogen, Phosphorus, Potassium, temperature, humidity, pH, and rainfall
- Confidence percentage, soil health score, soil suitability status, and basic recommendation
- Crop description, suitable season, estimated water requirement, and fertilizer suggestion
- Data cleaning, EDA, correlation matrix, pair plot, distribution plots, and confusion matrix
- Model training and comparison for Decision Tree, Random Forest, SVM, Logistic Regression, and KNN
- Best model saved to `models/crop_model.pkl` using Joblib
- Responsive Bootstrap 5 interface with agriculture-themed UI
- Compatible with Render, Railway, and PythonAnywhere

## Project Structure

```text
OptiCrop/
├── app.py
├── train_model.py
├── utils.py
├── requirements.txt
├── README.md
├── dataset/
│   └── Crop_recommendation.csv
├── models/
│   ├── crop_model.pkl
│   └── metrics.json
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── images/
│       ├── farm-hero.svg
│       └── charts/
└── templates/
    ├── base.html
    ├── index.html
    ├── result.html
    ├── dashboard.html
    └── about.html
```

## Installation

```bash
cd OptiCrop
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000`.

## Dataset

The app uses the Crop Recommendation Dataset with these columns:

```text
N, P, K, temperature, humidity, ph, rainfall, label
```

If `dataset/Crop_recommendation.csv` is missing, the training pipeline attempts to download it automatically from public GitHub mirrors. If the machine is offline, OptiCrop generates a realistic fallback dataset so the app remains runnable.

## Model Accuracy

Run:

```bash
python train_model.py
```

The script writes model comparison results to `models/metrics.json` and saves the best model to `models/crop_model.pkl`.

## Dashboard

The `/dashboard` route displays:

- Model comparison table
- Correlation matrix
- Pair plot
- Feature distributions
- Crop distribution
- Confusion matrix

## Sample Inputs

Rice-like profile:

```text
N=90
P=42
K=43
temperature=24.8
humidity=82
ph=6.5
rainfall=210
```

Expected output: usually `Rice`, with a high confidence score.

Banana-like profile:

```text
N=105
P=82
K=52
temperature=27.5
humidity=80
ph=6.1
rainfall=105
```

Expected output: usually `Banana`.

## Deployment

### Render

Use these settings:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Python version: 3.12

### Railway

Deploy the repository and set the start command:

```bash
gunicorn app:app
```

### PythonAnywhere

Create a virtual environment, install `requirements.txt`, and point the WSGI file to `app:app`.

## Screenshots

Run the app and visit:

- `/` for the prediction form
- `/dashboard` for charts
- `/about` for project details

## Notes

OptiCrop is an advisory tool. Final crop and fertilizer decisions should be confirmed with local agronomists and soil laboratory tests.
