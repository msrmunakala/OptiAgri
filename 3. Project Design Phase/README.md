# 3. Project Design Phase

## System Architecture

```
User (Browser)
      |
      v
Bootstrap Form (index.html)  --->  Flask Backend (app.py)  --->  Joblib Model (models/crop_model.pkl)
      |                                    |
      v                                    v
Input Validation                  Prediction + Derived Insights
      |                                    |
      +--------> result.html <-------------+

Flask Backend also serves:
  /dashboard  --> model comparison charts, EDA visuals (from models/metrics.json)
  /about      --> project description
```

## Routes (from `app.py`)
| Route | Purpose |
|---|---|
| `/` | Displays the prediction input form |
| `/predict` (POST) | Accepts form data, returns crop recommendation + confidence + soil insights |
| `/dashboard` | Displays model comparison table and EDA charts |
| `/about` | Displays project/about information |

## Data Flow
1. User enters N, P, K, temperature, humidity, pH, rainfall in the form.
2. Flask receives the POST request at `/predict`.
3. `utils.py` handles supporting logic (e.g. deriving soil health score,
   fertilizer suggestion, season, water requirement).
4. The persisted model (`models/crop_model.pkl`, loaded via Joblib) predicts
   the crop and a confidence percentage.
5. `result.html` renders the crop name, confidence, and additional insights.

## Model Design
- `train_model.py` trains and compares multiple candidate models: Decision
  Tree, Random Forest, SVM, Logistic Regression, and KNN.
- The best-performing model (by evaluation metric) is saved to
  `models/crop_model.pkl`, with comparison results saved to
  `models/metrics.json` for the dashboard to display.

## UI Design
- **Home Page (`index.html`):** Bootstrap-styled form for the 7 input
  parameters.
- **Result Page (`result.html`):** Recommended crop, confidence percentage,
  soil health score, suitability status, fertilizer/season/water guidance.
- **Dashboard (`dashboard.html`):** Model comparison table, correlation
  matrix, pair plot, feature distributions, crop distribution, confusion
  matrix.
- **About Page (`about.html`):** Project description.

## Database Design
Not applicable — the application is stateless at inference time. The
dataset is used only during offline/one-time model training
(`train_model.py`); no persistent database is required for serving
predictions.
