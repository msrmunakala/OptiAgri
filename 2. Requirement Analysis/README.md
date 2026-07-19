# 2. Requirement Analysis

## Functional Requirements
- FR1: The system shall accept 7 input parameters from the user: N, P, K,
  temperature, humidity, pH, and rainfall (via the `/` form).
- FR2: The system shall predict the most suitable crop via the `/predict`
  route and return a confidence percentage.
- FR3: The system shall report a soil health score and soil suitability
  status alongside the prediction.
- FR4: The system shall provide a crop description, suitable growing
  season, estimated water requirement, and fertilizer suggestion.
- FR5: The system shall expose a `/dashboard` route showing model
  comparison results and EDA visuals (correlation matrix, pair plot,
  distribution plots, confusion matrix).
- FR6: The system shall expose an `/about` route describing the project.
- FR7: If the trained model file is missing, the training pipeline should
  attempt to fetch the dataset automatically, falling back to a generated
  dataset if offline, so the app remains runnable.

## Non-Functional Requirements
- NFR1: The application should use a responsive Bootstrap 5 UI suitable for
  desktop and mobile.
- NFR2: The best-performing model (out of several candidates) should be
  automatically selected and persisted via Joblib.
- NFR3: The application should be deployable on common free-tier hosts
  (Render, Railway, PythonAnywhere) using `gunicorn`.
- NFR4: Prediction response time should remain low, since the model is
  loaded once at startup rather than retrained per request.

## Tech Stack (from `requirements.txt`)
| Library | Version | Purpose |
|---|---|---|
| Flask | 3.0.3 | Web framework / backend routing |
| pandas | 2.2.2 | Data loading, cleaning, EDA |
| numpy | 1.26.4 | Numerical operations |
| scikit-learn | 1.5.1 | Model training, comparison, evaluation |
| matplotlib | 3.10.3 | EDA plots |
| seaborn | 0.13.2 | Statistical visualizations |
| joblib | 1.4.2 | Model persistence |
| requests | 2.32.3 | Automatic dataset download fallback |
| gunicorn | 22.0.0 | Production WSGI server for deployment |

## Front-End
- Bootstrap 5 (agriculture-themed UI)
- Templates: `base.html`, `index.html`, `result.html`, `dashboard.html`, `about.html`
- Static assets: `static/css/style.css`, `static/js/script.js`, `static/images/`

## Dataset
Crop Recommendation Dataset with columns: `N, P, K, temperature, humidity,
ph, rainfall, label`. Located at `dataset/Crop_recommendation.csv` inside
the project.
