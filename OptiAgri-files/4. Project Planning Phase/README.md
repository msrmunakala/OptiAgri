# 4. Project Planning Phase

## Project Milestones

### Milestone 1: Define Problem and Understanding
Identify the agricultural problem — recommending the most suitable crop
based on soil nutrients and climate conditions, plus supporting guidance
(fertilizer, season, water needs).

### Milestone 2: Data Collection and Analysis
- Load the Crop Recommendation Dataset (`N, P, K, temperature, humidity,
  ph, rainfall, label`).
- Perform EDA: correlation matrix, pair plots, feature distributions, crop
  distribution.
*Tools: Pandas, Matplotlib, Seaborn*

### Milestone 3: Data Pre-processing
- Handle missing values / duplicates.
- Prepare train/test splits for model comparison.
*Tools: Pandas, NumPy, Scikit-learn*

### Milestone 4: Model Building & Comparison
- Train and compare Decision Tree, Random Forest, SVM, Logistic Regression,
  and KNN models.
- Evaluate with a confusion matrix and other metrics; write results to
  `models/metrics.json`.
- Save the best model with Joblib to `models/crop_model.pkl`.
*Tools: Scikit-learn, Joblib*

### Milestone 5: Application Building
- Build the Flask backend (`app.py`) with `/`, `/predict`, `/dashboard`,
  `/about` routes.
- Build the Bootstrap 5 front end (`templates/`, `static/`).
- Add derived insights logic (`utils.py`) — soil health score, fertilizer
  suggestion, suitable season, water requirement.
*Tools: Flask, Bootstrap 5, Joblib*

### Milestone 6: Testing & Deployment
- Validate predictions end-to-end (`test_backend.py`).
- Prepare deployment configuration for Render / Railway / PythonAnywhere
  using `gunicorn`.

### Milestone 7: Conclusion & Demo
Summarize model comparison results and demonstrate the working application.

## Suggested Timeline

| Phase | Estimated Duration |
|---|---|
| Brainstorming & Requirement Analysis | Week 1 |
| Design | Week 1–2 |
| Data Collection & EDA | Week 2 |
| Model Building & Comparison | Week 3 |
| Flask App Development | Week 4 |
| Testing & Deployment Prep | Week 5 |
| Documentation & Demonstration | Week 5 |
