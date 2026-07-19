# 6. Performance Testing

## Automated Backend Tests
The project includes `test_backend.py` (in
`5. Project Development Phase/OptiCrop/`), which runs smoke tests against
the live Flask app using its test client.

Run it with:
```bash
cd "5. Project Development Phase/OptiCrop"
python test_backend.py
```

### Checks performed
| Test | Request | Expected Status |
|---|---|---|
| GET / | Loads the home/prediction form | 200 |
| GET /about | Loads the about page | 200 |
| GET /dashboard | Loads the model comparison dashboard | 200 |
| POST /predict | Valid sample payload (N=90, P=42, K=43, temp=24.8, humidity=82, ph=6.5, rainfall=210) | 200 |
| POST /predict invalid | Same payload with an out-of-range `ph=99` | 400 |

This confirms the app correctly validates inputs (rejecting an invalid pH
of 99 with a 400 response) rather than silently predicting on nonsense
data.

## Model Evaluation
`train_model.py` trains and compares several candidate models (Decision
Tree, Random Forest, SVM, Logistic Regression, KNN) and writes comparison
metrics to `models/metrics.json`, which is displayed on the `/dashboard`
route along with:
- Correlation matrix
- Pair plot
- Feature distributions
- Crop distribution
- Confusion matrix (for the selected best model)

## Manual Test Cases

| Test ID | Description | Sample Input | Expected Result |
|---|---|---|---|
| TC01 | Rice-like profile | N=90, P=42, K=43, temp=24.8, humidity=82, ph=6.5, rainfall=210 | Predicts "Rice" with high confidence |
| TC02 | Banana-like profile | N=105, P=82, K=52, temp=27.5, humidity=80, ph=6.1, rainfall=105 | Predicts "Banana" |
| TC03 | Invalid pH | ph=99 (rest valid) | Returns 400 / validation error |
| TC04 | Dashboard loads | GET /dashboard | Displays model comparison + EDA charts |

## Notes
Per the project's own README, OptiCrop is an advisory tool — final crop and
fertilizer decisions should still be confirmed with local agronomists and
soil laboratory tests.
