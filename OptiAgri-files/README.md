# OptiCrop: Smart Agricultural Production Optimization Engine

OptiCrop is a production-ready Flask web application that recommends the
most suitable crop from soil and environmental conditions using machine
learning. It compares multiple models (Decision Tree, Random Forest, SVM,
Logistic Regression, KNN), saves the best-performing one, and serves
predictions through a Bootstrap-styled web interface with a results
dashboard.

## Repository Structure

| Folder | Description |
|---|---|
| `1. Brainstorming & Ideation` | Problem statement, idea generation, and motivation |
| `2. Requirement Analysis` | Functional & non-functional requirements, tools/tech stack |
| `3. Project Design Phase` | System architecture, data flow, UI design |
| `4. Project Planning Phase` | Milestones, task breakdown |
| `5. Project Development Phase` | Full OptiCrop source code (app, model training, templates, static assets) |
| `6. Performance Testing` | Model evaluation approach, test cases |
| `7. Documentation` | Prerequisites, project workflow |
| `8. Project Demonstration` | Demo video link, screenshots info |

**Note:** All application source code lives untouched inside
`5. Project Development Phase/OptiCrop/` — nothing in the code was modified,
only the surrounding folder organization was added.

## Key Features
- Crop prediction from N, P, K, temperature, humidity, pH, and rainfall
- Confidence percentage, soil health score, soil suitability status
- Crop description, suitable season, water requirement, fertilizer suggestion
- EDA visuals: correlation matrix, pair plot, distribution plots, confusion matrix
- Model comparison: Decision Tree, Random Forest, SVM, Logistic Regression, KNN
- Best model saved via Joblib to `models/crop_model.pkl`
- Responsive Bootstrap 5 agriculture-themed UI
- Deployable on Render, Railway, or PythonAnywhere

## How to Run
See `5. Project Development Phase/OptiCrop/README.md` for full setup and run
instructions.
