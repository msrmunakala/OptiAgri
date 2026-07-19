# 5. Project Development Phase

This folder contains the **complete, untouched OptiCrop application code**,
copied exactly as-is from the original repository into `OptiCrop/`.

No code inside `OptiCrop/` has been modified — only this folder wrapper and
note were added so it fits the numbered project-stage structure.

## What's inside `OptiCrop/`
```
OptiCrop/
├── app.py                # Flask backend (/, /predict, /dashboard, /about)
├── train_model.py        # Trains & compares models, saves best one
├── utils.py               # Helper logic (soil score, fertilizer suggestion, etc.)
├── test_backend.py       # Backend tests
├── requirements.txt
├── README.md              # Original project README with full run/deploy instructions
├── dataset/
│   └── Crop_recommendation.csv
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── images/ (farm-hero.svg, charts/)
└── templates/
    ├── base.html
    ├── index.html
    ├── result.html
    ├── dashboard.html
    └── about.html
```

## How to Run
See `OptiCrop/README.md` for the full, original setup and run instructions
(installation, training, deployment options, sample inputs).

Quick version:
```bash
cd "5. Project Development Phase/OptiCrop"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```
Then open `http://127.0.0.1:5000`.
