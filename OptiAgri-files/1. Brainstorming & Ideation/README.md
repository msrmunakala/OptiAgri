# 1. Brainstorming & Ideation

## Problem Statement
Farmers frequently choose crops based on tradition or guesswork rather than
actual soil and climate suitability, leading to poor yields, wasted
resources, and reduced income. There is a need for a data-driven tool that
recommends the best crop for given soil and climate conditions, along with
practical guidance (fertilizer, water needs, suitable season).

## Motivation
- Agriculture is highly sensitive to soil nutrient levels and weather.
- Data-driven crop selection can improve yield and reduce input wastage.
- A production-style web tool with clear visual feedback (confidence score,
  soil health score, dashboard charts) makes ML-backed recommendations more
  trustworthy and actionable for real users.

## Idea
Build **OptiCrop** — a Flask web application where a user enters soil
nutrient values (N, P, K) and climate parameters (temperature, humidity,
pH, rainfall), and the system:
- Recommends the most suitable crop, with a confidence percentage
- Reports a soil health score and suitability status
- Suggests fertilizer, suitable season, and estimated water requirement
- Provides a dashboard of model comparison charts and EDA visuals

## Target Users
- Farmers and agricultural cooperatives
- Agricultural extension officers / consultants
- Students and researchers exploring precision agriculture

## Proposed Solution Approach
1. Collect a labeled crop recommendation dataset.
2. Explore data via EDA (correlation matrix, pair plots, distributions).
3. Preprocess and train multiple candidate models.
4. Select and persist the best-performing model.
5. Serve predictions and supporting insights through a Flask + Bootstrap UI.

## Key Questions Explored
- Which of several candidate models (Decision Tree, Random Forest, SVM,
  Logistic Regression, KNN) performs best on this dataset?
- What additional, farmer-useful information can be derived alongside the
  raw prediction (soil health, fertilizer suggestion, season)?
- How can the solution be made deployable on free-tier hosting (Render,
  Railway, PythonAnywhere)?
