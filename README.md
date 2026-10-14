
# FIFA World Cup 2026 – Regression Analysis

## Project Overview
This repository contains the Python code, datasets, model evaluation results and visualisations for HIT140 Assessment 3. The project investigates two regression tasks using 104 FIFA World Cup 2026 match records.

## Research Objectives
- **Regression 2.1:** Predict the match goal difference (Team 1 goals minus Team 2 goals).
- **Regression 2.2:** Predict the number of goals scored by each team in a match.

## Data
The match dataset contains 104 match records. Scores were checked against FIFA match-report PDFs, with source URLs recorded in the dataset.

The data were transformed into:
- 104 match-level observations for Regression 2.1.
- 208 team-match observations for Regression 2.2.

Each task uses eight predictors derived from pre-match information and prior match performance. Match histories were updated after each match date to reduce information leakage.

## Methods
The project compares a mean-prediction baseline, ordinary least squares linear regression, Ridge regression and Lasso regression.

Models were evaluated using chronological training/test splits and time-aware cross-validation. Evaluation metrics include MAE, RMSE and R-squared.

## Selected Model Results

| Task | Selected model | Test MAE | Test RMSE | Test R-squared |
| --- | --- | ---: | ---: | ---: |
| Regression 2.1 | Lasso (alpha = 0.5) | 1.4627 | 1.7759 | -0.1769 |
| Regression 2.2 | Ridge (alpha = 100) | 0.9914 | 1.2979 | 0.0371 |

These results indicate limited predictive performance, especially for match goal difference. The models should not be interpreted as highly accurate score predictors.

## Repository Structure
- `data/raw/` – Original match data and supporting source records.
- `data/processed/` – Validated and regression-ready datasets.
- `src/` – Data preparation, feature engineering, model training, tuning and diagnostic scripts.
- `results/` – Cross-validation, predictions, scores and verification outputs.
- `figures/` – Model comparison and diagnostic plots.

## Running the Project
Install Python with pandas, numpy, scikit-learn and matplotlib. Run the scripts from the repository root after checking the input data and script dependencies.

The main modelling scripts are located in `src/`, including `build_features.py`, `train_models.py`, `tune_models.py` and `model_diagnostics.py`.

## Data Sources
FIFA post-match report URLs are recorded alongside the match records in `data/raw/matches.csv`. Additional raw records are included in `data/raw/` for reproducibility.

## Academic Context
Charles Darwin University – HIT140 Assessment 3. Group contributions and AI-assisted work should be described accurately in the accompanying assessment documentation.
  
