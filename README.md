# Sports Predictor

A machine learning and sports analytics project designed to predict player performance using historical and game-level data.

The project currently focuses on **MLB pitcher strikeout prediction**, with plans to expand into additional MLB predictions and eventually support NFL and NBA player performance models.

## Overview

The goal of Sports Predictor is to build an end-to-end machine learning pipeline that collects sports data, engineers predictive features, trains and evaluates models, and generates player performance predictions.

The MLB pitcher strikeout model uses historical pitching data and recent performance trends to estimate the number of strikeouts a starting pitcher will record in an upcoming game.

Rather than focusing only on model training, the project is structured around the full machine learning workflow, including:

* Data collection
* Data validation and preprocessing
* Feature engineering
* Time-based train/validation/test splitting
* Baseline modeling
* Machine learning model training
* Model evaluation
* Automated testing

## Current Focus: MLB Pitcher Strikeouts

The current model predicts:

**Target:** Pitcher strikeouts per start

Historical pitch-level data is aggregated into individual pitcher starts before features are generated for model training.

### Current Features

The model currently incorporates features such as:

* Recent strikeout performance
* Swinging-strike rate
* Average fastball/pitch velocity
* Average spin rate
* Pitch count and workload
* Rolling performance statistics

Rolling features are calculated using only information available **before the game being predicted** to prevent data leakage.

Additional contextual and opponent-based features are planned as the model develops.

## Data Pipeline

The MLB pipeline collects and processes data from multiple sources.

**Statcast data** is used for pitch-level information, including:

* Pitch velocity
* Spin rate
* Swinging strikes
* Pitch counts
* Strikeouts

**MLB game data** is used to identify official starting pitchers and ensure the training dataset contains legitimate pitcher starts.

The resulting pitch-level data is aggregated into a pitcher-start dataset used for feature engineering and model training.

## Modeling

The project uses a baseline-first modeling approach.

### Baseline Model

A rolling strikeout average is used as the initial benchmark.

Current performance:

| Metric | Score |
| ------ | ----: |
| MAE    |  2.01 |
| RMSE   |  2.49 |
| R²     |  0.09 |

### Linear Regression

A Linear Regression model was trained using rolling pitcher performance features.

Current performance:

| Metric | Score |
| ------ | ----: |
| MAE    |  1.99 |
| RMSE   |  2.42 |
| R²     |  0.14 |

The Linear Regression model currently improves upon the baseline. Additional models and features are being evaluated as development continues.

## Model Evaluation

Models are evaluated using:

* **MAE (Mean Absolute Error)** — average prediction error in strikeouts
* **RMSE (Root Mean Squared Error)** — penalizes larger prediction errors
* **R²** — measures the amount of variation explained by the model

Because sports data is time-dependent, the dataset is split chronologically rather than using a random train/test split.

This more closely represents the real-world scenario of training on historical games and predicting future games.

## Testing

Automated tests are included to validate important parts of the data and modeling pipeline.

Current tests cover areas such as:

* Statcast data processing
* MLB starting pitcher identification
* Pitcher dataset construction
* Time-based dataset splitting
* Model evaluation

The test suite is run using `pytest`.

## Tech Stack

**Language**

* Python

**Data & Machine Learning**

* Pandas
* NumPy
* scikit-learn
* pybaseball

**Testing & Development**

* pytest
* Git
* GitHub

## Project Structure

```text
sports-predictor/
│
├── data/               # Data collection and processing
├── features/           # Feature engineering
├── model/              # Modeling utilities
├── odds/               # Sportsbook/odds-related functionality
├── training/           # Training, splitting, and evaluation
├── tests/              # Automated tests
│
├── app.py
└── README.md
```

The project structure will continue to evolve as additional models and sports are added.

## Roadmap

### MLB

* [x] Build pitcher-start dataset
* [x] Create rolling pitcher features
* [x] Establish baseline model
* [x] Train Linear Regression model
* [x] Add automated pipeline tests
* [ ] Add opponent and matchup features
* [ ] Evaluate additional machine learning models
* [ ] Improve model performance
* [ ] Generate predictions for upcoming games
* [ ] Integrate predictions into the application

### Future Expansion

* [ ] Expand MLB prediction markets
* [ ] Add NFL player performance models
* [ ] Add NBA player performance models
* [ ] Build a unified prediction interface
* [ ] Explore deployment options

## Project Goals

Sports Predictor is being developed as an end-to-end machine learning project focused on applying data science techniques to real-world sports data.

The project emphasizes:

* Reproducible data pipelines
* Leakage-safe feature engineering
* Time-aware model evaluation
* Model comparison against meaningful baselines
* Automated testing
* Maintainable project structure

The long-term goal is to create a multi-sport platform capable of producing data-driven player performance predictions across MLB, NFL, and NBA.
