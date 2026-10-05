# Project Plan: Card Fraud Triage (AML Baseline Phase)

## Overview
Frame card fraud as anomaly detection by fitting a Gaussian Mixture Model (GMM) primarily on legitimate transactions, using probability density scores to identify anomalies on a held-out mixed test set.

## Milestone 1: Exploratory Data Analysis (EDA) & Setup
* Handle dataset loading and initial visual inspection.
* Analyze data distribution and heavy class imbalance.

## Milestone 2: Data Preprocessing & Leakage Prevention
* Sort data chronologically by `Time` to prevent data leakage.
* Split dataset into 60% Train, 20% Validation, 20% Test.
* Isolate training data strictly to `Class == 0` (legitimate only).
* Output processed datasets: `train_legit.csv`, `val.csv`, `test.csv` (via `src/preprocess.py`).

## Milestone 3: GMM Training & Threshold Tuning
* Scale features using `StandardScaler`.
* Evaluate AIC and BIC scores to tune the Gaussian component count (Optimal k=7).
* Train the final GMM solely on legitimate transactions.
* Calculate log-likelihood scores and establish an anomaly threshold based on validation data.

## Milestone 4: Final Evaluation Notebook & Trade-off Analysis
* **Part 4A: Test Run & PR-AUC Curve **
  * Create `4_final_evaluation.ipynb`.
  * Load the trained GMM and run predictions on the untouched `test.csv` set.
  * Plot the Precision-Recall AUC (PR-AUC) curve.
* **Part 4B: Triage Metrics & Documentation**
  * Report recall at fixed false-positive budgets and precision at a specific review capacity.
  * Document the final false-positive trade-offs (The 'Triage' logic) for the evaluator.