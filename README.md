# Card Fraud Triage: Anomaly Detection via GMMs

This repository contains Phase 1 (Advanced Machine Learning) of our anomaly detection pipeline, framing credit card fraud detection as an unsupervised probability density problem. 

## Methodology
Because fraud tactics evolve rapidly and represent less than 0.2% of transaction volume, traditional supervised classification often fails to generalize. Instead, we model the probability density of strictly legitimate transactions using Gaussian Mixture Models (GMMs). Any transaction exhibiting a low log-likelihood under this density model is flagged as an anomaly and routed to human investigators.

## Project Structure
* `data/raw/`: Raw CSV files (git-ignored)
* `data/processed/`: Leakage-free chronological splits (git-ignored)
* `notebooks/`: EDA and experimental validation
* `src/`: Modular Python pipeline scripts

## Setup & Reproduction
1. Clone the repository and create a virtual environment (Python 3.10+).
2. Install dependencies: `pip install -r requirements.txt`
3. Download the ULB Credit Card Fraud dataset from Kaggle and place `creditcard.csv` in `data/raw/`.
4. Run the data validation audit: `python src/data_audit.py`
5. Generate chronological splits: `python src/data_split.py`

## Operational Evaluation (In Progress)
Because accuracy is a misleading metric for severe class imbalances, this model is evaluated strictly on triage efficiency:
* **PR-AUC (Average Precision)**
* **Recall at fixed False Positive Rates (0.1%, 0.5%, 1.0%)**
* **Precision at fixed daily review capacities (Top-K)**