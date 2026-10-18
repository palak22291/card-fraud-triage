# Card Fraud Triage: Unsupervised Anomaly Detection

Probabilistic density estimation pipeline for credit card fraud detection and risk triage under extreme class imbalance.

---

## Overview

Credit card fraud detection presents two fundamental engineering challenges:
1. **Extreme Class Imbalance:** Fraudulent transactions typically constitute less than 0.2% of total transaction volume (in this benchmark, 0.173%, or a 578:1 ratio). Conventional supervised classifiers tend to overfit historical patterns or collapse toward majority-class predictions without aggressive, synthetic resampling.
2. **Adversarial Non-Stationarity:** Fraud patterns evolve rapidly. Supervised decision boundaries trained on known fraud typologies often fail to generalize to novel attack vectors.

This project implements an unsupervised anomaly detection framework based on **Gaussian Mixture Models (GMM)**. Rather than attempting to learn a boundary around sparse, noisy fraud labels, the model estimates the generative probability density distribution $p(\mathbf{x})$ of strictly legitimate transaction behavior. Incoming transactions are evaluated via negative log-likelihood:

$$\mathcal{S}(\mathbf{x}) = -\ln p(\mathbf{x}) = -\ln \sum_{j=1}^{k} \pi_j \mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j\right)$$

Transactions yielding extreme negative log-likelihoods ($-\ln p(\mathbf{x}) > \tau$) represent statistical deviations from legitimate behavior and are routed to prioritized human investigator queues.

---

## Technical Architecture & Pipeline

The pipeline is organized into modular stages to guarantee reproducibility, prevent temporal data leakage, and support automated evaluation:

```
card-fraud-triage/
├── data/
│   ├── raw/                  # Source CSV datasets (git-ignored)
│   │   └── creditcard.csv    # Benchmark dataset (284,807 transactions)
│   └── processed/            # Chronologically partitioned sets (git-ignored)
│       ├── train_legit.csv   # 60% partition: strictly Class == 0 (170,524 samples)
│       ├── val.csv           # 20% partition: mixed validation set (56,961 samples)
│       └── test.csv          # 20% partition: held-out mixed test set (56,962 samples)
├── models/                   # Serialized inference artifacts
│   ├── best_gmm.joblib       # Fitted Gaussian Mixture Model (k=7, full covariance)
│   └── scaler.joblib         # Fitted StandardScaler artifact
├── notebooks/                # Exploratory and experimental workflows
│   ├── 01_eda_and_quality.ipynb      # EDA, feature distributions, imbalance analysis
│   ├── gmm_model_selection.ipynb     # AIC/BIC component sweep and validation tuning
│   └── final_evaluation.ipynb        # Test PR-AUC, ROC-AUC, and triage metrics
├── src/                      # Production scripts
│   ├── data_audit.py         # Integrity, duplicate, and null-value verification
│   └── preprocess.py         # Chronological sorting and leakage-free splitting
├── Project_Plan.md           # Implementation milestones and technical specification
├── requirements.txt          # Python dependencies
└── README.md                 # System documentation
```

### End-to-End Workflow

```
[Raw Transactions: creditcard.csv]
               │
               ▼
[Chronological Sort by Time]  ──> Prevents temporal lookahead leakage
               │
               ├─────────────────────────┬─────────────────────────┐
               ▼                         ▼                         ▼
      [Train: 60% Chrono]       [Val: 20% Chrono]         [Test: 20% Chrono]
               │                         │                         │
     [Filter Class == 0]                 │                         │
     (170,524 Legit Txs)                 │                         │
               │                         │                         │
               ▼                         │                         │
      [Fit StandardScaler] ──────────────┼─────────────────────────┤
               │ (transform)             │ (transform)             │ (transform)
               ▼                         ▼                         ▼
   [Scale Train Features]      [Scale Val Features]      [Scale Test Features]
               │                         │                         │
               ▼                         │                         │
     [GMM Optimization]                  │                         │
      (AIC/BIC Grid: k=7)                │                         │
               │                         │                         │
               ▼                         ▼                         ▼
     [Serialized Artifacts]   [Threshold Calibration]   [Operational Evaluation]
     - best_gmm.joblib        - 1st Percentile Cutoff   - PR-AUC: 0.3622
     - scaler.joblib          - Fixed FPR Budgets       - Precision@K Triage
```

---

## Data Engineering & Leakage Prevention

The benchmark dataset consists of **284,807 transactions** across 30 numerical features (28 PCA components `V1`–`V28`, transaction `Time`, and `Amount`).

To ensure real-world validity, three strict engineering safeguards are enforced:
1. **Chronological Splitting:** Financial transaction streams exhibit temporal dependencies. Shuffling or standard k-fold cross-validation causes future-to-past data leakage. The dataset is sorted by `Time` and segmented into:
   - **Training Set (60%):** 170,884 chronological transactions.
   - **Validation Set (20%):** 56,961 chronological transactions.
   - **Held-Out Test Set (20%):** 56,962 chronological transactions.
2. **Semi-Supervised Class Isolation:** To ensure pure one-class density estimation, all fraudulent records are purged from the training set. Only legitimate transactions (`Class == 0`, 170,524 rows) are retained. The label column is removed prior to model fitting.
3. **Leakage-Free Feature Scaling:** `StandardScaler` is fitted strictly on the legitimate training features and applied via `transform()` to the validation and test sets.

---

## Model Selection & Mathematical Formulation

### Gaussian Mixture Model (GMM)

Legitimate credit card behavior forms complex, multi-modal clusters across merchant types, transaction frequencies, and expenditure levels. A single multivariate Gaussian fails to capture this structure. We model transaction space as a mixture of $k$ multivariate Gaussians:

$$p(\mathbf{x}) = \sum_{j=1}^{k} \pi_j \mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j\right)$$

where:
- $\pi_j$ is the mixing weight for component $j$, with $\sum_{j=1}^{k} \pi_j = 1$ and $\pi_j \ge 0$.
- $\boldsymbol{\mu}_j \in \mathbb{R}^{d}$ is the mean vector of component $j$.
- $\boldsymbol{\Sigma}_j \in \mathbb{R}^{d \times d}$ is the full covariance matrix, capturing inter-feature dependencies across principal components and transaction amounts.

Parameters $\Theta = \{\pi_j, \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j\}_{j=1}^{k}$ are estimated via the Expectation-Maximization (EM) algorithm with full covariance structures.

### Component Sweep ($k$) via AIC and BIC

The number of mixture components $k$ balances model expressiveness against overfitting. We evaluated $k \in \{1, 2, \dots, 10\}$ using the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC):

$$\text{AIC} = 2m - 2\ln \hat{L}$$

$$\text{BIC} = \ln(N)m - 2\ln \hat{L}$$

where $m$ denotes the number of free parameters, $N$ is sample size ($N = 170,524$), and $\hat{L}$ is the maximized likelihood.

- **Outcome:** While AIC continues to drop monotonically due to weak parameter penalties on large sample sizes, BIC exhibits an elbow stabilization around **$k = 7$**. Selecting $k=7$ provides optimal representation of legitimate spending modes while avoiding parameter explosion.

---

## Evaluation & Operational Triage Metrics

Under a 0.132% test prevalence (75 fraud cases out of 56,962 transactions), standard accuracy is uninformative (a naive null predictor achieves 99.87% accuracy while missing 100% of fraud). 

The system is evaluated against triage capacity constraints: **Precision-Recall AUC (PR-AUC)**, **Recall at Fixed False Positive Rates (FPR)**, and **Precision@K**.

### Summary Performance (Test Set: 56,962 Transactions, 75 Fraud Cases)

| Metric | Result | Baseline / Comparison |
| :--- | :--- | :--- |
| **PR-AUC (Average Precision)** | **0.3622** | Random chance baseline = 0.00132 (274x lift) |
| **ROC-AUC** | **0.9065** | Random chance baseline = 0.5000 |
| **Train Score 1st Percentile Cutoff** | **$-58.66$** | Recall = 76.00%, Specificity = 97.91% |

### 1. Alert Budget Constraints: Recall at Fixed False Positive Rates

Financial institutions must limit false alerts to prevent investigator burnout and customer friction. Operating points are defined by strict False Positive Rate budgets:

| Target FPR Budget | Actual Test FPR | Test Recall (TPR) | Anomaly Cutoff ($-\ln p$) | Operational Application |
| :---: | :---: | :---: | :---: | :--- |
| **$\le 0.1\%$** | **$0.091\%$** | **$56.00\%$** | $\ge 151.26$ | Automated step-up authentication / immediate card block |
| **$\le 0.5\%$** | **$0.468\%$** | **$72.00\%$** | $\ge 76.35$ | High-priority investigator review queue |
| **$\le 1.0\%$** | **$0.986\%$** | **$74.67\%$** | $\ge 64.89$ | Daily comprehensive risk audit queue |

*Insight:* At a strict false-positive budget of under 0.1% (less than 1 false alarm per 1,000 legitimate transactions), the model intercepts **56.00%** of all fraudulent transactions.

### 2. Review Capacity Constraints: Precision@K and Recall@K

When fraud operations teams operate with fixed daily case review bandwidth, ranking alerts by anomaly score yields the following yields:

| Review Capacity ($K$ Alerts) | Frauds Caught | Precision@K | Recall@K | Review Efficiency |
| :---: | :---: | :---: | :---: | :--- |
| **Top-50 Alerts** | 24 / 75 | **48.00%** | **32.00%** | ~1 fraud detected per 2.08 reviews |
| **Top-100 Alerts** | 42 / 75 | **42.00%** | **56.00%** | Captures over half of all frauds in 100 alerts |
| **Top-200 Alerts** | 49 / 75 | **24.50%** | **65.33%** | Captures nearly two-thirds of all fraud |
| **Top-500 Alerts** | 55 / 75 | **11.00%** | **73.33%** | High coverage for expanded investigative teams |

### 3. Classification Performance at Validation Cutoff ($\tau = 58.66$)

Using the threshold determined from the 1st percentile of training scores (thresholding at $-\ln p(\mathbf{x}) \ge 58.66$):

```
Confusion Matrix (Test Partition):
┌──────────────────────────┬──────────────────────────┐
│ True Negative: 55,696    │ False Positive: 1,191    │
├──────────────────────────┼──────────────────────────┤
│ False Negative: 18       │ True Positive: 57        │
└──────────────────────────┴──────────────────────────┘

Detailed Test Set Classification Report:
              Precision    Recall    F1-Score    Support
Class 0 (Legit)  0.9997    0.9791      0.9893      56,887
Class 1 (Fraud)  0.0457    0.7600      0.0862          75
       Accuracy                        0.9788      56,962
      Macro Avg  0.5227    0.8695      0.5377      56,962
   Weighted Avg  0.9984    0.9788      0.9881      56,962
```

---

## Tech Stack & Dependencies

- **Language:** Python 3.10+
- **Machine Learning & Preprocessing:** `scikit-learn` (v1.3+), `scipy` (v1.10+)
- **Data Manipulation:** `pandas` (v2.0+), `numpy` (v1.24+)
- **Model Serialization:** `joblib`
- **Visualization:** `matplotlib` (v3.7+), `seaborn` (v0.12+)
- **Interactive Development:** `jupyterlab` (v4.0+)

---

## Installation & Setup

### 1. Environment Configuration

Clone the repository and initialize a virtual environment:

```bash
git clone https://github.com/<your-username>/card-fraud-triage.git
cd card-fraud-triage

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Dataset Preparation

1. Download the Credit Card Fraud Detection dataset from Kaggle (`creditcard.csv`).
2. Place the raw CSV file in `data/raw/`:
   ```bash
   mkdir -p data/raw data/processed models
   # Move downloaded file to data/raw/creditcard.csv
   ```
3. Execute the data audit script to verify schema consistency, missing values, and integrity:
   ```bash
   python src/data_audit.py
   ```

### 3. Chronological Preprocessing

Generate the temporal, leakage-free partitions:

```bash
python src/preprocess.py
```

This creates:
- `data/processed/train_legit.csv`: Unsupervised legitimate training set.
- `data/processed/val.csv`: Mixed validation set for threshold tuning.
- `data/processed/test.csv`: Held-out mixed test set for final reporting.

### 4. Running Model Selection & Evaluation

Launch JupyterLab to inspect or re-execute the experimental notebooks:

```bash
jupyter lab
```

Execute the notebooks in sequence:
1. `notebooks/01_eda_and_quality.ipynb`: Exploratory data analysis and feature distributions.
2. `notebooks/gmm_model_selection.ipynb`: AIC/BIC hyperparameter search, component selection ($k=7$), and artifact serialization (`models/best_gmm.joblib`, `models/scaler.joblib`).
3. `notebooks/final_evaluation.ipynb`: Held-out test evaluation, PR-AUC plotting, and queue triage metric computation.

---

## Production Scoring Snippet

To load the trained artifacts and score incoming transactions in an operational pipeline:

```python
import joblib
import numpy as np
import pandas as pd

# 1. Load serialized inference artifacts
scaler = joblib.load("models/scaler.joblib")
model = joblib.load("models/best_gmm.joblib")

# 2. Prepare incoming transactions (must contain identical 30 features)
incoming_txs = pd.DataFrame(...)  # Features: Time, V1-V28, Amount
scaled_features = scaler.transform(incoming_txs)

# 3. Compute anomaly scores via negative log-likelihood
# Lower density under GMM -> Higher anomaly score
anomaly_scores = -model.score_samples(scaled_features)

# 4. Route based on operational triage thresholds
# Thresholds derived from validation tuning:
THRESH_CRITICAL = 151.26  # FPR <= 0.1% budget -> Immediate block
THRESH_REVIEW   = 76.35   # FPR <= 0.5% budget -> Priority human queue
THRESH_AUDIT    = 58.66   # 1st percentile cutoff -> General triage queue

def triage_transaction(score):
    if score >= THRESH_CRITICAL:
        return "ACTION_BLOCK"
    elif score >= THRESH_REVIEW:
        return "QUEUE_PRIORITY_INVESTIGATION"
    elif score >= THRESH_AUDIT:
        return "QUEUE_ROUTINE_AUDIT"
    return "ACTION_APPROVE"

routing_decisions = [triage_transaction(s) for s in anomaly_scores]
```

---

## Key Takeaways & Design Decisions

1. **Why Unsupervised Density Modeling?**
   In financial fraud, supervised models are prone to silent degradation as fraud rings adapt. By modeling the boundary of normalcy rather than the boundary of historical fraud, GMMs identify structural outliers regardless of attack mechanism.
2. **Why Full Covariance?**
   PCA decorrelates components across the overall dataset, but conditional distributions within sub-populations of legitimate spending retain correlations. Full covariance matrices allow each component to fit elliptical clusters accurately.
3. **Operational Alignment:**
   Machine learning in fraud risk cannot be evaluated by raw accuracy or ROC-AUC alone. Calibrating decisions to fixed investigator capacities (Top-K) and strict false positive budgets (FPR $\le 0.1\%$) translates statistical density estimation directly into measurable operational ROI.