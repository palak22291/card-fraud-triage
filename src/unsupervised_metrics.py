"""
Purely Unsupervised Evaluation Metrics & Pseudo-Label Generation.
Computes intrinsic density/manifold quality and inter-model anomaly agreement
WITHOUT touching ground-truth Class labels (0/1).
"""

from typing import Dict
import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    cohen_kappa_score,
    davies_bouldin_score,
    jaccard_score,
    silhouette_score,
)


def generate_unsupervised_pseudo_labels(
    train_scores: np.ndarray,
    eval_scores: np.ndarray,
    quantile: float = 0.99,
) -> Dict[str, object]:
    """
    Assign binary unsupervised anomaly labels y_hat_unsup in {0, 1}
    strictly from the q-th quantile of legitimate training anomaly scores.
    Zero ground-truth labels are used.
    """
    cutoff = float(np.quantile(train_scores, quantile))
    pseudo_labels = (eval_scores >= cutoff).astype(int)
    return {
        "quantile": quantile,
        "cutoff": cutoff,
        "pseudo_labels": pseudo_labels,
        "predicted_anomaly_count": int(pseudo_labels.sum()),
        "predicted_anomaly_rate": float(pseudo_labels.mean()),
    }


def compute_intrinsic_cluster_metrics(
    X_sample: np.ndarray,
    cluster_labels: np.ndarray,
    max_samples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate intrinsic unsupervised cluster separation using Silhouette Score,
    Davies-Bouldin Index (lower is better), and Calinski-Harabasz Index (higher is better).
    """
    rng = np.random.default_rng(seed)
    if len(X_sample) > max_samples:
        idx = rng.choice(len(X_sample), size=max_samples, replace=False)
        X_sub = X_sample[idx]
        labels_sub = cluster_labels[idx]
    else:
        X_sub = X_sample
        labels_sub = cluster_labels

    return {
        "silhouette_score": float(silhouette_score(X_sub, labels_sub)),
        "davies_bouldin_index": float(davies_bouldin_score(X_sub, labels_sub)),
        "calinski_harabasz_index": float(calinski_harabasz_score(X_sub, labels_sub)),
    }


def compute_unsupervised_consensus(
    gmm_pseudo_labels: np.ndarray,
    ae_pseudo_labels: np.ndarray,
    gmm_scores: np.ndarray,
    ae_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Measure purely unsupervised agreement between GMM density pseudo-labels
    and Autoencoder reconstruction pseudo-labels (without ground-truth Class).
    """
    both_flagged = int(((gmm_pseudo_labels == 1) & (ae_pseudo_labels == 1)).sum())
    either_flagged = int(((gmm_pseudo_labels == 1) | (ae_pseudo_labels == 1)).sum())
    spearman_corr = float(
        np.corrcoef(
            np.argsort(np.argsort(gmm_scores)),
            np.argsort(np.argsort(ae_scores)),
        )[0, 1]
    )
    return {
        "cohen_kappa": float(cohen_kappa_score(gmm_pseudo_labels, ae_pseudo_labels)),
        "jaccard_overlap": float(jaccard_score(gmm_pseudo_labels, ae_pseudo_labels)),
        "spearman_rank_corr": spearman_corr,
        "consensus_anomalies": both_flagged,
        "union_anomalies": either_flagged,
    }