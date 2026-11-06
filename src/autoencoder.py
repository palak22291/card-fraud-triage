"""
PyTorch Deep Autoencoder & Denoising Autoencoder for Unsupervised Fraud Triage.
Maps 29 standardized behavioral features (V1-V28, Amount) through a symmetric
non-linear bottleneck:
    R^29 -> R^22 -> R^14 -> R^(latent_dim) -> R^14 -> R^22 -> R^29
Scores anomalies via per-transaction reconstruction Mean Squared Error (MSE).
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_deterministic_seed(seed: int = 42) -> None:
    """Enforce deterministic reproducibility across NumPy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FraudAutoencoder(nn.Module):
    """
    Symmetric Deep Autoencoder with BatchNorm, LeakyReLU, and optional
    Denoising corruption during training.
    """

    def __init__(
        self,
        input_dim: int = 29,
        hidden_dims: Tuple[int, int] = (22, 14),
        latent_dim: int = 8,
        dropout_rate: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        h1, h2 = hidden_dims

        # Encoder: R^29 -> R^h1 -> R^h2 -> R^latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(0.1),
            nn.Linear(h2, latent_dim),
        )

        # Decoder: R^latent_dim -> R^h2 -> R^h1 -> R^29 (linear output for standardized features)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(h2, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(0.1),
            nn.Linear(h1, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    @torch.no_grad()
    def reconstruction_errors(self, X: np.ndarray, device: str = "cpu", batch_size: int = 2048) -> np.ndarray:
        """
        Compute per-sample Mean Squared Reconstruction Error:
            s_AE(x_i) = (1 / D) * ||x_i - x_hat_i||_2^2
        """
        self.eval()
        tensor_x = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor_x), batch_size=batch_size, shuffle=False)
        scores = []
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            x_hat, _ = self.forward(batch_x)
            mse_per_row = torch.mean((batch_x - x_hat) ** 2, dim=1)
            scores.append(mse_per_row.cpu().numpy())
        return np.concatenate(scores, axis=0)

    @torch.no_grad()
    def feature_squared_errors(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        """
        Return per-feature squared error matrix (N x 29): (x_ij - x_hat_ij)^2
        for whiteboard interpretability and root-cause feature attribution.
        """
        self.eval()
        tensor_x = torch.tensor(X, dtype=torch.float32, device=device)
        x_hat, _ = self.forward(tensor_x)
        return ((tensor_x - x_hat) ** 2).cpu().numpy()

    @torch.no_grad()
    def extract_latent(self, X: np.ndarray, device: str = "cpu", batch_size: int = 2048) -> np.ndarray:
        """Extract compressed bottleneck representations z in R^(latent_dim)."""
        self.eval()
        tensor_x = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor_x), batch_size=batch_size, shuffle=False)
        latents = []
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            z = self.encode(batch_x)
            latents.append(z.cpu().numpy())
        return np.concatenate(latents, axis=0)


def train_autoencoder(
    model: FraudAutoencoder,
    X_train: np.ndarray,
    X_val_legit: np.ndarray,
    epochs: int = 35,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    noise_std: float = 0.0,
    patience: int = 6,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[FraudAutoencoder, Dict[str, List[float]]]:
    """
    Train Autoencoder exclusively on legitimate transactions with AdamW,
    CosineAnnealingLR, optional Denoising Gaussian noise, and Early Stopping.
    """
    set_deterministic_seed(seed)
    model = model.to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_tensor = torch.tensor(X_val_legit, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            if noise_std > 0.0:
                noisy_x = batch_x + noise_std * torch.randn_like(batch_x)
            else:
                noisy_x = batch_x

            optimizer.zero_grad()
            x_hat, _ = model(noisy_x)
            loss = criterion(x_hat, batch_x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            n_samples += batch_x.size(0)

        scheduler.step()
        train_epoch_loss = running_loss / n_samples

        model.eval()
        with torch.no_grad():
            val_hat, _ = model(val_tensor)
            val_epoch_loss = criterion(val_hat, val_tensor).item()

        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)

        if val_epoch_loss < best_val_loss - 1e-5:
            best_val_loss = val_epoch_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model, history