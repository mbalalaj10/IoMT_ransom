import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.utils import set_seed, get_device, ensure_dir
from src.models.autoencoder import Autoencoder


def main(dataset_name: str = "ton"):
    cfg = Config()
    set_seed(cfg.random_seed)
    ensure_dir(cfg.model_dir)

    if dataset_name.lower() == "ton":
        split_dir = cfg.ton_splits_path
    elif dataset_name.lower() == "sim":
        split_dir = cfg.sim_splits_path
    elif dataset_name.lower() == "cic":
        split_dir = cfg.cic_splits_path
    else:
        raise ValueError("dataset_name must be 'ton', 'sim', or 'cic'")

    X_train = np.load(os.path.join(split_dir, "X_train.npy"))
    y_train = np.load(os.path.join(split_dir, "y_train.npy"))
    X_val   = np.load(os.path.join(split_dir, "X_val.npy"))
    y_val   = np.load(os.path.join(split_dir, "y_val.npy"))

    print(f"Train: {X_train.shape}  Val: {X_val.shape}")

    X_train_benign = X_train[y_train == 0]
    X_val_benign   = X_val[y_val == 0]
    print(f"Benign train: {X_train_benign.shape}  Benign val: {X_val_benign.shape}")

    if len(X_train_benign) == 0:
        raise ValueError("No benign samples found for autoencoder training.")

    device = get_device()
    print("Using device:", device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_benign, dtype=torch.float32),
                      torch.tensor(X_train_benign, dtype=torch.float32)),
        batch_size=cfg.ae_batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_benign, dtype=torch.float32),
                      torch.tensor(X_val_benign, dtype=torch.float32)),
        batch_size=cfg.ae_batch_size, shuffle=False,
    )

    model = Autoencoder(
        input_dim=X_train.shape[1],
        hidden_dim1=cfg.ae_hidden_dim1,
        hidden_dim2=cfg.ae_hidden_dim2,
        latent_dim=cfg.latent_dim,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ae_learning_rate)

    save_path = {
        "ton": cfg.ton_autoencoder_model_path,
        "sim": cfg.sim_autoencoder_model_path,
        "cic": cfg.cic_autoencoder_model_path,
    }[dataset_name.lower()]

    best_val_loss   = float("inf")
    patience_counter = 0
    train_losses    = []

    for epoch in range(cfg.ae_epochs):
        model.train()
        total = 0.0
        for x_batch, target in train_loader:
            x_batch, target = x_batch.to(device), target.to(device)
            optimizer.zero_grad()
            _, x_hat = model(x_batch)
            loss = criterion(x_hat, target)
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss = total / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x_batch, target in val_loader:
                x_batch = x_batch.to(device)
                _, x_hat = model(x_batch)
                val_total += criterion(x_hat, x_batch).item()
        val_loss = val_total / len(val_loader)

        print(f"Epoch [{epoch+1}/{cfg.ae_epochs}]  Train: {train_loss:.6f}  Val: {val_loss:.6f}", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            print("  ✓ saved")
        else:
            patience_counter += 1
            print(f"  (patience {patience_counter}/{cfg.early_stopping_patience})")
            if patience_counter >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    print(f"Best val loss: {best_val_loss:.6f}  →  {save_path}")

    os.makedirs(cfg.loss_dir, exist_ok=True)
    np.save(os.path.join(cfg.loss_dir, f"ae_{dataset_name}_losses.npy"), np.array(train_losses))


if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "ton"
    main(dataset)
