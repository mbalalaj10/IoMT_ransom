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
    else:
        raise ValueError("dataset_name must be 'ton' or 'sim'")

    # Load preprocessed splits
    X_train = np.load(os.path.join(split_dir, "X_train.npy"))
    y_train = np.load(os.path.join(split_dir, "y_train.npy"))

    print("Loaded X_train shape:", X_train.shape)
    print("Loaded y_train shape:", y_train.shape)

    # Train autoencoder on benign traffic only
    benign_mask = y_train == 0
    X_train_benign = X_train[benign_mask]

    print("Benign training shape:", X_train_benign.shape)

    if len(X_train_benign) == 0:
        raise ValueError("No benign samples found for autoencoder training.")

    device = get_device()
    print("Using device:", device)

    dataset = TensorDataset(
        torch.tensor(X_train_benign, dtype=torch.float32),
        torch.tensor(X_train_benign, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=cfg.ae_batch_size, shuffle=True)

    model = Autoencoder(
        input_dim=X_train.shape[1],
        hidden_dim1=cfg.ae_hidden_dim1,
        hidden_dim2=cfg.ae_hidden_dim2,
        latent_dim=cfg.latent_dim,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ae_learning_rate)

    for epoch in range(cfg.ae_epochs):
        model.train()
        total_loss = 0.0

        for x_batch, target_batch in loader:
            x_batch = x_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()

            _, x_hat = model(x_batch)
            loss = criterion(x_hat, target_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{cfg.ae_epochs}] Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), cfg.autoencoder_model_path)
    print(f"Autoencoder saved to: {cfg.autoencoder_model_path}")


if __name__ == "__main__":
    main("ton")