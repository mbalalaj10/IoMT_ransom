import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config
from src.utils import set_seed, get_device, ensure_dir
from src.datasets.sequence_dataset import ArraySequenceDataset
from src.models.autoencoder import Autoencoder
from src.models.mamba_classifier import MambaClassifier


def extract_latent_and_error(model, X, device, batch_size=256):
#Pass raw features through the autoencoder and return:

    model.eval()
    combined_features = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)

            z, x_hat = model(batch)

            # Reconstruction error per sample
            recon_error = torch.mean((batch - x_hat) ** 2, dim=1, keepdim=True)

            # Concatenate latent features with reconstruction error
            combined = torch.cat([z, recon_error], dim=1)

            combined_features.append(combined.cpu().numpy())

    return np.vstack(combined_features)


def main(dataset_name="ton"):
    cfg = Config()
    set_seed(cfg.random_seed)
    ensure_dir(cfg.model_dir)

    # Select split directory
    if dataset_name.lower() == "ton":
        split_dir = cfg.ton_splits_path
    elif dataset_name.lower() == "sim":
        split_dir = cfg.sim_splits_path
    else:
        raise ValueError("dataset_name must be 'ton' or 'sim'")

    # Load processed splits
    X_train = np.load(os.path.join(split_dir, "X_train.npy"))
    X_test = np.load(os.path.join(split_dir, "X_test.npy"))
    y_train = np.load(os.path.join(split_dir, "y_train.npy"))
    y_test = np.load(os.path.join(split_dir, "y_test.npy"))

    # Load group ids for grouped sequence creation
    group_ids_train = np.load(os.path.join(split_dir, "group_ids_train.npy"), allow_pickle=True)
    group_ids_test  = np.load(os.path.join(split_dir, "group_ids_test.npy"),  allow_pickle=True)

    # Diagnostic: show group size distribution
    unique, counts = np.unique(group_ids_train, return_counts=True)
    viable = np.sum(counts >= cfg.seq_len)
    print(f"Train groups: {len(unique)} total, {viable} with >= {cfg.seq_len} rows (viable for sequences)")
    print(f"Group sizes  — min: {counts.min()}, median: {int(np.median(counts))}, max: {counts.max()}")

    device = get_device()
    print("Using device:", device)
    print("Training on dataset:", dataset_name)
    print("Split directory:", split_dir)

    # Load trained autoencoder
    ae = Autoencoder(
        input_dim=X_train.shape[1],
        hidden_dim1=cfg.ae_hidden_dim1,
        hidden_dim2=cfg.ae_hidden_dim2,
        latent_dim=cfg.latent_dim
    ).to(device)

    ae_path = cfg.ton_autoencoder_model_path if dataset_name.lower() == "ton" else cfg.sim_autoencoder_model_path
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.eval()

    # Extract latent features + reconstruction error
    Z_train = extract_latent_and_error(ae, X_train, device)
    Z_test = extract_latent_and_error(ae, X_test, device)

    print("Encoded training shape:", Z_train.shape)
    print("Encoded testing shape :", Z_test.shape)

    # Build grouped sliding-window sequences.
    # label_mode="any": a window is labeled attack if any timestep in it is
    # under attack. With device-level group splits (no row shuffling) this
    # creates genuine early-detection windows where the last row is still
    # near-normal but the sequence already contains the onset of the attack.
    label_mode = "any" if dataset_name.lower() == "sim" else "last"

    train_dataset = ArraySequenceDataset(
        features=Z_train,
        labels=y_train,
        group_ids=group_ids_train,
        seq_len=cfg.seq_len,
        label_mode=label_mode
    )

    test_dataset = ArraySequenceDataset(
        features=Z_test,
        labels=y_test,
        group_ids=group_ids_test,
        seq_len=cfg.seq_len,
        label_mode=label_mode
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.clf_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.clf_batch_size, shuffle=False)

    print("Number of training sequences:", len(train_dataset))
    print("Number of testing sequences :", len(test_dataset))

    # Build Mamba classifier
    model = MambaClassifier(
        input_dim=cfg.latent_dim + 1,   # latent features + reconstruction error
        d_model=cfg.d_model,
        n_layers=cfg.num_layers,
        dropout=cfg.dropout
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.clf_learning_rate)

    # Training loop
    epoch_losses = []

    for epoch in range(cfg.clf_epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{cfg.clf_epochs}] Loss: {avg_loss:.6f}")
        torch.cuda.empty_cache()

    # Save classifier
    clf_path = cfg.ton_classifier_model_path if dataset_name.lower() == "ton" else cfg.sim_classifier_model_path
    torch.save(model.state_dict(), clf_path)
    print(f"Mamba classifier saved to: {clf_path}")

    os.makedirs(cfg.loss_dir, exist_ok=True)
    loss_path = os.path.join(cfg.loss_dir, f"mamba_{dataset_name}_losses.npy")
    np.save(loss_path, np.array(epoch_losses))
    print(f"Loss history saved to: {loss_path}")


if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "ton"
    main(dataset)