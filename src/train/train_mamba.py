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
    model.eval()
    combined_features = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            z, x_hat = model(batch)
            recon_error = torch.mean((batch - x_hat) ** 2, dim=1, keepdim=True)
            combined_features.append(torch.cat([z, recon_error], dim=1).cpu().numpy())
    return np.vstack(combined_features)


def main(dataset_name="ton"):
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

    # Lighter hyperparameters for CPU-only CIC training
    if dataset_name.lower() == "cic":
        d_model    = 32
        n_layers   = 1
        clf_epochs = 8
        batch_size = 128
    else:
        d_model    = cfg.d_model
        n_layers   = cfg.num_layers
        clf_epochs = cfg.clf_epochs
        batch_size = cfg.clf_batch_size

    X_train = np.load(os.path.join(split_dir, "X_train.npy"))
    X_val   = np.load(os.path.join(split_dir, "X_val.npy"))
    X_test  = np.load(os.path.join(split_dir, "X_test.npy"))
    y_train = np.load(os.path.join(split_dir, "y_train.npy"))
    y_val   = np.load(os.path.join(split_dir, "y_val.npy"))
    y_test  = np.load(os.path.join(split_dir, "y_test.npy"))

    group_ids_train = np.load(os.path.join(split_dir, "group_ids_train.npy"), allow_pickle=True)
    group_ids_val   = np.load(os.path.join(split_dir, "group_ids_val.npy"),   allow_pickle=True)
    group_ids_test  = np.load(os.path.join(split_dir, "group_ids_test.npy"),  allow_pickle=True)

    unique, counts = np.unique(group_ids_train, return_counts=True)
    viable = np.sum(counts >= cfg.seq_len)
    print(f"Train groups: {len(unique)} total, {viable} viable for sequences")

    device = get_device()
    print(f"Using device: {device}  |  dataset: {dataset_name}")

    ae_path = {
        "ton": cfg.ton_autoencoder_model_path,
        "sim": cfg.sim_autoencoder_model_path,
        "cic": cfg.cic_autoencoder_model_path,
    }[dataset_name.lower()]

    ae = Autoencoder(
        input_dim=X_train.shape[1],
        hidden_dim1=cfg.ae_hidden_dim1,
        hidden_dim2=cfg.ae_hidden_dim2,
        latent_dim=cfg.latent_dim,
    ).to(device)
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.eval()

    Z_train = extract_latent_and_error(ae, X_train, device)
    Z_val   = extract_latent_and_error(ae, X_val,   device)
    Z_test  = extract_latent_and_error(ae, X_test,  device)
    print(f"Encoded  train:{Z_train.shape}  val:{Z_val.shape}  test:{Z_test.shape}")

    label_mode = "any" if dataset_name.lower() == "sim" else "last"

    train_dataset = ArraySequenceDataset(Z_train, y_train, group_ids_train, cfg.seq_len, label_mode)
    val_dataset   = ArraySequenceDataset(Z_val,   y_val,   group_ids_val,   cfg.seq_len, label_mode)
    test_dataset  = ArraySequenceDataset(Z_test,  y_test,  group_ids_test,  cfg.seq_len, label_mode)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    print(f"Sequences  train:{len(train_dataset)}  val:{len(val_dataset)}  test:{len(test_dataset)}")

    model = MambaClassifier(
        input_dim=cfg.latent_dim + 1,
        d_model=d_model,
        n_layers=n_layers,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.clf_learning_rate)

    clf_path = {
        "ton": cfg.ton_classifier_model_path,
        "sim": cfg.sim_classifier_model_path,
        "cic": cfg.cic_classifier_model_path,
    }[dataset_name.lower()]

    best_val_loss    = float("inf")
    patience_counter = 0
    train_losses     = []

    for epoch in range(clf_epochs):
        model.train()
        total = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss = total / len(train_loader)
        train_losses.append(train_loss)
        torch.cuda.empty_cache()

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_total += criterion(model(X_batch.to(device)), y_batch.to(device)).item()
        val_loss = val_total / len(val_loader)

        print(f"Epoch [{epoch+1}/{clf_epochs}]  Train: {train_loss:.6f}  Val: {val_loss:.6f}", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), clf_path)
            patience_counter = 0
            print("  ✓ saved")
        else:
            patience_counter += 1
            print(f"  (patience {patience_counter}/{cfg.early_stopping_patience})")
            if patience_counter >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    print(f"Best val loss: {best_val_loss:.6f}  →  {clf_path}")

    os.makedirs(cfg.loss_dir, exist_ok=True)
    np.save(os.path.join(cfg.loss_dir, f"mamba_{dataset_name}_losses.npy"), np.array(train_losses))


if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "ton"
    main(dataset)
