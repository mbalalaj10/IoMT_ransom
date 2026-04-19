import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.utils import set_seed, get_device
from src.datasets.sequence_dataset import ArraySequenceDataset
from src.models.autoencoder import Autoencoder
from src.models.mamba_classifier import MambaClassifier
from src.models.lstm_classifier import LSTMClassifier
from src.evaluate.metrics import compute_metrics, print_metrics, find_optimal_threshold


def extract_latent_and_error(ae, X, device, batch_size=256):
    ae.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            z, x_hat = ae(batch)
            recon_error = torch.mean((batch - x_hat) ** 2, dim=1, keepdim=True)
            parts.append(torch.cat([z, recon_error], dim=1).cpu().numpy())
    return np.vstack(parts)


def run_inference(model, loader, threshold, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y_batch.numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred = (y_prob >= threshold).astype(int)
    return y_true, y_pred, y_prob


def main():
    cfg = Config()
    set_seed(cfg.random_seed)
    device = get_device()
    print("Using device:", device)

    X_test = np.load(os.path.join(cfg.cic_splits_path, "X_test.npy"))
    y_test = np.load(os.path.join(cfg.cic_splits_path, "y_test.npy"))
    group_ids_test = np.load(os.path.join(cfg.cic_splits_path, "group_ids_test.npy"), allow_pickle=True)

    ae = Autoencoder(
        input_dim=X_test.shape[1],
        hidden_dim1=cfg.ae_hidden_dim1,
        hidden_dim2=cfg.ae_hidden_dim2,
        latent_dim=cfg.latent_dim,
    ).to(device)
    ae.load_state_dict(torch.load(cfg.cic_autoencoder_model_path, map_location=device))
    ae.eval()

    Z_test = extract_latent_and_error(ae, X_test, device)
    print("Encoded test shape:", Z_test.shape)

    test_dataset = ArraySequenceDataset(
        features=Z_test,
        labels=y_test,
        group_ids=group_ids_test,
        seq_len=cfg.seq_len,
        label_mode="last",
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.clf_batch_size, shuffle=False)
    print("Number of test sequences:", len(test_dataset))

    # --- Mamba ---
    mamba = MambaClassifier(
        input_dim=cfg.latent_dim + 1,
        d_model=32,
        n_layers=1,
        dropout=cfg.dropout,
    ).to(device)
    mamba.load_state_dict(torch.load(cfg.cic_classifier_model_path, map_location=device))
    y_true, y_pred, y_prob = run_inference(mamba, test_loader, cfg.threshold, device)
    thresh_mamba = find_optimal_threshold(y_true, y_prob)
    print(f"Optimal threshold — Mamba: {thresh_mamba:.2f}")
    y_pred = (y_prob >= thresh_mamba).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, dataset_label="CICIoMT2024 — Mamba", threshold=thresh_mamba)

    # --- LSTM ---
    lstm = LSTMClassifier(
        input_dim=cfg.latent_dim + 1,
        hidden_dim=32,
        num_layers=1,
        dropout=cfg.dropout,
    ).to(device)
    lstm.load_state_dict(torch.load(cfg.cic_lstm_model_path, map_location=device))
    y_true, y_pred, y_prob = run_inference(lstm, test_loader, cfg.threshold, device)
    thresh_lstm = find_optimal_threshold(y_true, y_prob)
    print(f"Optimal threshold — LSTM : {thresh_lstm:.2f}")
    y_pred = (y_prob >= thresh_lstm).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, dataset_label="CICIoMT2024 — LSTM", threshold=thresh_lstm)


if __name__ == "__main__":
    main()
