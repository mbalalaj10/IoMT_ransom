import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.utils import set_seed, get_device
from src.datasets.sequence_dataset import ArraySequenceDataset
from src.models.autoencoder import Autoencoder
from src.models.mamba_classifier import MambaClassifier
from src.evaluate.metrics import compute_metrics, print_metrics, find_optimal_threshold


def extract_latent_and_error(model, X, device, batch_size=256):
    model.eval()
    combined_features = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            z, x_hat = model(batch)
            recon_error = torch.mean((batch - x_hat) ** 2, dim=1, keepdim=True)
            combined = torch.cat([z, recon_error], dim=1)
            combined_features.append(combined.cpu().numpy())

    return np.vstack(combined_features)


def main():
    cfg = Config()
    set_seed(cfg.random_seed)
    device = get_device()
    print("Using device:", device)

    split_dir = cfg.ton_splits_path

    X_test  = np.load(os.path.join(split_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(split_dir, "y_test.npy"))
    group_ids_test = np.load(os.path.join(split_dir, "group_ids_test.npy"), allow_pickle=True)

    # Load autoencoder
    ae = Autoencoder(
        input_dim=X_test.shape[1],
        hidden_dim1=cfg.ae_hidden_dim1,
        hidden_dim2=cfg.ae_hidden_dim2,
        latent_dim=cfg.latent_dim,
    ).to(device)
    ae.load_state_dict(torch.load(cfg.ton_autoencoder_model_path, map_location=device))
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

    # Load classifier
    model = MambaClassifier(
        input_dim=cfg.latent_dim + 1,
        d_model=cfg.d_model,
        n_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(torch.load(cfg.ton_classifier_model_path, map_location=device))
    model.eval()

    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits  = model(X_batch)
            probs   = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y_batch.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    threshold = find_optimal_threshold(y_true, y_prob)
    print(f"Optimal threshold (max F1): {threshold:.2f}")

    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, dataset_label="TON-IoT Test", threshold=threshold)


if __name__ == "__main__":
    main()
