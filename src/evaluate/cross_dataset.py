"""
Cross-dataset evaluation: latent-space transfer.

Each dataset's own AE encodes its test data into the shared 33-dim latent
space (latent_dim=32 + reconstruction error). A Mamba classifier trained on a
DIFFERENT dataset is then applied. Off-diagonal cells in the results table
measure how well the learned decision boundary generalises across datasets.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.utils import set_seed, get_device
from src.datasets.sequence_dataset import ArraySequenceDataset
from src.models.autoencoder import Autoencoder
from src.models.mamba_classifier import MambaClassifier
from src.evaluate.metrics import compute_metrics, find_optimal_threshold

# CIC was trained with a lighter architecture
DATASET_ARCH = {
    "sim": {"d_model": 64, "n_layers": 2},
    "ton": {"d_model": 64, "n_layers": 2},
    "cic": {"d_model": 32, "n_layers": 1},
}


def get_paths(cfg, name):
    return {
        "sim": {
            "splits":      cfg.sim_splits_path,
            "ae_model":    cfg.sim_autoencoder_model_path,
            "clf_model":   cfg.sim_classifier_model_path,
            "label_mode":  "any",
        },
        "ton": {
            "splits":      cfg.ton_splits_path,
            "ae_model":    cfg.ton_autoencoder_model_path,
            "clf_model":   cfg.ton_classifier_model_path,
            "label_mode":  "last",
        },
        "cic": {
            "splits":      cfg.cic_splits_path,
            "ae_model":    cfg.cic_autoencoder_model_path,
            "clf_model":   cfg.cic_classifier_model_path,
            "label_mode":  "last",
        },
    }[name]


def encode(ae, X, device, batch_size=256):
    ae.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            z, x_hat = ae(batch)
            recon_err = torch.mean((batch - x_hat) ** 2, dim=1, keepdim=True)
            parts.append(torch.cat([z, recon_err], dim=1).cpu().numpy())
    return np.vstack(parts)


def evaluate(clf, loader, device):
    clf.eval()
    probs, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = clf(X_batch.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y_batch.numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels)
    threshold = find_optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return compute_metrics(y_true, y_pred, y_prob)


def main():
    cfg = Config()
    set_seed(cfg.random_seed)
    device = get_device()
    print("Using device:", device)

    datasets = ["sim", "ton", "cic"]
    results = {}

    for test_ds in datasets:
        tp = get_paths(cfg, test_ds)

        # Load and encode test split with the test dataset's own AE
        X_test = np.load(os.path.join(tp["splits"], "X_test.npy"))
        y_test = np.load(os.path.join(tp["splits"], "y_test.npy"))
        g_test = np.load(os.path.join(tp["splits"], "group_ids_test.npy"), allow_pickle=True)

        ae = Autoencoder(
            input_dim=X_test.shape[1],
            hidden_dim1=cfg.ae_hidden_dim1,
            hidden_dim2=cfg.ae_hidden_dim2,
            latent_dim=cfg.latent_dim,
        ).to(device)
        ae.load_state_dict(torch.load(tp["ae_model"], map_location=device))

        Z_test = encode(ae, X_test, device)

        seq_dataset = ArraySequenceDataset(
            features=Z_test, labels=y_test,
            group_ids=g_test, seq_len=cfg.seq_len,
            label_mode=tp["label_mode"],
        )
        loader = DataLoader(seq_dataset, batch_size=cfg.clf_batch_size, shuffle=False)
        print(f"\n[test={test_ds}] {len(seq_dataset)} sequences encoded ({Z_test.shape[1]}-dim)")

        for train_ds in datasets:
            arch = DATASET_ARCH[train_ds]
            cp = get_paths(cfg, train_ds)

            clf = MambaClassifier(
                input_dim=cfg.latent_dim + 1,
                d_model=arch["d_model"],
                n_layers=arch["n_layers"],
                dropout=cfg.dropout,
            ).to(device)
            clf.load_state_dict(torch.load(cp["clf_model"], map_location=device))

            m = evaluate(clf, loader, device)
            results[(train_ds, test_ds)] = m

            tag = "within" if train_ds == test_ds else "CROSS"
            print(f"  train={train_ds} → test={test_ds} [{tag}]  "
                  f"F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  "
                  f"Recall={m['recall']:.4f}  Precision={m['precision']:.4f}")

    # Summary table
    print("\n" + "=" * 65)
    print(" Cross-Dataset Transfer — F1 Score (rows=train, cols=test)")
    print("=" * 65)
    header = f"{'':>10}" + "".join(f"  {d:>8}" for d in datasets)
    print(header)
    for train_ds in datasets:
        row = f"{train_ds:>10}"
        for test_ds in datasets:
            f1 = results[(train_ds, test_ds)]["f1"]
            marker = " *" if train_ds == test_ds else "  "
            row += f"  {f1:>6.4f}{marker}"
        print(row)
    print("=" * 65)
    print("* = within-dataset (diagonal)")

    print("\n" + "=" * 65)
    print(" Cross-Dataset Transfer — AUC-ROC (rows=train, cols=test)")
    print("=" * 65)
    print(header)
    for train_ds in datasets:
        row = f"{train_ds:>10}"
        for test_ds in datasets:
            auc = results[(train_ds, test_ds)]["auc_roc"]
            marker = " *" if train_ds == test_ds else "  "
            row += f"  {auc:>6.4f}{marker}"
        print(row)
    print("=" * 65)
    print("* = within-dataset (diagonal)")


if __name__ == "__main__":
    main()
