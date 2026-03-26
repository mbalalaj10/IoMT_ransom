"""
Early Detection Analysis

Measures how quickly Mamba, LSTM, and a per-row LR baseline detect a
ransomware attack after it begins on each ICU device.

Detection is defined as: the first sliding-window sequence that the model
classifies as an attack.  For Mamba and LSTM the window spans seq_len=20
rows; for LR the "window" is just the last row (per-row inference).

"""

import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.config import Config
from src.utils import set_seed, get_device
from src.models.autoencoder import Autoencoder
from src.models.mamba_classifier import MambaClassifier
from src.models.lstm_classifier import LSTMClassifier


ATTACK_START = 200   # must match simulate_icu.py


def extract_latent_and_error(ae, X, device, batch_size=256):
    ae.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            z, x_hat = ae(batch)
            err = torch.mean((batch - x_hat) ** 2, dim=1, keepdim=True)
            out.append(torch.cat([z, err], dim=1).cpu().numpy())
    return np.vstack(out)


def sliding_windows(Z, seq_len):
    """Return (N - seq_len + 1, seq_len, F) array of sliding windows."""
    N, F = Z.shape
    if N < seq_len:
        return np.empty((0, seq_len, F), dtype=Z.dtype)
    idx = np.arange(seq_len)[None, :] + np.arange(N - seq_len + 1)[:, None]
    return Z[idx]


def first_detection_timestep(preds, seq_len):
    """
    preds : binary array of length (N - seq_len + 1)
             preds[i] corresponds to the window ending at timestep i + seq_len - 1
    Returns the timestep of the first positive prediction, or None.
    """
    hits = np.where(preds == 1)[0]
    if len(hits) == 0:
        return None
    return hits[0] + seq_len - 1   # last timestep of that window


def seq_model_preds(model, windows, device, threshold):
    """Run a sequence model (Mamba or LSTM) over pre-built windows tensor."""
    X_tensor = torch.tensor(windows, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        return (torch.sigmoid(logits) >= threshold).cpu().numpy().astype(int)


def ascii_chart(title, x_vals, series, width=30):
    """series: list of (label, values) pairs."""
    headers = "".join(f"  {lbl:>7}" for lbl, _ in series)
    bars    = "".join(f"  {lbl:^{width}}" for lbl, _ in series)
    print(f"\n  {title}")
    print(f"  {'Step':>5}{headers}  " + "  ".join(f"{'#':^{width}}" for _ in series))
    print(f"  {'-'*5}" + "  -------" * len(series))
    for i, x in enumerate(x_vals):
        vals_str = "".join(f"  {v[i]:>6.1%}" for _, v in series)
        bar_str  = "  ".join(f"{'#' * int(v[i] * width):{width}}" for _, v in series)
        print(f"  {x:>5}{vals_str}  {bar_str}")


def main():
    cfg = Config()
    set_seed(cfg.random_seed)
    device = get_device()
    seq_len = cfg.seq_len

    split_dir = cfg.sim_splits_path

    X_train         = np.load(os.path.join(split_dir, "X_train.npy"))
    X_test          = np.load(os.path.join(split_dir, "X_test.npy"))
    y_train         = np.load(os.path.join(split_dir, "y_train.npy"))
    y_test          = np.load(os.path.join(split_dir, "y_test.npy"))
    group_ids_train = np.load(os.path.join(split_dir, "group_ids_train.npy"), allow_pickle=True)
    group_ids_test  = np.load(os.path.join(split_dir, "group_ids_test.npy"),  allow_pickle=True)

    # ── load models ───────────────────────────────────────────────────────────
    ae = Autoencoder(
        input_dim=X_train.shape[1],
        hidden_dim1=cfg.ae_hidden_dim1,
        hidden_dim2=cfg.ae_hidden_dim2,
        latent_dim=cfg.latent_dim,
    ).to(device)
    ae.load_state_dict(torch.load(cfg.sim_autoencoder_model_path, map_location=device))
    ae.eval()

    mamba = MambaClassifier(
        input_dim=cfg.latent_dim + 1,
        d_model=cfg.d_model,
        n_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    mamba.load_state_dict(torch.load(cfg.sim_classifier_model_path, map_location=device))
    mamba.eval()

    lstm = LSTMClassifier(
        input_dim=cfg.latent_dim + 1,
        hidden_dim=cfg.d_model,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    lstm.load_state_dict(torch.load(cfg.sim_lstm_model_path, map_location=device))
    lstm.eval()

    print("Encoding features...")
    Z_train = extract_latent_and_error(ae, X_train, device)
    Z_test  = extract_latent_and_error(ae, X_test,  device)

    # ── train LR on last-row features of training sequences ───────────────────
    unique_train, inv_train = np.unique(group_ids_train, return_inverse=True)
    train_last, train_labels = [], []
    for gi in range(len(unique_train)):
        rows = np.where(inv_train == gi)[0]
        if len(rows) < seq_len:
            continue
        windows   = sliding_windows(Z_train[rows], seq_len)
        last_rows = windows[:, -1, :]
        y_rows    = y_train[rows]
        labels    = np.array([int(np.any(y_rows[j:j + seq_len]))
                              for j in range(len(rows) - seq_len + 1)])
        train_last.append(last_rows)
        train_labels.append(labels)

    lr = LogisticRegression(max_iter=1000, random_state=cfg.random_seed)
    lr.fit(np.vstack(train_last), np.concatenate(train_labels))

    # ── per-device early detection on test set ────────────────────────────────
    unique_test, inv_test = np.unique(group_ids_test, return_inverse=True)

    mamba_lags,  lstm_lags,  lr_lags  = [], [], []
    mamba_missed, lstm_missed, lr_missed = 0, 0, 0
    n_attacked = 0

    for gi in range(len(unique_test)):
        rows   = np.where(inv_test == gi)[0]
        dev_id = unique_test[gi]

        if "attack" not in str(dev_id):
            continue

        n_attacked += 1
        Z_dev = Z_test[rows]

        if len(Z_dev) < seq_len:
            mamba_missed += 1
            lstm_missed  += 1
            lr_missed    += 1
            continue

        windows = sliding_windows(Z_dev, seq_len)   # (W, seq_len, F)

        mamba_preds = seq_model_preds(mamba, windows, device, cfg.threshold)
        lstm_preds  = seq_model_preds(lstm,  windows, device, cfg.threshold)
        lr_preds    = lr.predict(windows[:, -1, :])

        for model_name, preds, lags, missed_ref in [
            ("Mamba", mamba_preds, mamba_lags, None),
            ("LSTM",  lstm_preds,  lstm_lags,  None),
            ("LR",    lr_preds,    lr_lags,    None),
        ]:
            t = first_detection_timestep(preds, seq_len)
            if t is None:
                if model_name == "Mamba":  mamba_missed += 1
                elif model_name == "LSTM": lstm_missed  += 1
                else:                      lr_missed    += 1
            else:
                lag = t - ATTACK_START
                if lag < 0:
                    print(f"  [FP] {model_name} false alarm on {dev_id} "
                          f"at t={t} ({lag} steps before attack)")
                    if model_name == "Mamba":  mamba_missed += 1
                    elif model_name == "LSTM": lstm_missed  += 1
                    else:                      lr_missed    += 1
                else:
                    lags.append(lag)

    # ── summary statistics ────────────────────────────────────────────────────
    print(f"\nAttacked devices in test set : {n_attacked}")
    print(f"Mamba — detected: {len(mamba_lags)}/{n_attacked}, missed: {mamba_missed}")
    print(f"LSTM  — detected: {len(lstm_lags)}/{n_attacked},  missed: {lstm_missed}")
    print(f"LR    — detected: {len(lr_lags)}/{n_attacked},    missed: {lr_missed}")

    for label, lags in [("Mamba", mamba_lags), ("LSTM", lstm_lags), ("LR", lr_lags)]:
        if lags:
            print(f"\n{label} detection lag (steps after attack_start):")
            print(f"  Mean   : {np.mean(lags):.1f}")
            print(f"  Median : {np.median(lags):.1f}")
            print(f"  Min    : {np.min(lags)}")
            print(f"  Max    : {np.max(lags)}")

    # ── cumulative detection rate table ───────────────────────────────────────
    max_steps = 100
    steps = list(range(0, max_steps + 1, 5))

    mamba_cum = [sum(1 for l in mamba_lags if l <= s) / n_attacked for s in steps]
    lstm_cum  = [sum(1 for l in lstm_lags  if l <= s) / n_attacked for s in steps]
    lr_cum    = [sum(1 for l in lr_lags    if l <= s) / n_attacked for s in steps]

    print("\n\nCumulative Detection Rate vs Steps After Attack Onset")
    print(f"  {'Steps':>5}  {'Mamba':>7}  {'LSTM':>7}  {'LR':>7}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}")
    for s, mv, lstmv, lv in zip(steps, mamba_cum, lstm_cum, lr_cum):
        print(f"  {s:>5}  {mv:>6.1%}  {lstmv:>6.1%}  {lv:>6.1%}")

    ascii_chart(
        "Detection Rate (# = % detected)",
        steps,
        [("Mamba", mamba_cum), ("LSTM", lstm_cum), ("LR", lr_cum)],
        width=25,
    )


if __name__ == "__main__":
    main()
