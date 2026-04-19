import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.config import Config

MAX_BENIGN = 25_000
MAX_ATTACK = 25_000
CHUNK_SIZE = 50  # rows per temporal group (no device/IP identifier in CIC)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_cic_csvs(cfg):
    dfs = []
    sources = [
        (cfg.cic_wifi_mqtt_path, "WiFi/MQTT"),
        (cfg.cic_bluetooth_path, "Bluetooth"),
    ]
    for folder, protocol in sources:
        if not os.path.exists(folder):
            continue
        for fpath in glob.glob(os.path.join(folder, "*.csv")):
            fname = os.path.basename(fpath).lower()
            df = pd.read_csv(fpath, low_memory=False)
            df["label"] = 0 if "benign" in fname else 1
            df["protocol_group"] = protocol
            dfs.append(df)
            print(f"  Loaded {os.path.basename(fpath)}: {len(df):,} rows")

    if not dfs:
        raise FileNotFoundError(
            f"No CICIoMT2024 CSVs found in {cfg.cic_wifi_mqtt_path} or {cfg.cic_bluetooth_path}"
        )
    return pd.concat(dfs, ignore_index=True)


def main():
    cfg = Config()
    ensure_dir(cfg.cic_splits_path)
    ensure_dir(cfg.processed_cic_path)

    print("Loading CICIoMT2024 CSVs...")
    df = load_cic_csvs(cfg)
    print(f"Total rows: {len(df):,}")
    print("Label distribution:\n", df["label"].value_counts())

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Drop non-feature columns
    drop_cols = ["label", "protocol_group"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_df = df[feature_cols].copy()
    y = df["label"].values

    # Cap class sizes to keep training tractable
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]

    rng = np.random.default_rng(cfg.random_seed)
    if len(benign_idx) > MAX_BENIGN:
        benign_idx = rng.choice(benign_idx, MAX_BENIGN, replace=False)
    if len(attack_idx) > MAX_ATTACK:
        attack_idx = rng.choice(attack_idx, MAX_ATTACK, replace=False)

    keep_idx = np.sort(np.concatenate([benign_idx, attack_idx]))
    X_df = X_df.iloc[keep_idx].reset_index(drop=True)
    y = y[keep_idx]

    print(f"After capping — benign: {(y == 0).sum():,}  attack: {(y == 1).sum():,}")

    # Temporal chunk grouping (no device/IP columns)
    group_ids = np.arange(len(X_df)) // CHUNK_SIZE
    print(f"Temporal groups: {group_ids.max() + 1} chunks of ~{CHUNK_SIZE} rows")

    # Split 1: trainval vs test (85% / 15%)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_seed)
    trainval_idx, test_idx = next(gss1.split(X_df, y, groups=group_ids))

    X_trainval      = X_df.iloc[trainval_idx]
    y_trainval      = y[trainval_idx]
    groups_trainval = group_ids[trainval_idx]

    # Split 2: train vs val (≈70% / 15% of total)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=cfg.validation_size, random_state=cfg.random_seed)
    train_rel, val_rel = next(gss2.split(X_trainval, y_trainval, groups=groups_trainval))

    X_train_df = X_trainval.iloc[train_rel]
    X_val_df   = X_trainval.iloc[val_rel]
    X_test_df  = X_df.iloc[test_idx]

    y_train = y_trainval[train_rel]
    y_val   = y_trainval[val_rel]
    y_test  = y[test_idx]

    group_ids_train = groups_trainval[train_rel]
    group_ids_val   = groups_trainval[val_rel]
    group_ids_test  = group_ids[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_val   = scaler.transform(X_val_df)
    X_test  = scaler.transform(X_test_df)

    joblib.dump(scaler, os.path.join(cfg.processed_cic_path, "scaler.pkl"))

    sp = cfg.cic_splits_path
    np.save(os.path.join(sp, "X_train.npy"),         X_train)
    np.save(os.path.join(sp, "X_val.npy"),           X_val)
    np.save(os.path.join(sp, "X_test.npy"),          X_test)
    np.save(os.path.join(sp, "y_train.npy"),         y_train)
    np.save(os.path.join(sp, "y_val.npy"),           y_val)
    np.save(os.path.join(sp, "y_test.npy"),          y_test)
    np.save(os.path.join(sp, "group_ids_train.npy"), group_ids_train)
    np.save(os.path.join(sp, "group_ids_val.npy"),   group_ids_val)
    np.save(os.path.join(sp, "group_ids_test.npy"),  group_ids_test)

    print("Preprocessing complete.")
    print(f"Train : {X_train.shape}  — 0:{(y_train==0).sum():,}  1:{(y_train==1).sum():,}")
    print(f"Val   : {X_val.shape}    — 0:{(y_val==0).sum():,}    1:{(y_val==1).sum():,}")
    print(f"Test  : {X_test.shape}   — 0:{(y_test==0).sum():,}   1:{(y_test==1).sum():,}")


if __name__ == "__main__":
    main()
