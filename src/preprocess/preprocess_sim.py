import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.config import Config


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    cfg = Config()

    ensure_dir(cfg.sim_splits_path)
    ensure_dir(cfg.processed_icu_path)

    data_path = os.path.join(cfg.raw_icu_path, "icu_simulation.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Simulation data not found at {data_path}. "
            "Run simulate_icu.py first."
        )

    df = pd.read_csv(data_path)
    print("Loaded simulation data:", df.shape)
    print("Label distribution:\n", df["label"].value_counts())

    df = df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)

    group_ids = df["device_id"].values
    y = df["label"].values

    drop_cols = ["label", "timestamp", "device_id", "device_type"]
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X_df = X_df.replace([float("inf"), float("-inf")], float("nan")).fillna(0)

    # Split 1: trainval vs test (85% / 15%)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_seed)
    trainval_idx, test_idx = next(gss1.split(X_df, y, groups=group_ids))

    X_trainval    = X_df.iloc[trainval_idx]
    y_trainval    = y[trainval_idx]
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

    joblib.dump(scaler, os.path.join(cfg.processed_icu_path, "scaler.pkl"))

    sp = cfg.sim_splits_path
    np.save(os.path.join(sp, "X_train.npy"),         X_train)
    np.save(os.path.join(sp, "X_val.npy"),           X_val)
    np.save(os.path.join(sp, "X_test.npy"),          X_test)
    np.save(os.path.join(sp, "y_train.npy"),         y_train)
    np.save(os.path.join(sp, "y_val.npy"),           y_val)
    np.save(os.path.join(sp, "y_test.npy"),          y_test)
    np.save(os.path.join(sp, "group_ids_train.npy"), np.array(group_ids_train, dtype=object))
    np.save(os.path.join(sp, "group_ids_val.npy"),   np.array(group_ids_val,   dtype=object))
    np.save(os.path.join(sp, "group_ids_test.npy"),  np.array(group_ids_test,  dtype=object))

    print("Preprocessing complete.")
    print(f"Train : {X_train.shape}  — 0:{(y_train==0).sum()}  1:{(y_train==1).sum()}")
    print(f"Val   : {X_val.shape}    — 0:{(y_val==0).sum()}    1:{(y_val==1).sum()}")
    print(f"Test  : {X_test.shape}   — 0:{(y_test==0).sum()}   1:{(y_test==1).sum()}")


if __name__ == "__main__":
    main()
