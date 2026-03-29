"""
Exploratory analysis of the CICIoMT2024 dataset.

Produces (per SRS FR-101 to FR-108, FR-307, FR-801, FR-804):
  results/figures/cic_label_distribution.png
  results/figures/cic_protocol_composition.png
  results/figures/cic_correlation_heatmap.png
  results/figures/cic_feature_distributions.png
  results/figures/cic_anomaly_gap.png
  results/figures/cic_feature_importance.png
  results/figures/cic_feature_summary.csv
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

from src.config import Config

FIG_DIR = "results/figures"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_cic_data(cfg):
    """Load all CICIoMT2024 CSVs; assign label and protocol from filename/folder."""
    dfs = []
    sources = [
        (cfg.cic_wifi_mqtt_path, "WiFi/MQTT"),
        (cfg.cic_bluetooth_path, "Bluetooth"),
    ]
    for folder, protocol_name in sources:
        if not os.path.exists(folder):
            continue
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        for fpath in csv_files:
            fname = os.path.basename(fpath).lower()
            df = pd.read_csv(fpath)
            df["label"] = 0 if "benign" in fname else 1
            raw_name = os.path.basename(fpath)
            for suffix in ("_train.pcap.csv", "_test.pcap.csv", ".pcap.csv", ".csv"):
                raw_name = raw_name.replace(suffix, "")
            df["attack_type"] = "Benign" if "benign" in fname else raw_name
            df["protocol_group"] = protocol_name
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            "No CICIoMT2024 CSV files found. "
            f"Expected files in: {cfg.cic_wifi_mqtt_path} or {cfg.cic_bluetooth_path}"
        )

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.replace([float("inf"), float("-inf")], float("nan")).fillna(0)

    print(f"Loaded CICIoMT2024: {df_all.shape[0]:,} rows, {df_all.shape[1]} columns")
    print(f"\nLabel distribution:")
    counts = df_all["attack_type"].value_counts()
    for name, count in counts.items():
        print(f"  {name:<40} {count:>8,}  ({count / len(df_all) * 100:.1f}%)")
    return df_all


def get_feature_cols(df):
    exclude = {"label", "attack_type", "protocol_group"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


# FR-101, FR-102: Label distribution
def plot_label_distribution(df, fig_dir):
    counts = df["attack_type"].value_counts()
    colors = ["#2ecc71" if v == "Benign" else "#e74c3c" for v in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(len(counts)), counts.values, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(range(len(counts)))
    axes[0].set_xticklabels(counts.index, rotation=45, ha="right", fontsize=8)
    axes[0].set_title("Label Distribution — CICIoMT2024", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Traffic Class")
    axes[0].set_ylabel("Sample Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + counts.max() * 0.01, f"{v:,}", ha="center", fontsize=7)

    pct = counts / counts.sum() * 100
    wedge_colors = ["#2ecc71" if v == "Benign" else "#e74c3c" for v in pct.index]
    axes[1].pie(pct.values, labels=pct.index, autopct="%1.1f%%",
                colors=wedge_colors, startangle=140, textprops={"fontsize": 8})
    axes[1].set_title("Class Proportions — CICIoMT2024", fontsize=13, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(fig_dir, "cic_label_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# FR-103, FR-104: Protocol composition
def plot_protocol_composition(df, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pg_counts = df["protocol_group"].value_counts()
    axes[0].bar(pg_counts.index, pg_counts.values,
                color=["#3498db", "#9b59b6"][:len(pg_counts)], edgecolor="black")
    axes[0].set_title("Protocol Group Composition — CICIoMT2024", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Protocol Group")
    axes[0].set_ylabel("Sample Count")
    for i, v in enumerate(pg_counts.values):
        axes[0].text(i, v + pg_counts.max() * 0.01, f"{v:,}", ha="center", fontsize=9)

    proto_label = df.groupby(["protocol_group", "label"]).size().unstack(fill_value=0)
    proto_label.columns = ["Benign", "Attack"] if 0 in proto_label.columns else proto_label.columns
    proto_label.plot(kind="bar", ax=axes[1], color=["#2ecc71", "#e74c3c"], edgecolor="black")
    axes[1].set_title("Benign vs Attack by Protocol Group", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Protocol Group")
    axes[1].set_ylabel("Sample Count")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Class")

    plt.tight_layout()
    out = os.path.join(fig_dir, "cic_protocol_composition.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# FR-105: Per-feature summary stats (printed)
def print_feature_stats(df, feat_cols):
    benign = df[df["label"] == 0][feat_cols]
    attack = df[df["label"] == 1][feat_cols]
    diff = (attack.mean() - benign.mean()).abs().sort_values(ascending=False).head(10)
    print("\nTop 10 features by |benign_mean - attack_mean|:")
    print(f"  {'Feature':<30} {'Benign Mean':>12} {'Attack Mean':>12} {'|Diff|':>10}")
    print(f"  {'-'*68}")
    for feat in diff.index:
        print(f"  {feat:<30} {benign[feat].mean():>12.4f} {attack[feat].mean():>12.4f} {diff[feat]:>10.4f}")


# FR-106: Correlation heatmap
def plot_correlation_heatmap(df, feat_cols, fig_dir):
    corr = df[feat_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix — CICIoMT2024", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)

    plt.tight_layout()
    out = os.path.join(fig_dir, "cic_correlation_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.90:
                high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    if high_corr:
        print(f"\n  Highly correlated pairs (|r| > 0.90): {len(high_corr)} found")
        for a, b, r in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[:10]:
            print(f"    {a} <-> {b}: r={r:.3f}")


# FR-801: Per-class feature distribution (box plots, top discriminative features)
def plot_feature_distributions(df, feat_cols, fig_dir, top_n=8):
    benign = df[df["label"] == 0][feat_cols]
    attack = df[df["label"] == 1][feat_cols]
    top_feats = (attack.mean() - benign.mean()).abs().sort_values(ascending=False).head(top_n).index.tolist()

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, feat in enumerate(top_feats):
        data = [df[df["label"] == 0][feat].values, df[df["label"] == 1][feat].values]
        bp = axes[i].boxplot(data, patch_artist=True, labels=["Benign", "Attack"], showfliers=False)
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][1].set_facecolor("#e74c3c")
        axes[i].set_title(feat, fontsize=9, fontweight="bold")
        axes[i].set_ylabel("Value")

    for j in range(len(top_feats), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top Discriminative Features — Benign vs Attack (CICIoMT2024)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(fig_dir, "cic_feature_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# FR-307, FR-801: Anomaly gap — proxy reconstruction error (distance from benign mean)
def plot_anomaly_gap(df, feat_cols, fig_dir):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feat_cols].values)
    benign_mean = X[df["label"].values == 0].mean(axis=0)
    scores = np.mean((X - benign_mean) ** 2, axis=1)

    benign_scores = scores[df["label"].values == 0]
    attack_scores = scores[df["label"].values == 1]

    clip = np.percentile(scores, 99)
    bins = np.linspace(0, clip, 80)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(benign_scores.clip(0, clip), bins=bins, alpha=0.6,
            color="#2ecc71", label=f"Benign (n={len(benign_scores):,})", density=True)
    ax.hist(attack_scores.clip(0, clip), bins=bins, alpha=0.6,
            color="#e74c3c", label=f"Attack (n={len(attack_scores):,})", density=True)
    ax.set_title("Anomaly Gap — Distance from Benign Mean (CICIoMT2024)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Squared Distance from Benign Centroid (standardized features)")
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(fig_dir, "cic_anomaly_gap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    overlap = np.mean(attack_scores < np.percentile(benign_scores, 95))
    print(f"\n  Anomaly gap: {overlap*100:.1f}% of attack samples fall below the 95th-percentile benign threshold")


# FR-107: Feature importance via mutual information
def plot_feature_importance(df, feat_cols, fig_dir):
    X = df[feat_cols].values
    y = df["label"].values
    mi = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi, index=feat_cols).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e74c3c" if v > mi_series.median() else "#3498db" for v in mi_series.values]
    ax.barh(mi_series.index, mi_series.values, color=colors, edgecolor="black", linewidth=0.3)
    ax.axvline(mi_series.median(), color="black", linestyle="--", linewidth=1, label="Median")
    ax.set_title("Feature Importance — Mutual Information (CICIoMT2024)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Mutual Information Score")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(fig_dir, "cic_feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    print("\n  Top 10 features by mutual information:")
    for feat, score in mi_series.sort_values(ascending=False).head(10).items():
        print(f"    {feat:<30} MI={score:.4f}")

    return mi_series


# FR-804: Summary metrics table
def save_summary_table(df, feat_cols, mi_series, fig_dir):
    benign = df[df["label"] == 0][feat_cols]
    attack = df[df["label"] == 1][feat_cols]
    summary = pd.DataFrame({
        "feature": feat_cols,
        "benign_mean": benign.mean().values,
        "benign_std": benign.std().values,
        "attack_mean": attack.mean().values,
        "attack_std": attack.std().values,
        "mean_diff_abs": (attack.mean() - benign.mean()).abs().values,
        "mutual_information": mi_series.reindex(feat_cols).values,
    }).sort_values("mutual_information", ascending=False).reset_index(drop=True)

    out = os.path.join(fig_dir, "cic_feature_summary.csv")
    summary.to_csv(out, index=False)
    print(f"  Saved: {out}")


def main():
    cfg = Config()
    ensure_dir(FIG_DIR)

    print("=" * 50)
    print(" CICIoMT2024 Exploratory Analysis")
    print("=" * 50)

    df = load_cic_data(cfg)
    feat_cols = get_feature_cols(df)
    print(f"\nFeature columns: {len(feat_cols)}")

    print("\nGenerating figures...")
    plot_label_distribution(df, FIG_DIR)
    plot_protocol_composition(df, FIG_DIR)
    print_feature_stats(df, feat_cols)
    plot_correlation_heatmap(df, feat_cols, FIG_DIR)
    plot_feature_distributions(df, feat_cols, FIG_DIR)
    plot_anomaly_gap(df, feat_cols, FIG_DIR)
    mi_series = plot_feature_importance(df, feat_cols, FIG_DIR)
    save_summary_table(df, feat_cols, mi_series, FIG_DIR)

    print("\nExploratory analysis complete. All outputs in results/figures/")


if __name__ == "__main__":
    main()
