# IoMT Ransomware Detection

> Early detection of ransomware attacks on Internet of Medical Things (IoMT) devices using a two-stage deep learning pipeline.

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/ethanmalavia/IoMT_ransom/badge)](https://securityscorecards.dev/viewer/?uri=github.com/ethanmalavia/IoMT_ransom)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12508/badge)](https://www.bestpractices.dev/projects/12508)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

---

## Overview

This project presents a **two-stage deep learning pipeline** for detecting ransomware on IoMT (Internet of Medical Things) devices in real time:

1. **Autoencoder (AE)** — Trained exclusively on benign traffic. Compresses raw sensor/network features into a latent representation and produces a per-sample reconstruction error (anomaly score).
2. **Mamba Classifier** — A selective state-space model that ingests a sliding window of 20 consecutive latent vectors + reconstruction errors to classify sequences as benign or ransomware.

The pipeline is benchmarked against an LSTM baseline and evaluated through ablation studies, early-detection analysis, and McNemar significance tests across three IoMT datasets.

---

## Architecture

```
Raw IoMT Traffic
      │
      ▼
┌─────────────┐
│ Autoencoder │  ← trained on benign only
│  Encoder    │  → latent vector (dim=32)
│  Decoder    │  → reconstruction error
└─────────────┘
      │  latent + recon_error
      ▼
┌──────────────────────────┐
│  Sliding Window (len=20) │
│  Mamba Classifier        │  ← sequence-level binary classifier
└──────────────────────────┘
      │
      ▼
  Benign / Ransomware
```

---

## Datasets

| Dataset | Type | Notes |
|---|---|---|
| Simulated ICU | Synthetic | 80 devices, 500 timesteps, attack onset t=200. Included at `data/raw/sim_raw/`. |
| TON-IoT | Real network traffic | Download from [UNSW](https://research.unsw.edu.au/projects/toniot-datasets). Place at `data/raw/ton_raw/`. |
| CICIoMT2024 | Real WiFi/MQTT + Bluetooth | Download from [CIC](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html). Place at `data/raw/cic_raw/`. |

### Regenerate simulated data

```bash
python -m src.simulation.simulate_icu
```

---

## Installation

```bash
git clone https://github.com/mbalalaj10/IoMT_ransom.git
cd IoMT_ransom
pip install -r requirements.txt
```

**Requirements:** `torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `joblib`

---

## Project Structure

```
IoMT_ransom/
├── data/
│   ├── raw/
│   │   ├── sim_raw/          # Simulated ICU data (included)
│   │   ├── ton_raw/          # TON-IoT data (download separately)
│   │   └── cic_raw/          # CICIoMT2024 data (download separately)
│   ├── processed/            # Scalers saved after preprocessing
│   └── splits/               # Train/test numpy arrays
├── models/                   # Saved model weights (.pt files)
├── results/
│   ├── figures/              # Generated plots
│   └── losses/               # Training loss histories
└── src/
    ├── config.py             # All hyperparameters and paths
    ├── utils.py
    ├── simulation/
    │   └── simulate_icu.py
    ├── preprocess/
    │   ├── preprocess_sim.py
    │   └── preprocess_ton.py
    ├── datasets/
    │   └── sequence_dataset.py
    ├── models/
    │   ├── autoencoder.py
    │   ├── mamba_classifier.py
    │   └── lstm_classifier.py
    ├── train/
    │   ├── train_autoencoder.py
    │   ├── train_mamba.py
    │   └── train_lstm.py
    ├── explore/
    │   └── explore_cic.py
    └── evaluate/
        ├── evaluate_ton.py
        ├── evaluate_sim.py
        ├── ablation.py
        ├── early_detection.py
        ├── significance.py
        ├── visualize.py
        ├── plot_loss_curves.py
        └── sanity_check.py
```

---

## Running the Pipeline

All commands are run from the project root.

### Step 1 — Preprocess

```bash
python -m src.preprocess.preprocess_sim
python -m src.preprocess.preprocess_ton
```

### Step 2 — Train Autoencoder

```bash
python -m src.train.train_autoencoder sim
python -m src.train.train_autoencoder ton
```

### Step 3 — Train Classifiers

```bash
python -m src.train.train_mamba sim
python -m src.train.train_mamba ton

python -m src.train.train_lstm sim
python -m src.train.train_lstm ton
```

### Step 4 — Evaluate

```bash
# Standard metrics (accuracy, precision, recall, F1, AUC-ROC)
python -m src.evaluate.evaluate_sim
python -m src.evaluate.evaluate_ton

# Ablation study
python -m src.evaluate.ablation

# Early detection analysis
python -m src.evaluate.early_detection

# Statistical significance (McNemar's test)
python -m src.evaluate.significance

# Generate all figures
python -m src.evaluate.visualize
python -m src.evaluate.plot_loss_curves
```

---

## Configuration

All hyperparameters and paths are in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `seq_len` | 20 | Sliding window length |
| `latent_dim` | 32 | Autoencoder bottleneck size |
| `d_model` | 64 | Mamba/LSTM hidden dimension |
| `num_layers` | 2 | Number of Mamba/LSTM layers |
| `clf_epochs` | 15 | Classifier training epochs |
| `ae_epochs` | 15 | Autoencoder training epochs |
| `threshold` | 0.5 | Classification decision threshold |

---

## Security

To report a vulnerability, please see [SECURITY.md](SECURITY.md).

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
