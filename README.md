# IoMT Ransomware Detection

> Early detection of ransomware attacks on Internet of Medical Things (IoMT) devices using a two-stage deep learning pipeline.

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/ethanmalavia/IoMT_ransom/badge)](https://securityscorecards.dev/viewer/?uri=github.com/ethanmalavia/IoMT_ransom)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12508/badge?v=2)](https://www.bestpractices.dev/projects/12508)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

---

## What Is This Project?

**IoMT (Internet of Medical Things)** refers to the network of connected medical devices used in healthcare — things like patient monitors, infusion pumps, ventilators, and ICU sensors. These devices communicate over networks and generate continuous streams of data.

**Ransomware** is a type of malware that encrypts a victim's files and demands payment to restore access. When ransomware hits hospital infrastructure, it can render life-critical equipment unavailable — a direct patient safety risk. Detecting ransomware *early*, before encryption is complete, is essential.

This project builds an **automated, real-time ransomware detection system** for IoMT networks using deep learning. Rather than relying on signature-based antivirus (which can't detect new variants), the system learns what *normal* IoMT network behaviour looks like and flags deviations — including novel ransomware strains.

---

## How It Works

The detection pipeline has two stages:

### Stage 1 — Autoencoder (Anomaly Scoring)

An **autoencoder** is a neural network trained to compress data into a compact representation and then reconstruct it. We train it **only on normal (benign) traffic**, so it learns what healthy IoMT network behaviour looks like.

When ransomware traffic is passed through the same network, the autoencoder struggles to reconstruct it accurately — producing a high **reconstruction error**. This error acts as an anomaly score.

The autoencoder also produces a **latent vector** (a 32-dimensional compressed summary of the traffic sample). Both the latent vector and the reconstruction error are passed to Stage 2.

### Stage 2 — Sequence Classifier (Attack Confirmation)

A single anomalous packet could be noise. Ransomware, however, produces *sustained* anomalous behaviour. The second stage looks at a **sliding window of 20 consecutive timesteps** and classifies the sequence as benign or ransomware.

Two classifiers are trained and compared:
- **Mamba** — A selective state-space model (SSM) that efficiently captures long-range temporal patterns. Mamba is the primary model.
- **LSTM** — A long short-term memory network, used as a classical baseline.

```
Raw IoMT Network Traffic
         │
         ▼
┌─────────────────────┐
│     Autoencoder     │  ← trained on benign traffic only
│  Input → Latent     │  → 32-dim compressed representation
│  Latent → Output    │  → reconstruction error (anomaly score)
└─────────────────────┘
         │  [latent vector (32) + recon error (1)] = 33-dim
         ▼
┌──────────────────────────────┐
│  Sliding Window (20 steps)   │
│  Mamba Sequence Classifier   │  ← classifies sequence as benign/attack
└──────────────────────────────┘
         │
         ▼
    Benign / Ransomware
```

### Why This Design?

- **Unsupervised anomaly detection first:** The autoencoder doesn't need labelled attack data to detect that something is wrong — it just notices the traffic looks unfamiliar.
- **Sequence modelling second:** Temporal context eliminates false positives from transient network spikes.
- **Cross-dataset generalisation:** Because the latent space is architecture-agnostic, a classifier trained on one IoMT environment can be applied to another via latent-space transfer.

---

## Datasets

The system is trained and evaluated on three independent datasets representing different real-world IoMT environments:

| Dataset | Type | Scale | Source |
|---|---|---|---|
| **Simulated ICU** | Synthetic | 80 devices, 500 timesteps/device, attack onset at t=200 | Included (`data/raw/sim_raw/`) |
| **TON-IoT** | Real network captures (Bro/Zeek conn logs) | ~77K samples across normal and attack scenarios | [UNSW TON-IoT](https://research.unsw.edu.au/projects/toniot-datasets) |
| **CICIoMT2024** | Real WiFi/MQTT traffic | ~50K samples, 17 attack types | [CIC IoMT 2024](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html) |

**Simulated ICU** is a controlled synthetic environment where the exact attack onset is known, allowing precise early-detection measurement.

**TON-IoT** contains real network traffic captured from an IoT/IIoT testbed at UNSW Sydney, including telemetry data with genuine ransomware, DoS, and other attack traffic.

**CICIoMT2024** is a benchmark dataset from the Canadian Institute for Cybersecurity covering WiFi and MQTT protocol traffic from simulated IoMT environments, with 17 distinct attack categories.

> **Note on Bluetooth:** CICIoMT2024 includes a Bluetooth capture subset, but those files are distributed as raw `.pcap` files without pre-extracted CSV features. Only the WiFi/MQTT subset (which includes CSV feature files) is used in this project.

---

## Results

All models use threshold optimisation (max-F1 sweep) rather than a fixed 0.5 decision boundary.

### Per-Dataset Performance

| Dataset | Model | F1 | AUC-ROC | FPR | FNR |
|---|---|---|---|---|---|
| Simulated ICU | **Mamba** | 0.9991 | 0.9999 | 0.0002 | 0.0017 |
| Simulated ICU | LSTM | 0.9981 | 0.9999 | 0.0002 | 0.0036 |
| TON-IoT | **Mamba** | 0.9983 | 0.9940 | 0.0063 | 0.0000 |
| TON-IoT | LSTM | — | — | — | — |
| CICIoMT2024 | **Mamba** | 0.9840 | 0.9980 | — | — |
| CICIoMT2024 | LSTM | 0.9820 | 0.9830 | — | — |

**FPR** = False Positive Rate (benign traffic incorrectly flagged as attack).  
**FNR** = False Negative Rate (attack traffic missed). In a safety-critical setting, minimising FNR is the priority.

### Cross-Dataset Generalisation

One classifier trained on one dataset is evaluated on the test splits of all other datasets (via latent-space transfer). Diagonal entries (★) are within-dataset results.

| Train → Test | Simulated ICU | TON-IoT | CICIoMT2024 |
|---|---|---|---|
| **Simulated ICU ★** | 0.9991 | ~0.50 | ~0.50 |
| **TON-IoT ★** | ~0.50 | 0.9983 | ~0.50 |
| **CICIoMT2024 ★** | ~0.50 | **0.997** | 0.9840 |

The CICIoMT2024 → TON-IoT cross-dataset result (AUC ≈ 0.997) is a strong indicator that the CIC classifier has learned generalisable attack patterns, not just dataset-specific artefacts. The SIM classifier's poor cross-dataset transfer is expected — synthetic traffic has a very different statistical profile from real network captures.

---

## Installation

```bash
git clone https://github.com/mbalalaj10/IoMT_ransom.git
cd IoMT_ransom
pip install -r requirements.txt
```

**Core dependencies:** `torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `joblib`

---

## Project Structure

```
IoMT_ransom/
├── data/
│   ├── raw/
│   │   ├── sim_raw/          # Simulated ICU data (included)
│   │   ├── ton_raw/          # TON-IoT data (download separately)
│   │   └── cic_raw/
│   │       └── wifi_mqtt/    # CICIoMT2024 WiFi/MQTT CSVs
│   ├── processed/            # Fitted scalers saved after preprocessing
│   └── splits/               # Train/val/test numpy arrays per dataset
├── models/                   # Saved model weights (.pt files)
├── results/
│   ├── figures/              # Generated plots (ROC curves, confusion matrices)
│   └── losses/               # Training loss histories per model
├── progress_presentation.ipynb   # Full results notebook
└── src/
    ├── config.py             # All hyperparameters and file paths
    ├── utils.py              # Seed, device, directory helpers
    ├── simulation/
    │   └── simulate_icu.py   # Generates the synthetic ICU dataset
    ├── preprocess/
    │   ├── preprocess_sim.py
    │   ├── preprocess_ton.py
    │   └── preprocess_cic.py
    ├── datasets/
    │   └── sequence_dataset.py   # Sliding-window dataset builder
    ├── models/
    │   ├── autoencoder.py
    │   ├── mamba_classifier.py
    │   └── lstm_classifier.py
    ├── train/
    │   ├── train_autoencoder.py
    │   ├── train_mamba.py
    │   └── train_lstm.py
    └── evaluate/
        ├── metrics.py            # compute_metrics, find_optimal_threshold
        ├── evaluate_sim.py
        ├── evaluate_ton.py
        ├── evaluate_cic.py
        ├── cross_dataset.py      # 3×3 cross-dataset transfer evaluation
        ├── ablation.py
        ├── early_detection.py
        ├── significance.py       # McNemar's test
        └── visualize.py
```

---

## Running the Full Pipeline

All commands are run from the project root. Each dataset follows the same three steps: preprocess → train → evaluate.

### Data Setup

**Simulated ICU** is included. Run to regenerate if needed:
```bash
python -m src.simulation.simulate_icu
```

**TON-IoT:** Download the Network Dataset (Bro logs) from [UNSW](https://research.unsw.edu.au/projects/toniot-datasets). Extract to `data/raw/ton_raw/`.

**CICIoMT2024:** Download the WiFi/MQTT CSV files from [CIC](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html). Place in `data/raw/cic_raw/wifi_mqtt/`.

---

### Step 1 — Preprocess

Cleans raw data, encodes categorical features, fits a StandardScaler, and saves stratified 70/15/15 train/val/test splits grouped by device/IP pair to prevent data leakage.

```bash
python -m src.preprocess.preprocess_sim
python -m src.preprocess.preprocess_ton
python -m src.preprocess.preprocess_cic
```

### Step 2 — Train Autoencoder

Trains the anomaly-scoring autoencoder on **benign-only** traffic. Uses early stopping on validation reconstruction loss.

```bash
python -m src.train.train_autoencoder sim
python -m src.train.train_autoencoder ton
python -m src.train.train_autoencoder cic
```

### Step 3 — Train Sequence Classifiers

Encodes all splits through the trained autoencoder, then trains the Mamba and LSTM classifiers on 20-step sliding window sequences. Both use early stopping with patience=5.

```bash
python -m src.train.train_mamba sim
python -m src.train.train_mamba ton
python -m src.train.train_mamba cic

python -m src.train.train_lstm sim
python -m src.train.train_lstm ton
python -m src.train.train_lstm cic
```

### Step 4 — Evaluate

```bash
# Per-dataset results (accuracy, precision, recall, F1, AUC-ROC, FPR, FNR)
python -m src.evaluate.evaluate_sim
python -m src.evaluate.evaluate_ton
python -m src.evaluate.evaluate_cic

# Cross-dataset generalisation (3×3 transfer matrix)
python -m src.evaluate.cross_dataset

# Ablation study (AE Only vs AE+LR vs AE+LSTM vs AE+Mamba)
python -m src.evaluate.ablation

# Early detection analysis (how many timesteps before full encryption?)
python -m src.evaluate.early_detection

# Statistical significance (McNemar's test: Mamba vs LSTM)
python -m src.evaluate.significance

# Generate plots
python -m src.evaluate.visualize
python -m src.evaluate.plot_loss_curves
```

### Step 5 — View Results Notebook

Open `progress_presentation.ipynb` in Jupyter for the full results walkthrough including tables, ROC curves, confusion matrices, ablation charts, and cross-dataset analysis.

```bash
jupyter notebook progress_presentation.ipynb
```

---

## Key Hyperparameters

All settings are centralised in `src/config.py`.

| Parameter | Default | Description |
|---|---|---|
| `seq_len` | 20 | Sliding window length (timesteps per sequence) |
| `latent_dim` | 32 | Autoencoder bottleneck dimension |
| `d_model` | 64 | Mamba/LSTM hidden dimension (32 for CIC) |
| `num_layers` | 2 | Number of Mamba/LSTM layers (1 for CIC) |
| `ae_epochs` | 15 | Max autoencoder training epochs |
| `clf_epochs` | 15 | Max classifier training epochs (8 for CIC) |
| `early_stopping_patience` | 5 | Epochs without val improvement before stopping |
| `test_size` | 0.15 | Held-out test fraction |
| `validation_size` | 0.1765 | Validation fraction of trainval (≈15% of total) |

> CIC uses lighter hyperparameters (d_model=32, 1 layer, 8 epochs) due to its smaller scale and CPU-only feasibility requirements.

---

## Design Decisions and Limitations

**Group-aware splits:** `GroupShuffleSplit` ensures that all traffic from a given device or IP pair stays within a single split. This prevents the model from memorising device-specific patterns and inflating test performance.

**Threshold optimisation:** Rather than using a fixed 0.5 threshold, each model's decision boundary is swept from 0.01–0.99 and the value maximising F1 is selected. This is important for imbalanced test sets.

**Bluetooth exclusion:** The CICIoMT2024 Bluetooth subset is distributed as raw `.pcap` files with no pre-extracted feature CSVs. Converting pcap to feature vectors requires packet inspection tooling (Zeek, CICFlowMeter) beyond the scope of this project. Only the WiFi/MQTT CSV subset is used.

**Cross-dataset transfer:** Because each dataset has a different number of input features (SIM=19, TON=309, CIC=45), direct feature-level transfer is not possible. Transfer is performed in the shared 33-dimensional latent space (32 latent + 1 reconstruction error) produced by each dataset's own autoencoder.

---

## Security

To report a vulnerability, please see [SECURITY.md](SECURITY.md).

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
