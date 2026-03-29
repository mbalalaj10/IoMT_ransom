# IoMT Ransomware Detection

Early detection of ransomware attacks on Internet of Medical Things (IoMT) devices using a two-stage deep learning pipeline: an **Autoencoder** for feature compression and anomaly scoring, followed by a **Mamba** selective state space classifier for sequence-level attack detection.

---

## Datasets

### Simulated ICU (included)
Located at `data/raw/sim_raw/icu_simulation.csv`. If it is missing, regenerate it:
```bash
python -m src.simulation.simulate_icu
```
80 devices (ventilators, infusion pumps, patient monitors, IoMT gateways), 500 timesteps each. Attack onset at t=200.

### TON-IoT (must be downloaded separately)
Download the **Network dataset** from the [TON-IoT dataset page](https://research.unsw.edu.au/projects/toniot-datasets) and place it at:
```
data/raw/ton_raw/network_data/network_data/
```

### CICIoMT2024 (must be downloaded separately)
Download the **CICIoMT2024** dataset and place the CSV files at:
```
data/raw/cic_raw/wifi_mqtt/      # WiFi/MQTT traffic CSVs
data/raw/cic_raw/bluetooth/      # Bluetooth traffic CSVs
```
Benign files should have `benign` in the filename; all others are treated as attacks.

---

## Installation

**Requirements:**

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib joblib
```


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
    │   └── simulate_icu.py   # Generate the ICU dataset
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
    │   └── explore_cic.py    # EDA for CICIoMT2024 dataset
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

All commands are run from the project root directory.

### Step 1 — Preprocess

**Simulated ICU:**
```bash
python -m src.preprocess.preprocess_sim
```

**TON-IoT:**
```bash
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

**Standard metrics (accuracy, precision, recall, F1, AUC-ROC):**
```bash
python -m src.evaluate.evaluate_sim
python -m src.evaluate.evaluate_ton
```

**Ablation study (AE Only vs AE+LR vs AE+LSTM vs AE+Mamba):**
```bash
python -m src.evaluate.ablation
```

**Early detection analysis (detection lag after attack onset):**
```bash
python -m src.evaluate.early_detection
```

**Statistical significance testing (McNemar's test):**
```bash
python -m src.evaluate.significance
```

**Generate all figures:**
```bash
python -m src.evaluate.visualize
python -m src.evaluate.plot_loss_curves
```

---

## Exploratory Analysis (CICIoMT2024)

Run EDA on the CICIoMT2024 dataset to generate label distributions, protocol composition, correlation heatmaps, feature importance, and anomaly gap plots:

```bash
python -m src.explore.explore_cic
```

Outputs are saved to `results/figures/` with the prefix `cic_`.

---

## Configuration

All hyperparameters and file paths are in `src/config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `seq_len` | 20 | Sliding window length |
| `latent_dim` | 32 | Autoencoder bottleneck size |
| `d_model` | 64 | Mamba/LSTM hidden dimension |
| `num_layers` | 2 | Number of Mamba/LSTM layers |
| `clf_epochs` | 15 | Classifier training epochs |
| `ae_epochs` | 15 | Autoencoder training epochs |
| `threshold` | 0.5 | Classification decision threshold |
