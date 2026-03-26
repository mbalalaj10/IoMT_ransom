# IoMT Ransomware Detection

Early detection of ransomware attacks on Internet of Medical Things (IoMT) devices using a two-stage deep learning pipeline: an Autoencoder for feature compression and anomaly scoring, followed by a Mamba selective state space classifier for sequence-level attack detection.

## Architecture


Raw network/device features
        
  Autoencoder
    Encoder : latent vector z  
    Decoder : reconstruction   
    Error signal (MSE)         
        z + reconstruction error
  
  Sequence Classifier (Mamba / LSTM)

    Sliding window (seq_len=20)
    Selective SSM layers       
    Binary: normal / attack    


## Datasets

### Simulated ICU (included)
Located at `data/raw/sim_raw/icu_simulation.csv`. If it is missing, regenerate it:

python -m src.simulation.simulate_icu

80 devices (ventilators, infusion pumps, patient monitors, IoMT gateways), 500 timesteps each. Attack onset at t=200.

### TON-IoT (must be downloaded separately)
Download the **Network dataset (Bro/Zeek logs)** from the [TON-IoT dataset page](https://research.unsw.edu.au/projects/toniot-datasets) and place it at:

data/raw/ton_raw/Network_dataset_Bro/
    normal_Bro/
    normal_attack_Bro/


---

## Installation

**Requirements:

pip install torch numpy pandas scikit-learn scipy matplotlib joblib

## Project Structure


IoMT_ransom/
├── data/
   ├── raw/
      ├── sim_raw/          # Simulated ICU data (included)
      └── ton_raw/          # TON-IoT data (download separately)
   ├── processed/           
   └── splits/               
├── models/                   
├── results/
   ├── figures/            
   └── losses/              
└── src/
    ├── config.py            
    ├── utils.py
    ├── simulation/
       └── simulate_icu.py   
    ├── preprocess/
       ├── preprocess_sim.py
       └── preprocess_ton.py
    ├── datasets/
       └── sequence_dataset.py
    ├── models/
       ├── autoencoder.py
       ├── mamba_classifier.py
       └── lstm_classifier.py
    ├── train/
       ├── train_autoencoder.py
       ├── train_mamba.py
       └── train_lstm.py
    └── evaluate/
        ├── evaluate_ton.py
        ├── evaluate_sim.py
        ├── ablation.py
        ├── early_detection.py
        ├── significance.py
        ├── visualize.py
        ├── plot_loss_curves.py
        └── sanity_check.py


### Step 1 — Preprocess

**Simulated ICU:**

python -m src.preprocess.preprocess_sim


**TON-IoT:**

python -m src.preprocess.preprocess_ton


### Step 2 — Train Autoencoder


python -m src.train.train_autoencoder sim
python -m src.train.train_autoencoder ton


### Step 3 — Train Classifiers


python -m src.train.train_mamba sim
python -m src.train.train_mamba ton

python -m src.train.train_lstm sim
python -m src.train.train_lstm ton


### Step 4 — Evaluate

**Standard metrics (accuracy, precision, recall, F1, AUC-ROC):**

python -m src.evaluate.evaluate_sim
python -m src.evaluate.evaluate_ton

**Ablation study (AE Only vs AE+LR vs AE+LSTM vs AE+Mamba):**

python -m src.evaluate.ablation


**Early detection analysis (detection lag after attack onset):**

python -m src.evaluate.early_detection


**Statistical significance testing:**

python -m src.evaluate.significance


**Generate all figures:**

python -m src.evaluate.visualize
python -m src.evaluate.plot_loss_curves


## Configuration

All hyperparameters and file paths are in `src/config.py`

| Parameter | Default | Description |
|---|---|---|
| `seq_len` | 20 | Sliding window length |
| `latent_dim` | 32 | Autoencoder bottleneck size |
| `d_model` | 64 | Mamba/LSTM hidden dimension |
| `num_layers` | 2 | Number of Mamba/LSTM layers |
| `clf_epochs` | 15 | Classifier training epochs |
| `ae_epochs` | 15 | Autoencoder training epochs |
| `threshold` | 0.5 | Classification decision threshold |
