"""
Autoencoder-based anomaly detection for zero-day traffic -- trains a dense
autoencoder on BENIGN TRAFFIC ONLY, then flags packets with abnormally high
reconstruction error as potential attacks (known or novel).

Rationale: the confidence/entropy experiment failed because standard
softmax classifiers tend to be OVERCONFIDENT on out-of-distribution inputs,
not uncertain -- every input lands in some decision region and gets a
confident prediction, since nothing in training ever taught the network to
express "I don't know". An autoencoder sidesteps this: it isn't a
classifier at all, it just learns to compress and reconstruct what benign
traffic looks like. Anything structurally different -- any attack, known
or unseen -- should reconstruct poorly, because the network never learned
to encode/decode that kind of input in the first place.

Architecture: 1500 -> 256 -> 64 -> 256 -> 1500, dense, sigmoid output
(inputs are normalized to [0,1] byte values), MSE reconstruction loss.

Threshold fitting: on the reconstruction error of the BENIGN VALIDATION
set (never seen during AE training) -- not the training set, to avoid an
artificially low threshold from data the network was directly optimized
to reconstruct.

Evaluated: standalone, and fused (OR) with the neural branch and the
causal symbolic KG layer.

Run:
    py autoencoder_anomaly_detector.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf

DATA_DIR = Path(r"E:\NeuroSymbolic-IDS1\causal_kg\data\processed")
KG_DIR = Path(r"E:\NeuroSymbolic-IDS1\causal_kg\data\kg")
MODEL_DIR = Path(r"E:\NeuroSymbolic-IDS1\models")  # shared, read-only (existing CNN/LTN checkpoint)
AE_MODEL_DIR = Path(r"E:\NeuroSymbolic-IDS1\causal_kg\models")  # NEW, for this experiment's own artifacts
AE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(r"E:\NeuroSymbolic-IDS1\causal_kg\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label"
BENIGN_LABEL = "BENIGN"
BATCH_SIZE = 128
AE_EPOCHS = 30
ERROR_PERCENTILE = 95.0  # top 5% highest-reconstruction-error benign val samples set the bar
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NEURAL_CHECKPOINT = MODEL_DIR / "hybrid_ltn_richer_omega5_epoch_30.keras"

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
print("Loading splits...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")
test_known_df = pd.read_csv(DATA_DIR / "test_known.csv")
test_zeroday_df = pd.read_csv(DATA_DIR / "test_zero_day.csv")

byte_cols = [c for c in train_df.columns if "payload_byte" in c]


def get_X(df):
    return df[byte_cols].values.astype("float32") / 255.0  # (N, 1500), no reshape needed for dense AE


X_train_full = get_X(train_df)
X_val_full = get_X(val_df)
X_test_known = get_X(test_known_df)
X_test_zeroday = get_X(test_zeroday_df)

# Autoencoder trains on BENIGN TRAINING TRAFFIC ONLY
benign_train_mask = (train_df[LABEL_COL] == BENIGN_LABEL).values
X_ae_train = X_train_full[benign_train_mask]
print(f"AE training set (benign only): {X_ae_train.shape}")

# Threshold fit on BENIGN VALIDATION TRAFFIC ONLY -- held out from AE training
benign_val_mask = (val_df[LABEL_COL] == BENIGN_LABEL).values
X_ae_val_benign = X_val_full[benign_val_mask]
print(f"AE threshold-fitting set (benign val only): {X_ae_val_benign.shape}")

# ----------------------------------------------------------------------
# Build and train the autoencoder
# ----------------------------------------------------------------------


def build_autoencoder(input_dim=1500):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dense(64, activation="relu", name="bottleneck")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(input_dim, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)


autoencoder = build_autoencoder()
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
autoencoder.summary()

print(f"\nTraining autoencoder for {AE_EPOCHS} epochs on benign traffic only...")
history = autoencoder.fit(
    X_ae_train, X_ae_train,
    validation_data=(X_ae_val_benign, X_ae_val_benign),
    epochs=AE_EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
)

ae_path = AE_MODEL_DIR / "benign_autoencoder.keras"
autoencoder.save(ae_path)
print(f"\nSaved autoencoder -> {ae_path}")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv(AE_MODEL_DIR / "autoencoder_training_history.csv", index_label="epoch")

# ----------------------------------------------------------------------
# Compute reconstruction error and fit threshold on benign val
# ----------------------------------------------------------------------


def reconstruction_error(X):
    X_recon = autoencoder.predict(X, batch_size=BATCH_SIZE, verbose=0)
    return np.mean((X - X_recon) ** 2, axis=1)


print("\nComputing reconstruction error on benign validation set (threshold fitting)...")
val_benign_error = reconstruction_error(X_ae_val_benign)
error_threshold = np.percentile(val_benign_error, ERROR_PERCENTILE)
print(f"Reconstruction error threshold (P{ERROR_PERCENTILE} of benign val): {error_threshold:.6f}")
print(f"Benign val error stats: mean={val_benign_error.mean():.6f}, "
      f"std={val_benign_error.std():.6f}, max={val_benign_error.max():.6f}")

print("\nComputing reconstruction error on test_known...")
known_error = reconstruction_error(X_test_known)
print("Computing reconstruction error on test_zero_day...")
zeroday_error = reconstruction_error(X_test_zeroday)

print(f"\ntest_known error stats: mean={known_error.mean():.6f}, max={known_error.max():.6f}")
print(f"test_zero_day error stats: mean={zeroday_error.mean():.6f}, max={zeroday_error.max():.6f}")

anomaly_flag_known = (known_error > error_threshold).astype(int)
anomaly_flag_zeroday = (zeroday_error > error_threshold).astype(int)

# ----------------------------------------------------------------------
# Load neural + symbolic predictions for fusion comparison
# ----------------------------------------------------------------------
label_encoder = LabelEncoder()
label_encoder.fit(train_df[LABEL_COL])
y_test_known_int = label_encoder.transform(test_known_df[LABEL_COL])
benign_index = list(label_encoder.classes_).index(BENIGN_LABEL)

print(f"\nLoading neural checkpoint: {NEURAL_CHECKPOINT}")
cnn_model = tf.keras.models.load_model(NEURAL_CHECKPOINT)

X_test_known_cnn = X_test_known.reshape(-1, 1500, 1)
X_test_zeroday_cnn = X_test_zeroday.reshape(-1, 1500, 1)

known_pred_int = np.argmax(cnn_model.predict(X_test_known_cnn, batch_size=BATCH_SIZE, verbose=0), axis=1)
zeroday_pred_int = np.argmax(cnn_model.predict(X_test_zeroday_cnn, batch_size=BATCH_SIZE, verbose=0), axis=1)
neural_binary_known = (known_pred_int != benign_index).astype(int)
neural_binary_zeroday = (zeroday_pred_int != benign_index).astype(int)

test_known_kg = pd.read_csv(KG_DIR / "test_known_kg_causal.csv")
test_zeroday_kg = pd.read_csv(KG_DIR / "test_zero_day_kg_causal.csv")
symbolic_binary_known = test_known_kg["symbolic_flag_causal"].astype(int).values
symbolic_binary_zeroday = test_zeroday_kg["symbolic_flag_causal"].astype(int).values

true_binary_known = (y_test_known_int != benign_index).astype(int)
true_binary_zeroday = np.ones(len(test_zeroday_df), dtype=int)

# ----------------------------------------------------------------------
# Evaluate
# ----------------------------------------------------------------------


def report(name, pred_known, pred_zeroday):
    acc_known = accuracy_score(true_binary_known, pred_known)
    f1_known = f1_score(true_binary_known, pred_known)
    acc_zeroday = accuracy_score(true_binary_zeroday, pred_zeroday)
    f1_zeroday = f1_score(true_binary_zeroday, pred_zeroday, zero_division=0)

    n_known = len(true_binary_known)
    n_zeroday = len(true_binary_zeroday)
    n_total = n_known + n_zeroday
    correct_known = (pred_known == true_binary_known).sum()
    correct_zeroday = pred_zeroday.sum()
    acc_15 = (correct_known + correct_zeroday) / n_total

    print(f"\n--- {name} ---")
    print(f"  Binary 9 known (acc):     {acc_known*100:.2f}%")
    print(f"  Binary 9 known (F1):      {f1_known*100:.2f}%")
    print(f"  Binary 15 (acc):          {acc_15*100:.2f}%")
    print(f"  Binary 6 unknown (acc):   {acc_zeroday*100:.2f}%")
    print(f"  Binary 6 unknown (F1):    {f1_zeroday*100:.2f}%")
    return {
        "binary_9_known_acc": acc_known, "binary_9_known_f1": f1_known,
        "binary_15_acc": acc_15,
        "binary_6_unknown_acc": acc_zeroday, "binary_6_unknown_f1": f1_zeroday,
    }


fused_neural_ae_known = ((neural_binary_known == 1) | (anomaly_flag_known == 1)).astype(int)
fused_neural_ae_zeroday = ((neural_binary_zeroday == 1) | (anomaly_flag_zeroday == 1)).astype(int)

fused_all_known = ((neural_binary_known == 1) | (anomaly_flag_known == 1) |
                    (symbolic_binary_known == 1)).astype(int)
fused_all_zeroday = ((neural_binary_zeroday == 1) | (anomaly_flag_zeroday == 1) |
                      (symbolic_binary_zeroday == 1)).astype(int)

results = {}
results["neural_only"] = report("Neural only (baseline, for reference)",
                                  neural_binary_known, neural_binary_zeroday)
results["autoencoder_only"] = report("Autoencoder anomaly flag only",
                                       anomaly_flag_known, anomaly_flag_zeroday)
results["neural_plus_autoencoder"] = report("Neural + Autoencoder (fused)",
                                              fused_neural_ae_known, fused_neural_ae_zeroday)
results["neural_plus_autoencoder_plus_symbolic"] = report(
    "Neural + Autoencoder + Symbolic (full 3-way fusion)",
    fused_all_known, fused_all_zeroday)

results_df = pd.DataFrame(results).T
results_df.to_csv(RESULTS_DIR / "autoencoder_results.csv")
print(f"\nSaved comparison table -> {RESULTS_DIR / 'autoencoder_results.csv'}")
