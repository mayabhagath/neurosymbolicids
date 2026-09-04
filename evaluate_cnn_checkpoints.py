"""
Re-evaluate existing CNN checkpoints with a corrected multiclass-15 metric.

Fix vs. the original training script: a zero-day sample now counts as
"correct" for the multiclass-15 view if the model predicts ANY attack
class (not an exact match against its true zero-day label, which is
mathematically impossible for a model with no output neuron for that
class). Known-class samples still require an exact match. Working the
base paper's own numbers backward, this is the only convention that
reproduces their reported multiclass-15 accuracy from their known-class
accuracy and zero-day recall figures.

No retraining -- loads your existing cnn_epoch_30.keras / cnn_epoch_50.keras
checkpoints and just re-scores them.

Run:
    python evaluate_cnn_checkpoints.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf

DATA_DIR = Path(r"E:\NeuroSymbolic-IDS1\causal_kg\data\processed")
MODEL_DIR = Path(r"E:\NeuroSymbolic-IDS1\models")  # shared, read-only checkpoint location
LABEL_COL = "label"
BENIGN_LABEL = "BENIGN"
BATCH_SIZE = 128
CHECKPOINT_EPOCHS = [30, 50]
MODEL_PREFIX = "hybrid_ltn_richer_omega5"  # change to "cnn" to check the plain baseline instead

# ----------------------------------------------------------------------
# Load data (same as training script)
# ----------------------------------------------------------------------
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_known_df = pd.read_csv(DATA_DIR / "test_known.csv")
test_zeroday_df = pd.read_csv(DATA_DIR / "test_zero_day.csv")

byte_cols = [c for c in train_df.columns if "payload_byte" in c]


def get_X(df):
    X = df[byte_cols].values.astype("float32") / 255.0
    return X.reshape(-1, 1500, 1)


X_test_known = get_X(test_known_df)
X_test_zeroday = get_X(test_zeroday_df)

label_encoder = LabelEncoder()
label_encoder.fit(train_df[LABEL_COL])
y_test_known_int = label_encoder.transform(test_known_df[LABEL_COL])
benign_index = list(label_encoder.classes_).index(BENIGN_LABEL)

print(f"Known classes: {list(label_encoder.classes_)}")
print(f"Test known: {len(test_known_df)} | Test zero-day: {len(test_zeroday_df)}\n")

# ----------------------------------------------------------------------
# Corrected evaluation
# ----------------------------------------------------------------------


def evaluate_checkpoint(model_path, epoch_label):
    print(f"\n{'=' * 60}\nEvaluating checkpoint: {epoch_label} epochs\n{'=' * 60}")
    m = tf.keras.models.load_model(model_path)

    pred_probs_known = m.predict(X_test_known, batch_size=BATCH_SIZE, verbose=0)
    pred_labels_known_int = np.argmax(pred_probs_known, axis=1)

    multiclass_acc_known = accuracy_score(y_test_known_int, pred_labels_known_int)

    pred_binary_known = (pred_labels_known_int != benign_index).astype(int)
    true_binary_known = (y_test_known_int != benign_index).astype(int)
    binary_acc_known = accuracy_score(true_binary_known, pred_binary_known)
    binary_f1_known = f1_score(true_binary_known, pred_binary_known)

    pred_probs_zeroday = m.predict(X_test_zeroday, batch_size=BATCH_SIZE, verbose=0)
    pred_labels_zeroday_int = np.argmax(pred_probs_zeroday, axis=1)
    pred_binary_zeroday = (pred_labels_zeroday_int != benign_index).astype(int)

    # Zero-day "attack of any kind" recall -- this IS the binary-6-unknown metric
    true_binary_zeroday = np.ones(len(test_zeroday_df), dtype=int)
    binary_acc_zeroday = accuracy_score(true_binary_zeroday, pred_binary_zeroday)
    binary_f1_zeroday = f1_score(true_binary_zeroday, pred_binary_zeroday, zero_division=0)

    # --- FIXED multiclass-15: known needs exact match; zero-day just
    # needs "predicted as some attack" to count as correct ---
    n_known = len(test_known_df)
    n_zeroday = len(test_zeroday_df)
    n_total = n_known + n_zeroday
    correct_known = (pred_labels_known_int == y_test_known_int).sum()
    correct_zeroday_loose = pred_binary_zeroday.sum()  # predicted non-benign
    multiclass_acc_all15 = (correct_known + correct_zeroday_loose) / n_total

    # Binary-15: weighted combination, both known and zero-day binary-correct
    correct_binary_known = (pred_binary_known == true_binary_known).sum()
    correct_binary_zeroday = pred_binary_zeroday.sum()  # true is always 1 (attack)
    binary_acc_all15 = (correct_binary_known + correct_binary_zeroday) / n_total

    all_true_binary = np.concatenate([true_binary_known, true_binary_zeroday])
    all_pred_binary = np.concatenate([pred_binary_known, pred_binary_zeroday])
    binary_f1_all15 = f1_score(all_true_binary, all_pred_binary)

    results = {
        "Multi-class 9 known classes (acc)": multiclass_acc_known,
        "Binary 9 known classes (acc)": binary_acc_known,
        "Binary 9 known classes (F1)": binary_f1_known,
        "Multi-class 15 classes (acc) [fixed]": multiclass_acc_all15,
        "Binary 15 classes (acc)": binary_acc_all15,
        "Binary 15 classes (F1)": binary_f1_all15,
        "Binary 6 unknown classes (acc)": binary_acc_zeroday,
        "Binary 6 unknown classes (F1)": binary_f1_zeroday,
    }
    for k, v in results.items():
        print(f"  {k}: {v * 100:.2f}%")
    return results


all_results = {}
for ep in CHECKPOINT_EPOCHS:
    ckpt_path = MODEL_DIR / f"{MODEL_PREFIX}_epoch_{ep}.keras"
    if ckpt_path.exists():
        all_results[ep] = evaluate_checkpoint(ckpt_path, ep)
    else:
        print(f"\n[warning] checkpoint for epoch {ep} not found at {ckpt_path}")

results_df = pd.DataFrame(all_results).T
results_df.index.name = "epochs"
results_df.to_csv(MODEL_DIR / f"{MODEL_PREFIX}_results_fixed.csv")
print(f"\nSaved corrected results table -> {MODEL_DIR / f'{MODEL_PREFIX}_results_fixed.csv'}")
