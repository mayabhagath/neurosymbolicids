"""
1D CNN baseline — replicates the base paper's (Bizzarri et al., ICCCN 2024)
benchmark architecture (Fig. 2) and evaluation protocol (Table II).

Architecture:
    Input (batch, 1500, 1)
    -> Conv1D(32, k=3) -> ReLU -> MaxPool(4)
    -> Conv1D(63, k=3) -> ReLU -> MaxPool(8)
    -> Conv1D(128, k=3) -> ReLU -> MaxPool(16)
    -> Flatten
    -> Dense(N*5) -> ReLU
    -> Dense(N) -> Softmax          (N = 9: BENIGN + 8 known attacks)

Trains to 50 epochs, saving a checkpoint at epoch 30 as well, so both of
the paper's reported checkpoints (30 and 50 epochs) can be evaluated.

Evaluates 5 views per checkpoint, matching Table II:
    1. Multi-class accuracy — 9 known classes only
    2. Binary accuracy       — 9 known classes (benign vs attack)
    3. Multi-class accuracy — all 15 classes (known + zero-day)
    4. Binary accuracy       — all 15 classes
    5. Binary accuracy       — 6 unknown (zero-day) classes only
    (+ F1-score for all binary evaluations)

Run:
    python train_cnn_baseline.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DATA_DIR = Path(r"E:\NeuroSymbolic-IDS1\data\processed")
MODEL_DIR = Path(r"E:\NeuroSymbolic-IDS1\models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label"
BENIGN_LABEL = "BENIGN"
BATCH_SIZE = 128
TOTAL_EPOCHS = 50
CHECKPOINT_EPOCHS = [30, 50]
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
print("Loading preprocessed splits...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")
test_known_df = pd.read_csv(DATA_DIR / "test_known.csv")
test_zeroday_df = pd.read_csv(DATA_DIR / "test_zero_day.csv")

byte_cols = [c for c in train_df.columns if "payload_byte" in c]
print(f"Payload byte columns: {len(byte_cols)}")
print(f"Train: {len(train_df)} | Val: {len(val_df)} | "
      f"Test known: {len(test_known_df)} | Test zero-day: {len(test_zeroday_df)}")

# ----------------------------------------------------------------------
# Feature/label prep
# ----------------------------------------------------------------------
# Paper uses only the raw payload bytes as input (treated as a 1x1500
# "image"), not the extra ttl/total_len/protocol/t_delta metadata columns
# this release happens to include. We match that here for a faithful
# baseline; those extra columns are still available in the CSVs if you
# want to experiment with including them later.


def get_X(df):
    X = df[byte_cols].values.astype("float32") / 255.0  # normalize to [0,1]
    return X.reshape(-1, 1500, 1)


X_train = get_X(train_df)
X_val = get_X(val_df)
X_test_known = get_X(test_known_df)
X_test_zeroday = get_X(test_zeroday_df)

# Label encoder fit ONLY on the 9 known training classes -- zero-day
# labels are intentionally never seen by this encoder or the model.
label_encoder = LabelEncoder()
y_train_int = label_encoder.fit_transform(train_df[LABEL_COL])
y_val_int = label_encoder.transform(val_df[LABEL_COL])
y_test_known_int = label_encoder.transform(test_known_df[LABEL_COL])

N_CLASSES = len(label_encoder.classes_)
print(f"\nKnown classes ({N_CLASSES}): {list(label_encoder.classes_)}")

y_train = tf.keras.utils.to_categorical(y_train_int, N_CLASSES)
y_val = tf.keras.utils.to_categorical(y_val_int, N_CLASSES)
y_test_known = tf.keras.utils.to_categorical(y_test_known_int, N_CLASSES)

benign_index = list(label_encoder.classes_).index(BENIGN_LABEL)

# ----------------------------------------------------------------------
# Model (Fig. 2 architecture)
# ----------------------------------------------------------------------


def build_cnn(n_classes):
    model = models.Sequential([
        layers.Input(shape=(1500, 1)),
        layers.Conv1D(32, kernel_size=3, activation="relu"),
        layers.MaxPooling1D(pool_size=4),
        layers.Conv1D(63, kernel_size=3, activation="relu"),
        layers.MaxPooling1D(pool_size=8),
        layers.Conv1D(128, kernel_size=3, activation="relu"),
        layers.MaxPooling1D(pool_size=16),
        layers.Flatten(),
        layers.Dense(n_classes * 5, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=optimizers.Adamax(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_cnn(N_CLASSES)
model.summary()

# ----------------------------------------------------------------------
# Checkpoint callback -- saves the model at epoch 30 and epoch 50 so both
# of the paper's reported training lengths can be evaluated afterward.
# ----------------------------------------------------------------------


class EpochCheckpoint(callbacks.Callback):
    def __init__(self, target_epochs, save_dir):
        super().__init__()
        self.target_epochs = set(target_epochs)
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs=None):
        completed = epoch + 1  # epoch is 0-indexed
        if completed in self.target_epochs:
            path = self.save_dir / f"cnn_epoch_{completed}.keras"
            self.model.save(path)
            print(f"\n[checkpoint] saved model at epoch {completed} -> {path}")


checkpoint_cb = EpochCheckpoint(CHECKPOINT_EPOCHS, MODEL_DIR)

# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------
print(f"\nTraining for {TOTAL_EPOCHS} epochs, batch size {BATCH_SIZE}...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=TOTAL_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint_cb],
    verbose=1,
)

# Save training history for later plotting (loss/accuracy curves like Fig. 4)
hist_df = pd.DataFrame(history.history)
hist_df.to_csv(MODEL_DIR / "training_history.csv", index_label="epoch")
print(f"\nSaved training history -> {MODEL_DIR / 'training_history.csv'}")

# ----------------------------------------------------------------------
# Evaluation -- replicates the paper's Table II, 5 views per checkpoint
# ----------------------------------------------------------------------


def evaluate_checkpoint(model_path, epoch_label):
    print(f"\n{'=' * 60}\nEvaluating checkpoint: {epoch_label} epochs\n{'=' * 60}")
    m = tf.keras.models.load_model(model_path)

    # --- View 1 & 2: known classes only (9-class test set) ---
    pred_probs_known = m.predict(X_test_known, batch_size=BATCH_SIZE, verbose=0)
    pred_labels_known_int = np.argmax(pred_probs_known, axis=1)
    true_labels_known_int = y_test_known_int

    multiclass_acc_known = accuracy_score(true_labels_known_int, pred_labels_known_int)

    pred_binary_known = (pred_labels_known_int != benign_index).astype(int)
    true_binary_known = (true_labels_known_int != benign_index).astype(int)
    binary_acc_known = accuracy_score(true_binary_known, pred_binary_known)
    binary_f1_known = f1_score(true_binary_known, pred_binary_known)

    # --- View 3, 4, 5: all 15 classes (known test + zero-day) ---
    pred_probs_zeroday = m.predict(X_test_zeroday, batch_size=BATCH_SIZE, verbose=0)
    pred_labels_zeroday_int = np.argmax(pred_probs_zeroday, axis=1)
    pred_labels_zeroday_str = label_encoder.inverse_transform(pred_labels_zeroday_int)
    true_labels_zeroday_str = test_zeroday_df[LABEL_COL].values

    pred_labels_known_str = label_encoder.inverse_transform(pred_labels_known_int)
    true_labels_known_str = test_known_df[LABEL_COL].values

    # Multi-class over all 15: zero-day samples can only ever be "wrong"
    # in the multiclass sense, since the model has no output neuron for
    # them -- this dilution is expected and matches the paper's protocol.
    all_pred_str = np.concatenate([pred_labels_known_str, pred_labels_zeroday_str])
    all_true_str = np.concatenate([true_labels_known_str, true_labels_zeroday_str])
    multiclass_acc_all15 = accuracy_score(all_true_str, all_pred_str)

    # Binary over all 15: any non-BENIGN true label counts as "attack",
    # including zero-day classes never seen in training.
    all_true_binary = (all_true_str != BENIGN_LABEL).astype(int)
    all_pred_binary = np.concatenate([pred_binary_known,
                                       (pred_labels_zeroday_int != benign_index).astype(int)])
    binary_acc_all15 = accuracy_score(all_true_binary, all_pred_binary)
    binary_f1_all15 = f1_score(all_true_binary, all_pred_binary)

    # Binary on 6 unknown (zero-day) classes only
    true_binary_zeroday = np.ones(len(true_labels_zeroday_str), dtype=int)  # all attacks
    pred_binary_zeroday = (pred_labels_zeroday_int != benign_index).astype(int)
    binary_acc_zeroday = accuracy_score(true_binary_zeroday, pred_binary_zeroday)
    binary_f1_zeroday = f1_score(true_binary_zeroday, pred_binary_zeroday, zero_division=0)

    results = {
        "Multi-class 9 known classes (acc)": multiclass_acc_known,
        "Binary 9 known classes (acc)": binary_acc_known,
        "Binary 9 known classes (F1)": binary_f1_known,
        "Multi-class 15 classes (acc)": multiclass_acc_all15,
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
    ckpt_path = MODEL_DIR / f"cnn_epoch_{ep}.keras"
    if ckpt_path.exists():
        all_results[ep] = evaluate_checkpoint(ckpt_path, ep)
    else:
        print(f"\n[warning] checkpoint for epoch {ep} not found at {ckpt_path}")

results_df = pd.DataFrame(all_results).T
results_df.index.name = "epochs"
results_df.to_csv(MODEL_DIR / "cnn_baseline_results.csv")
print(f"\nSaved results table -> {MODEL_DIR / 'cnn_baseline_results.csv'}")
print("\nCompare these numbers against the base paper's Table II, 1D CNN row.")
