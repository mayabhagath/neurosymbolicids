"""
Hybrid-LTN training — replicates the base paper's (Bizzarri et al., ICCCN 2024)
LTN layer on top of the same CNN architecture used in the plain baseline.

Predicate:
    P(x, l) = l^T . softmax(CNN(x))

Axioms (benign vs. non-benign only, per paper Eq. 6):
    forall x_b : P(x_b, l_b)          (benign examples should satisfy P w.r.t. benign label)
    forall x_a : NOT P(x_a, l_b)      (attack examples should NOT satisfy P w.r.t. benign label)

Aggregation: product fuzzy logic, p=2 (paper's stated choice).

Loss:
    SAT_loss    = 1 - SatAgg(axioms)
    Hybrid_loss = CE_loss + omega * SAT_loss     (omega = 1, per paper)

Requires:
    pip install ltn

Run:
    python train_hybrid_ltn.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import ltn

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
OMEGA = 5.0  # weight on SAT_loss -- paper used 1.0; testing whether a
             # higher weight keeps the axiom term influential for longer
             # (see note below on SAT loss saturating early at omega=1)
P_VALUE = 2  # aggregator exponent, per paper
RANDOM_SEED = 42
CHECKPOINT_PREFIX = "hybrid_ltn_omega5"  # distinct name so this run doesn't
                                          # overwrite the omega=1 checkpoints

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
print("Loading preprocessed splits...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")

byte_cols = [c for c in train_df.columns if "payload_byte" in c]
print(f"Payload byte columns: {len(byte_cols)}")
print(f"Train: {len(train_df)} | Val: {len(val_df)}")


def get_X(df):
    X = df[byte_cols].values.astype("float32") / 255.0
    return X.reshape(-1, 1500, 1)


X_train = get_X(train_df)
X_val = get_X(val_df)

label_encoder = LabelEncoder()
y_train_int = label_encoder.fit_transform(train_df[LABEL_COL])
y_val_int = label_encoder.transform(val_df[LABEL_COL])

N_CLASSES = len(label_encoder.classes_)
print(f"\nKnown classes ({N_CLASSES}): {list(label_encoder.classes_)}")

y_train_onehot = tf.keras.utils.to_categorical(y_train_int, N_CLASSES).astype("float32")
y_val_onehot = tf.keras.utils.to_categorical(y_val_int, N_CLASSES).astype("float32")

benign_index = list(label_encoder.classes_).index(BENIGN_LABEL)
is_benign_train = (y_train_int == benign_index)

# Fixed one-hot "benign" label constant, reused in every axiom evaluation
l_benign_onehot = tf.one_hot(benign_index, N_CLASSES, dtype=tf.float32)

# ----------------------------------------------------------------------
# CNN (same architecture as the plain baseline, Fig. 2)
# ----------------------------------------------------------------------


def build_cnn(n_classes):
    inputs = tf.keras.Input(shape=(1500, 1))
    x = tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(63, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=8)(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=16)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(n_classes * 5, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


cnn_model = build_cnn(N_CLASSES)
cnn_model.summary()

# ----------------------------------------------------------------------
# LTN wiring
# ----------------------------------------------------------------------
# P(x, l) = l . softmax(CNN(x))  -- dot product of the one-hot label with
# the CNN's softmax output, exactly as defined in the paper.


class PModel(tf.keras.Model):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def call(self, x, l=None):
        probs = self.cnn(x)                      # (batch, N_CLASSES)
        l_tensor = l.tensor if isinstance(l, ltn.Variable) else l
        return tf.reduce_sum(probs * l_tensor, axis=-1)  # (batch,)


P = ltn.Predicate(PModel(cnn_model))

Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Forall = ltn.Wrapper_Quantifier(
    ltn.fuzzy_ops.Aggreg_pMeanError(p=P_VALUE), semantics="forall"
)
# NOTE: the TF `ltn` package doesn't ship a SatAgg class (that's only in the
# newer PyTorch LTNtorch port). With only two axioms in the knowledge base
# (benign satisfaction, attack satisfaction), the paper's formula-aggregating
# operator SatAgg reduces to a simple mean of their truth values -- a
# mathematically equivalent, dependency-free substitute.


def sat_agg_fn(*sat_values):
    return tf.reduce_mean(tf.stack(sat_values))

ce_loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adamax()

# ----------------------------------------------------------------------
# tf.data pipeline
# ----------------------------------------------------------------------
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_onehot, is_benign_train))
train_ds = train_ds.shuffle(buffer_size=10000, seed=RANDOM_SEED).batch(BATCH_SIZE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_onehot))
val_ds = val_ds.batch(BATCH_SIZE)


@tf.function
def train_step(x_batch, y_batch, is_benign_batch):
    with tf.GradientTape() as tape:
        # Standard cross-entropy over the full (mixed) batch
        probs = cnn_model(x_batch, training=True)
        ce_loss = ce_loss_fn(y_batch, probs)

        # Split the batch into benign / attack for the axioms.
        # If a batch happens to have zero benign or zero attack examples
        # (rare with this batch size but possible), fall back to SAT=1
        # (no penalty) for that missing side.
        x_b = tf.boolean_mask(x_batch, is_benign_batch)
        x_a = tf.boolean_mask(x_batch, ~is_benign_batch)

        n_b = tf.shape(x_b)[0]
        n_a = tf.shape(x_a)[0]

        l_b_var = tf.tile(tf.expand_dims(l_benign_onehot, 0), [tf.maximum(n_b, 1), 1])
        l_a_var = tf.tile(tf.expand_dims(l_benign_onehot, 0), [tf.maximum(n_a, 1), 1])

        def sat_benign():
            xb_var = ltn.Variable("x_b", x_b)
            lb_var = ltn.Variable("l_b", l_b_var)
            return Forall(ltn.diag(xb_var, lb_var), P(xb_var, l=lb_var)).tensor

        def sat_attack():
            xa_var = ltn.Variable("x_a", x_a)
            la_var = ltn.Variable("l_a", l_a_var)
            return Forall(ltn.diag(xa_var, la_var), Not(P(xa_var, l=la_var))).tensor

        sat_b = tf.cond(n_b > 0, sat_benign, lambda: tf.constant(1.0))
        sat_a = tf.cond(n_a > 0, sat_attack, lambda: tf.constant(1.0))

        sat_agg = sat_agg_fn(sat_b, sat_a)
        sat_loss = 1.0 - sat_agg

        hybrid_loss = ce_loss + OMEGA * sat_loss

    gradients = tape.gradient(hybrid_loss, cnn_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, cnn_model.trainable_variables))

    pred_labels = tf.argmax(probs, axis=1)
    true_labels = tf.argmax(y_batch, axis=1)
    acc = tf.reduce_mean(tf.cast(pred_labels == true_labels, tf.float32))

    return hybrid_loss, ce_loss, sat_loss, acc


@tf.function
def val_step(x_batch, y_batch):
    probs = cnn_model(x_batch, training=False)
    ce_loss = ce_loss_fn(y_batch, probs)
    pred_labels = tf.argmax(probs, axis=1)
    true_labels = tf.argmax(y_batch, axis=1)
    acc = tf.reduce_mean(tf.cast(pred_labels == true_labels, tf.float32))
    return ce_loss, acc


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
history = []
print(f"\nTraining Hybrid-LTN for {TOTAL_EPOCHS} epochs, batch size {BATCH_SIZE}...")

for epoch in range(1, TOTAL_EPOCHS + 1):
    epoch_hybrid, epoch_ce, epoch_sat, epoch_acc, n_batches = 0.0, 0.0, 0.0, 0.0, 0
    for x_batch, y_batch, is_benign_batch in train_ds:
        hybrid_loss, ce_loss, sat_loss, acc = train_step(x_batch, y_batch, is_benign_batch)
        epoch_hybrid += float(hybrid_loss)
        epoch_ce += float(ce_loss)
        epoch_sat += float(sat_loss)
        epoch_acc += float(acc)
        n_batches += 1

    epoch_hybrid /= n_batches
    epoch_ce /= n_batches
    epoch_sat /= n_batches
    epoch_acc /= n_batches

    val_ce, val_acc, n_val_batches = 0.0, 0.0, 0
    for x_batch, y_batch in val_ds:
        ce_loss, acc = val_step(x_batch, y_batch)
        val_ce += float(ce_loss)
        val_acc += float(acc)
        n_val_batches += 1
    val_ce /= n_val_batches
    val_acc /= n_val_batches

    print(f"Epoch {epoch}/{TOTAL_EPOCHS} - "
          f"hybrid_loss: {epoch_hybrid:.4f} - ce: {epoch_ce:.4f} - sat: {epoch_sat:.4f} - "
          f"acc: {epoch_acc:.4f} - val_ce: {val_ce:.4f} - val_acc: {val_acc:.4f}")

    history.append({
        "epoch": epoch, "hybrid_loss": epoch_hybrid, "ce_loss": epoch_ce,
        "sat_loss": epoch_sat, "accuracy": epoch_acc,
        "val_ce_loss": val_ce, "val_accuracy": val_acc,
    })

    if epoch in CHECKPOINT_EPOCHS:
        ckpt_path = MODEL_DIR / f"{CHECKPOINT_PREFIX}_epoch_{epoch}.keras"
        cnn_model.save(ckpt_path)
        print(f"[checkpoint] saved model at epoch {epoch} -> {ckpt_path}")

hist_df = pd.DataFrame(history)
hist_df.to_csv(MODEL_DIR / f"{CHECKPOINT_PREFIX}_training_history.csv", index=False)
print(f"\nSaved training history -> {MODEL_DIR / f'{CHECKPOINT_PREFIX}_training_history.csv'}")

# Quick visibility into whether the higher omega actually kept the SAT
# loss meaningfully non-zero for longer than the omega=1 run did.
print("\nSAT loss at selected epochs (compare against your omega=1 run's log):")
for ep in [1, 10, 20, 30, 40, 50]:
    row = hist_df[hist_df["epoch"] == ep]
    if not row.empty:
        print(f"  epoch {ep}: sat_loss = {row['sat_loss'].values[0]:.4f}")

print(f"\nNow run evaluate_cnn_checkpoints.py with MODEL_PREFIX = '{CHECKPOINT_PREFIX}' "
      f"to score these checkpoints against your omega=1 Hybrid-LTN and CNN baseline.")
