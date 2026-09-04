"""
Hybrid-LTN with richer axioms -- extends the base paper's 2-axiom Hybrid Loss
with per-class satisfaction axioms, testing whether finer-grained logic
constraints improve zero-day generalization beyond what the paper's coarse
benign/non-benign axioms alone achieve.

Base paper's axioms (kept):
    forall x_b : P(x_b, l_b)          benign examples satisfy P w.r.t. benign
    forall x_a : NOT P(x_a, l_b)      attack examples do NOT satisfy P w.r.t. benign

New axioms added (one per known attack class c, 8 total):
    forall x_c : P(x_c, l_c)          examples of attack class c satisfy P
                                       w.r.t. THEIR OWN specific label

Rationale: the paper's Hybrid-LTN only ever reasons about "attack vs. not,"
collapsing all 8 known attack types into one region. Forcing distinct,
confident per-class regions may produce a better-structured attack-side
feature space, giving novel (zero-day) attacks more "surface area" to be
recognized as attack-like, rather than everything attack-related being one
undifferentiated blob.

SAT_loss now aggregates 10 axioms total (2 base + 8 per-class) via mean.
Everything else (CNN architecture, CE loss, omega weighting) unchanged from
the omega=5 Hybrid-LTN run.

Requires:
    pip install ltn

Run:
    py train_hybrid_ltn_richer_axioms.py
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
OMEGA = 5.0  # carried over from the best-performing prior experiment
P_VALUE = 2
RANDOM_SEED = 42
CHECKPOINT_PREFIX = "hybrid_ltn_richer_omega5"

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
CLASS_NAMES = list(label_encoder.classes_)
print(f"\nKnown classes ({N_CLASSES}): {CLASS_NAMES}")

y_train_onehot = tf.keras.utils.to_categorical(y_train_int, N_CLASSES).astype("float32")
y_val_onehot = tf.keras.utils.to_categorical(y_val_int, N_CLASSES).astype("float32")

benign_index = list(label_encoder.classes_).index(BENIGN_LABEL)
attack_class_indices = [i for i in range(N_CLASSES) if i != benign_index]

is_benign_train = (y_train_int == benign_index)

# One-hot constants for every class, reused across axiom evaluations
l_onehot_per_class = [tf.one_hot(i, N_CLASSES, dtype=tf.float32) for i in range(N_CLASSES)]

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


class PModel(tf.keras.Model):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def call(self, x, l=None):
        probs = self.cnn(x)
        l_tensor = l.tensor if isinstance(l, ltn.Variable) else l
        return tf.reduce_sum(probs * l_tensor, axis=-1)


P = ltn.Predicate(PModel(cnn_model))

Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
Forall = ltn.Wrapper_Quantifier(
    ltn.fuzzy_ops.Aggreg_pMeanError(p=P_VALUE), semantics="forall"
)


def sat_agg_fn(*sat_values):
    return tf.reduce_mean(tf.stack(sat_values))


ce_loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adamax()

# ----------------------------------------------------------------------
# tf.data pipeline -- now carries the integer label too, needed to split
# the batch per-class for the new axioms
# ----------------------------------------------------------------------
train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train_onehot, is_benign_train, y_train_int.astype("int32"))
)
train_ds = train_ds.shuffle(buffer_size=10000, seed=RANDOM_SEED).batch(BATCH_SIZE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_onehot))
val_ds = val_ds.batch(BATCH_SIZE)


@tf.function
def train_step(x_batch, y_batch, is_benign_batch, y_int_batch):
    with tf.GradientTape() as tape:
        probs = cnn_model(x_batch, training=True)
        ce_loss = ce_loss_fn(y_batch, probs)

        sat_values = []

        # --- Base axioms (benign / non-benign), same as before ---
        x_b = tf.boolean_mask(x_batch, is_benign_batch)
        x_a = tf.boolean_mask(x_batch, ~is_benign_batch)
        n_b = tf.shape(x_b)[0]
        n_a = tf.shape(x_a)[0]

        l_b_var = tf.tile(tf.expand_dims(l_onehot_per_class[benign_index], 0), [tf.maximum(n_b, 1), 1])
        l_a_var = tf.tile(tf.expand_dims(l_onehot_per_class[benign_index], 0), [tf.maximum(n_a, 1), 1])

        def sat_benign():
            xb_var = ltn.Variable("x_b", x_b)
            lb_var = ltn.Variable("l_b", l_b_var)
            return Forall(ltn.diag(xb_var, lb_var), P(xb_var, l=lb_var)).tensor

        def sat_nonbenign():
            xa_var = ltn.Variable("x_a", x_a)
            la_var = ltn.Variable("l_a", l_a_var)
            return Forall(ltn.diag(xa_var, la_var), Not(P(xa_var, l=la_var))).tensor

        sat_values.append(tf.cond(n_b > 0, sat_benign, lambda: tf.constant(1.0)))
        sat_values.append(tf.cond(n_a > 0, sat_nonbenign, lambda: tf.constant(1.0)))

        # --- New per-class axioms: forall x_c : P(x_c, l_c) ---
        for c in attack_class_indices:
            mask_c = tf.equal(y_int_batch, c)
            x_c = tf.boolean_mask(x_batch, mask_c)
            n_c = tf.shape(x_c)[0]
            l_c_var = tf.tile(tf.expand_dims(l_onehot_per_class[c], 0), [tf.maximum(n_c, 1), 1])

            def sat_class(x_c=x_c, l_c_var=l_c_var, c=c):
                xc_var = ltn.Variable(f"x_c{c}", x_c)
                lc_var = ltn.Variable(f"l_c{c}", l_c_var)
                return Forall(ltn.diag(xc_var, lc_var), P(xc_var, l=lc_var)).tensor

            sat_values.append(tf.cond(n_c > 0, sat_class, lambda: tf.constant(1.0)))

        sat_agg = sat_agg_fn(*sat_values)
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
print(f"\nTraining Hybrid-LTN (richer axioms) for {TOTAL_EPOCHS} epochs, "
      f"batch size {BATCH_SIZE}, omega={OMEGA}, {2 + len(attack_class_indices)} total axioms...")

for epoch in range(1, TOTAL_EPOCHS + 1):
    epoch_hybrid, epoch_ce, epoch_sat, epoch_acc, n_batches = 0.0, 0.0, 0.0, 0.0, 0
    for x_batch, y_batch, is_benign_batch, y_int_batch in train_ds:
        hybrid_loss, ce_loss, sat_loss, acc = train_step(x_batch, y_batch, is_benign_batch, y_int_batch)
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

print("\nSAT loss at selected epochs (compare against your omega=5, 2-axiom run):")
for ep in [1, 10, 20, 30, 40, 50]:
    row = hist_df[hist_df["epoch"] == ep]
    if not row.empty:
        print(f"  epoch {ep}: sat_loss = {row['sat_loss'].values[0]:.4f}")

print(f"\nNow run evaluate_cnn_checkpoints.py with MODEL_PREFIX = '{CHECKPOINT_PREFIX}' "
      f"to score these checkpoints against your other three baselines.")
