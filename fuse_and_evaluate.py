"""
Fusion: combines the symbolic KG layer's flags with the neural branch's
predictions and re-evaluates the same 5-view metric suite used throughout
this project.

Fusion rule (OR-based escalation): if EITHER the neural model predicts
"attack" OR the symbolic layer flags the packet, the final decision is
"attack". Only when both agree "benign" does the fused system call it
benign. This directly implements the design goal: the symbolic layer
should catch what the neural branch misses, not the other way around.

Multiclass metrics (which specific attack class) are left as the neural
model's own prediction unchanged -- the symbolic layer only ever produces
a binary attack/not-attack flag, it doesn't predict a specific class.

Uses the best neural checkpoint so far: hybrid_ltn_richer_omega5 (10-axiom
Hybrid-LTN, omega=5), at 30 epochs.

Run:
    py fuse_and_evaluate.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf

DATA_DIR = Path(r"E:\NeuroSymbolic-IDS1\data\processed")
KG_DIR = Path(r"E:\NeuroSymbolic-IDS1\data\kg")
MODEL_DIR = Path(r"E:\NeuroSymbolic-IDS1\models")

LABEL_COL = "label"
BENIGN_LABEL = "BENIGN"
BATCH_SIZE = 128
NEURAL_CHECKPOINT = MODEL_DIR / "hybrid_ltn_richer_omega5_epoch_30.keras"

# ----------------------------------------------------------------------
# Load data (same order as when the KG files were generated -- both were
# read from the same source CSVs with no shuffling, so row order matches)
# ----------------------------------------------------------------------
train_df = pd.read_csv(DATA_DIR / "train.csv")  # only needed to fit label encoder
test_known_df = pd.read_csv(DATA_DIR / "test_known.csv")
test_zeroday_df = pd.read_csv(DATA_DIR / "test_zero_day.csv")

test_known_kg = pd.read_csv(KG_DIR / "test_known_kg.csv")
test_zeroday_kg = pd.read_csv(KG_DIR / "test_zero_day_kg.csv")

assert len(test_known_df) == len(test_known_kg), "test_known row count mismatch -- check KG file generation"
assert len(test_zeroday_df) == len(test_zeroday_kg), "test_zero_day row count mismatch -- check KG file generation"

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
print(f"Test known: {len(test_known_df)} | Test zero-day: {len(test_zeroday_df)}")

# ----------------------------------------------------------------------
# Neural predictions
# ----------------------------------------------------------------------
print(f"\nLoading neural checkpoint: {NEURAL_CHECKPOINT}")
model = tf.keras.models.load_model(NEURAL_CHECKPOINT)

pred_probs_known = model.predict(X_test_known, batch_size=BATCH_SIZE, verbose=0)
pred_labels_known_int = np.argmax(pred_probs_known, axis=1)
neural_binary_known = (pred_labels_known_int != benign_index).astype(int)

pred_probs_zeroday = model.predict(X_test_zeroday, batch_size=BATCH_SIZE, verbose=0)
pred_labels_zeroday_int = np.argmax(pred_probs_zeroday, axis=1)
neural_binary_zeroday = (pred_labels_zeroday_int != benign_index).astype(int)

# ----------------------------------------------------------------------
# Symbolic flags (already computed and saved)
# ----------------------------------------------------------------------
symbolic_binary_known = test_known_kg["symbolic_flag"].astype(int).values
symbolic_binary_zeroday = test_zeroday_kg["symbolic_flag"].astype(int).values

# No-TTL variant: excludes r3_ttl_anomaly, which may be partly detecting
# testbed artifacts (attacker vs. benign traffic from different VM images)
# rather than genuine attack behavior -- see ablate_ttl_rule.py findings.
symbolic_no_ttl_known = (test_known_kg["r1_flood"] | test_known_kg["r2_scan"] |
                          test_known_kg["r4_high_entropy"]).astype(int).values
symbolic_no_ttl_zeroday = (test_zeroday_kg["r1_flood"] | test_zeroday_kg["r2_scan"] |
                            test_zeroday_kg["r4_high_entropy"]).astype(int).values

# ----------------------------------------------------------------------
# Fusion: OR-based escalation
# ----------------------------------------------------------------------
fused_binary_known = ((neural_binary_known == 1) | (symbolic_binary_known == 1)).astype(int)
fused_binary_zeroday = ((neural_binary_zeroday == 1) | (symbolic_binary_zeroday == 1)).astype(int)

# Conservative fusion variant, excluding the TTL rule
fused_no_ttl_known = ((neural_binary_known == 1) | (symbolic_no_ttl_known == 1)).astype(int)
fused_no_ttl_zeroday = ((neural_binary_zeroday == 1) | (symbolic_no_ttl_zeroday == 1)).astype(int)

true_binary_known = (y_test_known_int != benign_index).astype(int)
true_binary_zeroday = np.ones(len(test_zeroday_df), dtype=int)  # zero-day set is all attacks

# ----------------------------------------------------------------------
# Evaluate: neural-only vs. symbolic-only vs. fused, side by side
# ----------------------------------------------------------------------


def report(name, pred_known, pred_zeroday):
    acc_known = accuracy_score(true_binary_known, pred_known)
    f1_known = f1_score(true_binary_known, pred_known)
    acc_zeroday = accuracy_score(true_binary_zeroday, pred_zeroday)
    f1_zeroday = f1_score(true_binary_zeroday, pred_zeroday, zero_division=0)

    # Binary-15: weighted combination (same convention as evaluate_cnn_checkpoints.py)
    n_known = len(true_binary_known)
    n_zeroday = len(true_binary_zeroday)
    n_total = n_known + n_zeroday
    correct_known = (pred_known == true_binary_known).sum()
    correct_zeroday = pred_zeroday.sum()  # true is always 1
    acc_15 = (correct_known + correct_zeroday) / n_total

    all_true = np.concatenate([true_binary_known, true_binary_zeroday])
    all_pred = np.concatenate([pred_known, pred_zeroday])
    f1_15 = f1_score(all_true, all_pred)

    print(f"\n--- {name} ---")
    print(f"  Binary 9 known (acc):     {acc_known*100:.2f}%")
    print(f"  Binary 9 known (F1):      {f1_known*100:.2f}%")
    print(f"  Binary 15 (acc):          {acc_15*100:.2f}%")
    print(f"  Binary 15 (F1):           {f1_15*100:.2f}%")
    print(f"  Binary 6 unknown (acc):   {acc_zeroday*100:.2f}%")
    print(f"  Binary 6 unknown (F1):    {f1_zeroday*100:.2f}%")
    return {
        "binary_9_known_acc": acc_known, "binary_9_known_f1": f1_known,
        "binary_15_acc": acc_15, "binary_15_f1": f1_15,
        "binary_6_unknown_acc": acc_zeroday, "binary_6_unknown_f1": f1_zeroday,
    }


results = {}
results["neural_only"] = report("Neural only (Hybrid-LTN richer, omega=5)",
                                  neural_binary_known, neural_binary_zeroday)
results["symbolic_only"] = report("Symbolic only (KG rules, with TTL)",
                                    symbolic_binary_known, symbolic_binary_zeroday)
results["symbolic_only_no_ttl"] = report("Symbolic only (KG rules, NO TTL)",
                                           symbolic_no_ttl_known, symbolic_no_ttl_zeroday)
results["fused"] = report("Fused (OR escalation, with TTL)",
                            fused_binary_known, fused_binary_zeroday)
results["fused_no_ttl"] = report("Fused (OR escalation, NO TTL -- conservative)",
                                   fused_no_ttl_known, fused_no_ttl_zeroday)

results_df = pd.DataFrame(results).T
results_df.to_csv(MODEL_DIR / "fusion_results.csv")
print(f"\nSaved comparison table -> {MODEL_DIR / 'fusion_results.csv'}")
