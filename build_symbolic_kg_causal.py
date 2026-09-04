"""
Rebuilds the symbolic layer's flood/scan rules using the CAUSAL repetition
count (from compute_causal_repetition.py) instead of the original
transductive, whole-split repetition count.

Also drops R3 (TTL anomaly) from the PRIMARY evaluation, per the ablation
finding that it likely reflects a CICIDS2017 testbed artifact (attacker vs.
benign traffic from different VM images) rather than genuine attack
behavior -- it is still computed and reported for transparency, but no
longer folded into "symbolic_flag" by default.

Prerequisites (run in this order):
    1. preprocess_cicids.py (updated version, preserves orig_index)
    2. build_symbolic_kg.py (updated version, preserves orig_index) --
       still needed for nonzero_bytes/entropy, which are legitimately
       per-row features (not transductive) and don't need recomputing
    3. compute_causal_repetition.py -- produces causal_repetition.csv
    4. this script

Run:
    py build_symbolic_kg_causal.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

KG_DIR = Path(r"E:\NeuroSymbolic-IDS1\causal_kg\data\kg")  # NEW location
CAUSAL_PATH = Path(r"E:\NeuroSymbolic-IDS1\causal_kg\data\causal_repetition.csv")  # NEW location

LABEL_COL = "label"
BENIGN_LABEL = "BENIGN"

# Same conceptual rule as before, but the threshold is now fit on the
# CAUSAL count (which is naturally comparable across splits of different
# sizes, since the window size is a fixed constant -- unlike the old
# transductive repetition_fraction, which needed per-split normalization).
FLOOD_REPETITION_PERCENTILE = 99.0
SCAN_NONZERO_BYTE_MAX = 25
SCAN_REPETITION_MIN = 5

print("Loading existing KG feature files (nonzero_bytes, entropy, label)...")
train_kg = pd.read_csv(KG_DIR / "train_kg.csv")
val_kg = pd.read_csv(KG_DIR / "val_kg.csv")
test_known_kg = pd.read_csv(KG_DIR / "test_known_kg.csv")
test_zeroday_kg = pd.read_csv(KG_DIR / "test_zero_day_kg.csv")

for name, df in [("train", train_kg), ("val", val_kg),
                  ("test_known", test_known_kg), ("test_zero_day", test_zeroday_kg)]:
    if "orig_index" not in df.columns:
        raise ValueError(
            f"{name}_kg.csv has no 'orig_index' column. You need to re-run "
            f"preprocess_cicids.py and build_symbolic_kg.py with the updated "
            f"versions that preserve orig_index before this script will work."
        )

print("Loading causal repetition counts...")
causal_rep = pd.read_csv(CAUSAL_PATH)
print(f"Causal repetition file: {len(causal_rep)} rows")


def attach_causal(df, name):
    merged = df.merge(causal_rep, on="orig_index", how="left")
    n_missing = merged["causal_repetition_count"].isna().sum()
    if n_missing > 0:
        print(f"  [warning] {name}: {n_missing} rows failed to match on orig_index")
    print(f"  {name}: {len(merged)} rows after merge")
    return merged


print("\nRejoining causal repetition counts onto each split...")
train_c = attach_causal(train_kg, "train")
val_c = attach_causal(val_kg, "val")
test_known_c = attach_causal(test_known_kg, "test_known")
test_zeroday_c = attach_causal(test_zeroday_kg, "test_zero_day")

# ----------------------------------------------------------------------
# Fit flood threshold on BENIGN train's CAUSAL repetition count
# ----------------------------------------------------------------------
benign_train_c = train_c[train_c[LABEL_COL] == BENIGN_LABEL]
flood_threshold = np.percentile(benign_train_c["causal_repetition_count"], FLOOD_REPETITION_PERCENTILE)
print(f"\nFitted causal flood threshold (P{FLOOD_REPETITION_PERCENTILE} of benign train): "
      f"{flood_threshold:.1f}")


def apply_causal_rules(df):
    df = df.copy()
    df["r1_flood_causal"] = df["causal_repetition_count"] > flood_threshold
    df["r2_scan_causal"] = (df["nonzero_bytes"] <= SCAN_NONZERO_BYTE_MAX) & \
                            (df["causal_repetition_count"] >= SCAN_REPETITION_MIN)
    # Primary flag: causal rules only, NO ttl, NO entropy dependency on
    # whole-split statistics (entropy itself is a legitimate per-row
    # feature, kept from the original r4_high_entropy column unchanged).
    df["symbolic_flag_causal"] = df["r1_flood_causal"] | df["r2_scan_causal"] | df["r4_high_entropy"]
    return df


def summarize(name, df, has_labels=True):
    print(f"\n--- {name} ---")
    total = len(df)
    flagged = df["symbolic_flag_causal"].sum()
    print(f"Flagged (causal, no TTL): {flagged}/{total} ({100*flagged/total:.2f}%)")
    print(f"  r1_flood_causal: {df['r1_flood_causal'].sum()} "
          f"({100*df['r1_flood_causal'].sum()/total:.2f}%)")
    print(f"  r2_scan_causal:  {df['r2_scan_causal'].sum()} "
          f"({100*df['r2_scan_causal'].sum()/total:.2f}%)")
    print(f"  r4_high_entropy: {df['r4_high_entropy'].sum()} "
          f"({100*df['r4_high_entropy'].sum()/total:.2f}%)")

    if has_labels:
        is_attack = df[LABEL_COL] != BENIGN_LABEL
        attack_total = is_attack.sum()
        attack_caught = (is_attack & df["symbolic_flag_causal"]).sum()
        print(f"Attack recall: {attack_caught}/{attack_total} "
              f"({100*attack_caught/attack_total if attack_total else 0:.2f}%)")
        is_benign = ~is_attack
        benign_total = is_benign.sum()
        benign_flagged = (is_benign & df["symbolic_flag_causal"]).sum()
        print(f"Benign false-positive rate: {benign_flagged}/{benign_total} "
              f"({100*benign_flagged/benign_total if benign_total else 0:.2f}%)")
    else:
        attack_total = total
        attack_caught = df["symbolic_flag_causal"].sum()
        print(f"Zero-day attack recall: {attack_caught}/{attack_total} "
              f"({100*attack_caught/attack_total:.2f}%)")


train_c = apply_causal_rules(train_c)
val_c = apply_causal_rules(val_c)
test_known_c = apply_causal_rules(test_known_c)
test_zeroday_c = apply_causal_rules(test_zeroday_c)

summarize("Train (causal)", train_c)
summarize("Val (causal)", val_c)
summarize("Test known (causal)", test_known_c)
summarize("Test zero-day (causal)", test_zeroday_c, has_labels=False)

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
keep_cols = ["orig_index", "causal_repetition_count", "nonzero_bytes", "entropy",
             "r1_flood_causal", "r2_scan_causal", "r4_high_entropy",
             "symbolic_flag_causal", LABEL_COL]

train_c[keep_cols].to_csv(KG_DIR / "train_kg_causal.csv", index=False)
val_c[keep_cols].to_csv(KG_DIR / "val_kg_causal.csv", index=False)
test_known_c[keep_cols].to_csv(KG_DIR / "test_known_kg_causal.csv", index=False)
test_zeroday_c[keep_cols].to_csv(KG_DIR / "test_zero_day_kg_causal.csv", index=False)

print(f"\nSaved causal KG files to {KG_DIR}")
print("\nNext: update fuse_and_evaluate.py to use symbolic_flag_causal from "
      "these _kg_causal.csv files instead of the old transductive symbolic_flag, "
      "and re-run the fusion comparison.")
