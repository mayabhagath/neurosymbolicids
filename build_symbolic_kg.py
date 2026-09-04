"""
Symbolic rule engine (the "KG layer") -- extracts interpretable, non-learned
features per packet and applies threshold rules to flag suspicious traffic,
independently of the CNN/Hybrid-LTN branch.

Design rationale:
    The available features (payload bytes + ttl/total_len/protocol/t_delta)
    lack IP/port/session context, so this cannot reconstruct real network
    flows. Instead it reasons about structural properties a per-packet CNN
    cannot see directly: template repetition, payload sparsity, TTL
    anomalies, and payload randomness -- each targeting a specific gap
    identified earlier in this project (e.g. the DDoS repetition finding).

Rules (each independently interpretable, with its own threshold):
    R1 (flood/repetition):   this exact packet signature recurs abnormally
                              often within the traffic being examined
    R2 (scan-like):          very sparse payload + moderate repetition
    R3 (ttl anomaly):        TTL outside common OS-default values
    R4 (high entropy):       payload byte distribution is unusually random

KNOWN LIMITATION (fixed in compute_causal_repetition.py + the "_causal"
rebuild script): repetition_count/repetition_fraction here are computed
over each split's ENTIRE population, including rows "after" any given
packet -- a real streaming detector could never do this. Also, R3 (TTL)
was found via ablation to likely reflect a CICIDS2017 testbed artifact
(attacker vs. benign traffic generated from different VM images with
different default TTLs) rather than genuine attack behavior. Both issues
are addressed by the causal pipeline; this script is kept as-is so the
before/after comparison stays reproducible.

Methodology: all thresholds are FIT ON THE TRAINING SPLIT ONLY (percentile
cutoffs), then frozen and applied unchanged to val/test_known/test_zero_day
-- same train/test discipline as the neural models, no leakage.

Run:
    py build_symbolic_kg.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"E:\NeuroSymbolic-IDS1\causal_kg\data\processed")  # NEW location
OUT_DIR = Path(r"E:\NeuroSymbolic-IDS1\causal_kg\data\kg")  # NEW location
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label"
BENIGN_LABEL = "BENIGN"

# Percentile / fixed thresholds -- tune later if needed, but the *fitting*
# itself is done against BENIGN training traffic only (the "normal"
# baseline), never the full mixed population. Fitting against a population
# that already contains attacks risks the anomaly itself corrupting the
# threshold (this is exactly what happened in an earlier version of this
# script: DDoS's own massive repetition spike pushed the P99 cutoff high
# enough to swallow itself, making the rule never fire).
FLOOD_REPETITION_PERCENTILE = 99.0   # top 1% most-repeated templates in BENIGN train
SCAN_NONZERO_BYTE_MAX = 25           # "sparse payload" cutoff (DDoS median was ~20)
SCAN_REPETITION_MIN = 5              # must recur at least a handful of times, not just once
ENTROPY_PERCENTILE = 99.0            # top 1% highest-entropy payloads in BENIGN train
TTL_COVERAGE_FRACTION = 0.99         # "common" TTLs = smallest set covering 99% of
                                      # BENIGN traffic, empirically -- NOT a generic
                                      # assumed OS-default set (which turned out wrong
                                      # for this capture: it fired on 31.5% of benign
                                      # traffic in an earlier version of this script)

print("Loading preprocessed splits...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")
test_known_df = pd.read_csv(DATA_DIR / "test_known.csv")
test_zeroday_df = pd.read_csv(DATA_DIR / "test_zero_day.csv")

byte_cols = [c for c in train_df.columns if "payload_byte" in c]
meta_cols = ["ttl", "total_len", "protocol"]
template_cols = byte_cols + meta_cols

print(f"Train: {len(train_df)} | Val: {len(val_df)} | "
      f"Test known: {len(test_known_df)} | Test zero-day: {len(test_zeroday_df)}")


# ----------------------------------------------------------------------
# Feature extraction (applied independently to each split -- repetition
# and entropy are computed WITHIN each split's own population, since the
# rule is "does this traffic contain repeated/anomalous packets", not
# "does this match something from training". TTL/entropy thresholds are
# still frozen from training.)
# ----------------------------------------------------------------------


def compute_features(df):
    df = df.copy()

    # non-zero payload byte count (sparsity)
    byte_vals = df[byte_cols].values
    df["nonzero_bytes"] = (byte_vals != 0).sum(axis=1)

    # Shannon entropy of the payload byte distribution per packet
    def row_entropy(row):
        counts = np.bincount(row.astype(np.uint8), minlength=256)
        probs = counts[counts > 0] / row.shape[0]
        return -np.sum(probs * np.log2(probs))

    print("  computing per-packet entropy (this is the slow step)...")
    df["entropy"] = np.apply_along_axis(row_entropy, 1, byte_vals)

    # repetition count: how many times this exact template appears WITHIN
    # this split's own population -- expressed as a FRACTION of the split
    # size, not a raw count, so the same threshold is meaningful across
    # splits of very different sizes (train is 8x larger than test_known).
    template_counts = df.groupby(template_cols)[template_cols[0]].transform("size")
    df["repetition_count"] = template_counts
    df["repetition_fraction"] = template_counts / len(df)

    return df


print("\nExtracting features for train (used to fit thresholds)...")
train_feat = compute_features(train_df)

# ----------------------------------------------------------------------
# Fit thresholds on BENIGN TRAINING TRAFFIC ONLY -- this is the "normal"
# baseline. Fitting against the full mixed population (including attacks)
# is what caused the flood rule to swallow itself in an earlier version.
# ----------------------------------------------------------------------
benign_train_feat = train_feat[train_feat[LABEL_COL] == BENIGN_LABEL]

flood_threshold = np.percentile(benign_train_feat["repetition_fraction"], FLOOD_REPETITION_PERCENTILE)
entropy_threshold = np.percentile(benign_train_feat["entropy"], ENTROPY_PERCENTILE)

# Empirically derive "common" TTLs from benign traffic: the smallest set of
# TTL values whose cumulative frequency covers TTL_COVERAGE_FRACTION of
# benign packets -- not an assumed generic OS-default set.
ttl_counts = benign_train_feat["ttl"].value_counts(normalize=True).sort_values(ascending=False)
cumulative = ttl_counts.cumsum()
COMMON_TTLS = set(ttl_counts[cumulative <= TTL_COVERAGE_FRACTION].index)
if not COMMON_TTLS:  # guard: always include at least the single most common value
    COMMON_TTLS = {ttl_counts.index[0]}

print(f"\nFitted thresholds (from BENIGN train traffic only):")
print(f"  flood repetition_fraction > {flood_threshold:.6f} (P{FLOOD_REPETITION_PERCENTILE} of benign)")
print(f"  entropy > {entropy_threshold:.3f} bits (P{ENTROPY_PERCENTILE} of benign)")
print(f"  scan: nonzero_bytes <= {SCAN_NONZERO_BYTE_MAX} AND repetition_count >= {SCAN_REPETITION_MIN}")
print(f"  common TTLs (covering {TTL_COVERAGE_FRACTION*100:.0f}% of benign traffic): "
      f"{sorted(COMMON_TTLS)}")


# ----------------------------------------------------------------------
# Apply rules
# ----------------------------------------------------------------------


def apply_rules(df_feat):
    r1_flood = df_feat["repetition_fraction"] > flood_threshold
    r2_scan = (df_feat["nonzero_bytes"] <= SCAN_NONZERO_BYTE_MAX) & \
              (df_feat["repetition_count"] >= SCAN_REPETITION_MIN)
    r3_ttl = ~df_feat["ttl"].isin(COMMON_TTLS)
    r4_entropy = df_feat["entropy"] > entropy_threshold

    df_feat = df_feat.copy()
    df_feat["r1_flood"] = r1_flood
    df_feat["r2_scan"] = r2_scan
    df_feat["r3_ttl_anomaly"] = r3_ttl
    df_feat["r4_high_entropy"] = r4_entropy
    df_feat["symbolic_flag"] = r1_flood | r2_scan | r3_ttl | r4_entropy
    df_feat["n_rules_fired"] = r1_flood.astype(int) + r2_scan.astype(int) + \
        r3_ttl.astype(int) + r4_entropy.astype(int)
    return df_feat


def summarize(name, df_flagged, has_labels=True):
    print(f"\n--- {name} ---")
    total = len(df_flagged)
    flagged = df_flagged["symbolic_flag"].sum()
    print(f"Flagged: {flagged}/{total} ({100*flagged/total:.2f}%)")
    for rule in ["r1_flood", "r2_scan", "r3_ttl_anomaly", "r4_high_entropy"]:
        n = df_flagged[rule].sum()
        print(f"  {rule}: {n} ({100*n/total:.2f}%)")

    if has_labels:
        is_attack = df_flagged[LABEL_COL] != BENIGN_LABEL
        # How many actual attacks did the rules catch? (recall)
        attack_total = is_attack.sum()
        attack_caught = (is_attack & df_flagged["symbolic_flag"]).sum()
        print(f"Attack recall (of {attack_total} true attacks): "
              f"{attack_caught}/{attack_total} "
              f"({100*attack_caught/attack_total if attack_total else 0:.2f}%)")
        # How many flags were actually benign? (false positive rate)
        is_benign = ~is_attack
        benign_total = is_benign.sum()
        benign_flagged = (is_benign & df_flagged["symbolic_flag"]).sum()
        print(f"Benign false-positive rate: {benign_flagged}/{benign_total} "
              f"({100*benign_flagged/benign_total if benign_total else 0:.2f}%)")
    else:
        # zero-day set is entirely attacks by construction
        attack_total = total
        attack_caught = df_flagged["symbolic_flag"].sum()
        print(f"Zero-day attack recall: {attack_caught}/{attack_total} "
              f"({100*attack_caught/attack_total:.2f}%)")


train_flagged = apply_rules(train_feat)
summarize("Train", train_flagged)

print("\nExtracting features for val...")
val_feat = compute_features(val_df)
val_flagged = apply_rules(val_feat)
summarize("Val", val_flagged)

print("\nExtracting features for test_known...")
test_known_feat = compute_features(test_known_df)
test_known_flagged = apply_rules(test_known_feat)
summarize("Test known", test_known_flagged)

print("\nExtracting features for test_zero_day...")
test_zeroday_feat = compute_features(test_zeroday_df)
test_zeroday_flagged = apply_rules(test_zeroday_feat)
summarize("Test zero-day", test_zeroday_flagged, has_labels=False)

# ----------------------------------------------------------------------
# Save -- only the derived columns + label, not the full byte matrix
# again (keeps these files small; join back to the original CSVs on
# row order if you need the byte columns alongside these flags later)
# ----------------------------------------------------------------------
keep_cols = ["orig_index", "nonzero_bytes", "entropy", "repetition_count", "repetition_fraction",
             "ttl", "protocol", "r1_flood", "r2_scan", "r3_ttl_anomaly", "r4_high_entropy",
             "symbolic_flag", "n_rules_fired", LABEL_COL]

train_flagged[keep_cols].to_csv(OUT_DIR / "train_kg.csv", index=False)
val_flagged[keep_cols].to_csv(OUT_DIR / "val_kg.csv", index=False)
test_known_flagged[keep_cols].to_csv(OUT_DIR / "test_known_kg.csv", index=False)
test_zeroday_flagged[keep_cols].to_csv(OUT_DIR / "test_zero_day_kg.csv", index=False)

print(f"\nSaved KG feature/flag files to {OUT_DIR}")
print("\nNext: fuse these symbolic_flag columns with your CNN/Hybrid-LTN "
      "predictions and re-evaluate the 5-view metric suite.")
