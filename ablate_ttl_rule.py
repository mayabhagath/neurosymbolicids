"""
Ablation: how much of the symbolic layer's zero-day recall depends
specifically on the TTL anomaly rule (r3), which may be partly detecting
testbed artifacts (attacker vs. benign traffic generated from different VM
images with different default TTLs) rather than genuine attack behavior.

Recomputes symbolic_flag as (r1_flood OR r2_scan OR r4_high_entropy),
excluding r3_ttl_anomaly, using the already-saved KG feature files -- no
retraining or re-extraction needed.

Run:
    py ablate_ttl_rule.py
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

KG_DIR = Path(r"E:\NeuroSymbolic-IDS1\data\kg")
LABEL_COL = "label"
BENIGN_LABEL = "BENIGN"

test_known_kg = pd.read_csv(KG_DIR / "test_known_kg.csv")
test_zeroday_kg = pd.read_csv(KG_DIR / "test_zero_day_kg.csv")


def summarize(name, df, has_labels, flag_col):
    total = len(df)
    flagged = df[flag_col].sum()
    print(f"\n--- {name} ---")
    print(f"Flagged: {flagged}/{total} ({100*flagged/total:.2f}%)")
    if has_labels:
        is_attack = (df[LABEL_COL] != BENIGN_LABEL)
        attack_total = is_attack.sum()
        attack_caught = (is_attack & df[flag_col]).sum()
        recall = attack_caught / attack_total if attack_total else 0
        is_benign = ~is_attack
        benign_total = is_benign.sum()
        benign_flagged = (is_benign & df[flag_col]).sum()
        fpr = benign_flagged / benign_total if benign_total else 0
        print(f"Attack recall: {100*recall:.2f}%")
        print(f"Benign false-positive rate: {100*fpr:.2f}%")
    else:
        recall = flagged / total
        print(f"Zero-day recall: {100*recall:.2f}%")
    return recall


for df, name in [(test_known_kg, "test_known"), (test_zeroday_kg, "test_zero_day")]:
    df["symbolic_flag_no_ttl"] = df["r1_flood"] | df["r2_scan"] | df["r4_high_entropy"]

print("=" * 60)
print("WITH TTL rule (original)")
print("=" * 60)
summarize("Test known (with TTL)", test_known_kg, True, "symbolic_flag")
summarize("Test zero-day (with TTL)", test_zeroday_kg, False, "symbolic_flag")

print("\n" + "=" * 60)
print("WITHOUT TTL rule (r1_flood, r2_scan, r4_high_entropy only)")
print("=" * 60)
summarize("Test known (no TTL)", test_known_kg, True, "symbolic_flag_no_ttl")
summarize("Test zero-day (no TTL)", test_zeroday_kg, False, "symbolic_flag_no_ttl")

# Per-rule breakdown on zero-day, so it's clear what each rule alone contributes
print("\n" + "=" * 60)
print("Per-rule zero-day recall (each rule alone)")
print("=" * 60)
for rule in ["r1_flood", "r2_scan", "r3_ttl_anomaly", "r4_high_entropy"]:
    recall = test_zeroday_kg[rule].sum() / len(test_zeroday_kg)
    print(f"  {rule}: {100*recall:.2f}%")
