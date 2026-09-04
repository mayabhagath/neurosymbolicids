"""
Preprocessing script: replicate the base paper's (Bizzarri et al., ICCCN 2024)
dataset preparation from the raw Payload-Byte CICIDS2017 CSV.

Steps:
    1. Load raw data (tagging each row with its original raw-file position,
       orig_index, so causal/order-dependent features can be computed later
       and rejoined after this script's sampling destroys row order)
    2. Remove rows with no payload data (all-zero payload) — NOT full-row
       dedup, since this feature set lacks IP/port/timestamp and treating
       repeated flood packets as "duplicates" would destroy real attack
       signal (see note in step 2 below)
    3. Split off the 6 zero-day classes (held out entirely from train/val)
    4. Undersample the 8 known attack classes to match the smallest one
    5. Undersample BENIGN to 200,000
    6. Stratified 80/10/10 split on the known-class set
    7. Save all resulting splits to disk

Run:
    python preprocess_cicids.py
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Config — adjust paths as needed
# ----------------------------------------------------------------------
INPUT_PATH = Path(r"E:\NeuroSymbolic-IDS1\data\CICIDS_converted_data.csv")  # shared, read-only
OUTPUT_DIR = Path(r"E:\NeuroSymbolic-IDS1\causal_kg\data\processed")  # NEW location
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label"
BENIGN_LABEL = "BENIGN"
BENIGN_TARGET = 200_000

# Classes held out entirely as zero-day (never seen in train/val)
ZERO_DAY_CLASSES = [
    "Heartbleed",
    "Web Attack – Brute Force",
    "Web Attack – XSS",
    "Bot",
    "PortScan",
    "Web Attack – Sql Injection",
]

RANDOM_SEED = 42

# ----------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------
print(f"Loading {INPUT_PATH} ...")

# Read just the header first to build a memory-efficient dtype map.
# Payload bytes are 0-255 -> uint8 (1 byte) instead of pandas' default
# int64 (8 bytes) -- an 8x memory reduction across 1500 columns, which is
# what caused the previous MemoryError on a 1.4M-row file.
header_cols = pd.read_csv(INPUT_PATH, nrows=0).columns.tolist()
dtype_map = {}
for c in header_cols:
    if "payload_byte" in c:
        dtype_map[c] = "uint8"
    elif c == "ttl":
        dtype_map[c] = "uint8"
    elif c == "protocol":
        dtype_map[c] = "category"  # string values like "tcp"/"udp", not numeric
    elif c == "total_len":
        dtype_map[c] = "uint16"
    elif c == "t_delta":
        dtype_map[c] = "float32"
    # label column left as default (object/string)

df = pd.read_csv(INPUT_PATH, dtype=dtype_map)
print(f"Raw shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# Capture each row's position in the RAW file before any filtering/sampling.
# This is what lets a later causal (windowed, order-respecting) feature --
# computed once on the full raw file in its original order -- be rejoined
# back onto these splits after undersampling/shuffling destroys row order.
df["orig_index"] = df.index

byte_cols = [c for c in df.columns if "payload_byte" in c]
print(f"Found {len(byte_cols)} payload byte columns")

# Normalize label whitespace/dash inconsistencies just in case
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()

print("\nRaw class counts:")
print(df[LABEL_COL].value_counts())

# ----------------------------------------------------------------------
# 2. Remove empty-payload rows only.
#
# NOTE: We deliberately do NOT do a full-row drop_duplicates() here.
# This feature set (payload bytes + ttl/total_len/protocol/t_delta) lacks
# IP/port/timestamp, so "duplicate rows" in this reduced representation
# are often genuine repeated packets from flood-style attacks (e.g. DDoS),
# not erroneous duplicate records. Diagnostic: DDoS collapsed from 241,405
# rows to 33 unique combinations, with one template repeating 241,138
# times -- that's the attack signature (volume), not noise. Deduping here
# would silently destroy the class's defining characteristic. The base
# paper's dedup step almost certainly ran upstream on richer
# CICFlowMeter/PCAP metadata (IP/port/timestamp) that isn't present in
# this stripped feature CSV, so it isn't reproducible -- or appropriate
# -- at this stage.
# ----------------------------------------------------------------------
before = len(df)
empty_payload_mask = (df[byte_cols] == 0).all(axis=1)
n_empty = empty_payload_mask.sum()
df = df[~empty_payload_mask]
print(f"Dropped {n_empty} rows with all-zero payload (before={before}) -> {len(df)} remaining")

print("\nClass counts after empty-payload removal:")
counts_after_dedup = df[LABEL_COL].value_counts()
print(counts_after_dedup)

# ----------------------------------------------------------------------
# 3. Split off zero-day classes
# ----------------------------------------------------------------------
missing_zd = [c for c in ZERO_DAY_CLASSES if c not in counts_after_dedup.index]
if missing_zd:
    raise ValueError(
        f"These zero-day classes were not found in the label column: {missing_zd}\n"
        f"Available labels: {sorted(counts_after_dedup.index.tolist())}"
    )

zero_day_df = df[df[LABEL_COL].isin(ZERO_DAY_CLASSES)]
known_df = df[~df[LABEL_COL].isin(ZERO_DAY_CLASSES)]
del df  # free the full raw frame now that it's been split

print(f"\nZero-day pool: {len(zero_day_df)} rows across {len(ZERO_DAY_CLASSES)} classes")
print(zero_day_df[LABEL_COL].value_counts())

print(f"\nKnown-class pool (before undersampling): {len(known_df)} rows")
print(known_df[LABEL_COL].value_counts())

# ----------------------------------------------------------------------
# 4. Undersample known attack classes to match the smallest one
# ----------------------------------------------------------------------
known_attack_df = known_df[known_df[LABEL_COL] != BENIGN_LABEL]
benign_df = known_df[known_df[LABEL_COL] == BENIGN_LABEL]
del known_df  # free now that it's split into attack/benign

class_counts = known_attack_df[LABEL_COL].value_counts()
min_class = class_counts.idxmin()
min_count = class_counts.min()
print(f"\nSmallest known-attack class: {min_class} ({min_count} samples)")
print("(Base paper's floor was FTP-Patator = 31,843 -- comparing against that "
      "tells you whether your pool is large enough to match their setup.)")

balanced_attacks = []
for cls in class_counts.index:
    cls_df = known_attack_df[known_attack_df[LABEL_COL] == cls]
    sampled = cls_df.sample(n=min_count, random_state=RANDOM_SEED)
    balanced_attacks.append(sampled)
balanced_attack_df = pd.concat(balanced_attacks, ignore_index=True)

print(f"\nBalanced known-attack set: {len(balanced_attack_df)} rows "
      f"({len(class_counts)} classes x {min_count})")

# ----------------------------------------------------------------------
# 5. Undersample BENIGN
# ----------------------------------------------------------------------
if len(benign_df) < BENIGN_TARGET:
    raise ValueError(
        f"Only {len(benign_df)} BENIGN samples available after dedup, "
        f"need {BENIGN_TARGET}. Lower BENIGN_TARGET or investigate the dedup drop."
    )
benign_sampled = benign_df.sample(n=BENIGN_TARGET, random_state=RANDOM_SEED)
print(f"Sampled BENIGN down to {len(benign_sampled)} rows")

# ----------------------------------------------------------------------
# 6. Combine and stratified split
# ----------------------------------------------------------------------
final_known_df = pd.concat([benign_sampled, balanced_attack_df], ignore_index=True)
print(f"\nFinal known-class dataset: {len(final_known_df)} rows")
print(final_known_df[LABEL_COL].value_counts())

train_df, temp_df = train_test_split(
    final_known_df,
    test_size=0.2,
    stratify=final_known_df[LABEL_COL],
    random_state=RANDOM_SEED,
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df[LABEL_COL],
    random_state=RANDOM_SEED,
)

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ----------------------------------------------------------------------
# 7. Save
# ----------------------------------------------------------------------
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test_known.csv", index=False)
zero_day_df.to_csv(OUTPUT_DIR / "test_zero_day.csv", index=False)

print(f"\nSaved all splits to {OUTPUT_DIR}")
print("  train.csv, val.csv, test_known.csv, test_zero_day.csv")

# For the "all 15 classes" evaluation views the paper reports, you can simply
# concatenate test_known.csv + test_zero_day.csv at evaluation time — keep
# them as separate files so you always know which rows are true zero-day.
