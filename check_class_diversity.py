"""
Check payload diversity across all known attack classes -- extends the
earlier DDoS-specific diagnostic to every class, to see whether the
zero-day generalization gap is explained by low training diversity
beyond just DDoS.

Run:
    python check_class_diversity.py
"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path(r"E:\NeuroSymbolic-IDS1\data\CICIDS_converted_data.csv")

df = pd.read_csv(RAW_PATH)
byte_cols = [c for c in df.columns if "payload_byte" in c]
group_cols = byte_cols + ["ttl", "total_len", "protocol"]

known_attacks = [
    "DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest",
    "DoS slowloris", "FTP-Patator", "Infiltration", "SSH-Patator",
]

print(f"{'Class':<20} {'Total rows':>12} {'Unique combos':>15} {'% unique':>10} {'Top combo share':>18}")
print("-" * 80)

for cls in known_attacks:
    sub = df[df["label"] == cls]
    total = len(sub)
    group_sizes = sub.groupby(group_cols).size()
    n_unique = len(group_sizes)
    pct_unique = 100 * n_unique / total if total else 0
    top_share = 100 * group_sizes.max() / total if total else 0
    print(f"{cls:<20} {total:>12} {n_unique:>15} {pct_unique:>9.2f}% {top_share:>17.2f}%")
