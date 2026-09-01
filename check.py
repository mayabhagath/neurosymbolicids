import pandas as pd

df = pd.read_csv(r"E:\NeuroSymbolic-IDS1\data\CICIDS_converted_data.csv")
byte_cols = [c for c in df.columns if "payload_byte" in c]

ddos = df[df['label'] == 'DDoS']
print("Total DDoS rows:", len(ddos))
print("Unique DDoS rows (full dedup):", len(ddos.drop_duplicates()))

# non-zero payload byte distribution
nonzero_counts = (ddos[byte_cols] != 0).sum(axis=1)
print("\nDistribution of non-zero payload bytes per DDoS packet:")
print(nonzero_counts.describe())
print("Packets with <10 non-zero bytes:", (nonzero_counts < 10).sum())
print("Packets with 0 non-zero bytes (fully empty payload):", (nonzero_counts == 0).sum())

# cluster sizes WITHOUT printing the group keys themselves
dup_group_sizes = ddos.groupby(byte_cols + ['ttl', 'total_len', 'protocol']).size()
print("\nNumber of distinct duplicate groups:", len(dup_group_sizes))
print("Distribution of group sizes (i.e. how many times each unique packet repeats):")
print(dup_group_sizes.describe())
print("\nTop 10 largest group sizes (just the counts, not the packets):")
print(dup_group_sizes.sort_values(ascending=False).head(10).values)