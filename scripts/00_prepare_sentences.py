"""
00_prepare_sentences.py
-----------------------
Reads raw Excel file containing sentences from B2B commercial documents,
creates anonymized client/document IDs (client_uid, doc_uid), assigns
a stable sentence_id to each sentence, and saves:
  - data/original/original_sentences.csv        
  - data/original/client_map_private.csv 
  - data/original/doc_map_private.csv    
"""

import pandas as pd
from pathlib import Path

# ---------------- CONFIG ----------------
INPUT_XLSX   = "data/original/sentences_all_clients.xlsx"
OUT_DIR      = Path("data/anonymized")
PRIVATE_DIR  = Path("data/original")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

SENTENCE_COL = "Sentences"
CLIENT_COL   = "client_id"
SOURCE_COL   = "source_file"
# ----------------------------------------

# Load Excel
df = pd.read_excel(INPUT_XLSX)

# Verify expected columns
for col in [SENTENCE_COL, CLIENT_COL, SOURCE_COL]:
    if col not in df.columns:
        raise ValueError(f"Missing expected column: {col}")

# Clean whitespace and drop empties
df = df[[SENTENCE_COL, CLIENT_COL, SOURCE_COL]].dropna(subset=[SENTENCE_COL])
df[SENTENCE_COL] = df[SENTENCE_COL].astype(str).str.strip()
df = df[df[SENTENCE_COL].str.len() > 0]

# Create stable anonymized IDs
clients = sorted(df[CLIENT_COL].astype(str).unique())
client_map = {c: f"C{i:03d}" for i, c in enumerate(clients, start=1)}

doc_rows = []
doc_map = {}
for client in clients:
    docs = sorted(df.loc[df[CLIENT_COL] == client, SOURCE_COL].fillna("").astype(str).unique())
    for j, doc in enumerate(docs, start=1):
        doc_uid = f"D{j:02d}"
        doc_map[(client, doc)] = doc_uid
        doc_rows.append({
            "client_id_real": client,
            "client_uid": client_map[client],
            "source_file_real": doc,
            "doc_uid": doc_uid
        })

# Replace names with IDs
df["client_uid"] = df[CLIENT_COL].map(client_map)
df["doc_uid"] = df.apply(
    lambda r: doc_map.get((str(r[CLIENT_COL]), str(r[SOURCE_COL])), "D00"), axis=1
)

# Add sentence_id
df = df.reset_index(drop=False).rename(columns={"index": "sentence_id"})
df["sentence_id"] = df["sentence_id"] + 1

# Keep only clean columns
df_clean = df[["sentence_id", "client_uid", "doc_uid", SENTENCE_COL]].rename(
    columns={SENTENCE_COL: "sentence"}
)

# Save base file (safe to share)
base_path = OUT_DIR / "sentences.csv"
df_clean.to_csv(base_path, index=False, encoding="utf-8")
print(f"Base file saved to {base_path} ({len(df_clean)} sentences)")

# Save mappings (private, do NOT push to GitHub)
pd.DataFrame(list(client_map.items()), columns=["client_id_real", "client_uid"])\
  .to_csv(PRIVATE_DIR / "client_map_private.csv", index=False, encoding="utf-8")

pd.DataFrame(doc_rows).to_csv(
    PRIVATE_DIR / "doc_map_private.csv", index=False, encoding="utf-8"
)

print(f"Private mappings saved in {PRIVATE_DIR} — do NOT push these to GitHub")