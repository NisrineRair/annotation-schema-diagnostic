"""
05_prepare_human_annotations.py
--------------------------------
Reads 5 expert annotation files, keeps ALL rows including repeated
sentences (used for reliability analysis in the notebook).
Anonymizes expert names (E1-E5), removes identifying columns,
and outputs:
  - data/anonymized/human_annotations_merged.csv

Usage:
  python scripts/05_prepare_human_annotations.py
"""

import pandas as pd
from pathlib import Path
import numpy as np 

# ── Config ────────────────────────────────────────────────────────
INPUT_DIR  = Path("data/original/human_annotation")
OUTPUT_DIR = Path("data/anonymized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Only keep anonymized columns — drop sentence text and real IDs
KEEP_COLS = ["sentence_id", "client_uid", "doc_uid", "Label"]

# Drop these if they appear
DROP_COLS = [
    "sentence", "client_id_real", "source_file_real",
    "sector", "class_overlap_type", "n_classes", "activated_classes",
    "dimension_pattern", "n_dimensions", "activated_dimensions",
    "clarity_level", "avg_clarity_score", "complex_pattern",
    "Commentaire", "Unnamed: 18", "Unnamed: 19", "Unnamed: 20"
]

LABEL_MAP = {
    "Retour sur investissement": "c1",
    "Notoriété":                 "c2",
    "Obligation":                "c3",
    "Description":               "c0",
}

# ── Read all files — keep ALL rows ────────────────────────────────
files = [f for f in sorted(INPUT_DIR.glob("*.xlsx")) 
         if not f.name.startswith("~$")]
print(f"Found {len(files)} annotation files")

# In 05_prepare_human_annotations.py
# Change the merge to use LEFT join with deduplication first

dfs_clean = []
for i, file in enumerate(files, start=1):
    expert_id = f"E{i}"
    df = pd.read_excel(file)
    df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
    df = df.rename(columns={"Label": f"label_{expert_id}"})
    df[f"label_{expert_id}"] = df[f"label_{expert_id}"].map(LABEL_MAP)
    
    # Majority vote BEFORE merging
    df = df.groupby("sentence_id").agg(
        client_uid=("client_uid", "first"),
        doc_uid=("doc_uid", "first"),
        **{f"label_{expert_id}": (f"label_{expert_id}",
           lambda x: x.dropna().mode().iloc[0] 
           if len(x.dropna()) > 0 else np.nan)}
    ).reset_index()
    
    print(f"  {expert_id}: {len(df)} unique sentences")
    dfs_clean.append(df)

# Merge clean deduplicated files
merged = dfs_clean[0]
for i, df in enumerate(dfs_clean[1:], start=2):
    merged = merged.merge(
        df[["sentence_id", f"label_E{i}"]],
        on="sentence_id",
        how="inner"
    )

# ── Save ──────────────────────────────────────────────────────────
out_path = OUTPUT_DIR / "human_annotations_merged.csv"
merged.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Shape: {merged.shape}")
print(f"Unique sentence_ids: {merged['sentence_id'].nunique()}")
print(f"Columns: {list(merged.columns)}")
print(f"\nSample:")
print(merged.head())