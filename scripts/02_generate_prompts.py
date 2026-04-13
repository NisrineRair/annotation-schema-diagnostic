"""
02_generate_prompts.py
-----------------------
Generates LLM annotation prompts for each criterion and sentence pair.
For each criterion (q01–q09), produces a JSON file containing one prompt
per sentence, ready to be consumed by 03_run_llm_annotation.py.

Inputs:
  - data/anonymized/sentences_anonymized.csv
      Anonymized sentences with sentence_id, client_uid, doc_uid, sentence.
      Output of 01_anonymize_sentences.py.

  - prompts/contexte_analyse.txt
      The shared instruction template shown to the LLM before each question.
      Contains the B2B analysis context and Oui/Non response instructions
      (see Appendix E of the paper for the full prompt text).

  - prompts/liste_questions.txt
      The 9 annotation criteria (q01-q09) formatted as:
        [q01][ROI] Does the sentence mention financial gain or ROI?
        [q04][NOT] Does the sentence emphasize user well-being?
        [q07][OBL] Does the sentence mention regulatory compliance?
      Each criterion has an ID, a category tag (ROI/NOT/OBL/GEN),
      and optional [SPC] tag for criteria with a custom prompt format.

  - Categories map to the four PVE schema categories (see Section 4.1):
      ROI → c1: Performance & Efficiency (q01, q02, q03)
      NOT → c2: User Experience & Brand Value (q04, q05, q06)
      OBL → c3: Obligation & Safety (q07, q08, q09)
      GEN → c0: Non-Persuasive (default, no criterion triggered)

Outputs:
  - prompts/{q_id}_{category}.json
      One JSON file per criterion (e.g. q01_ROI.json), containing 4,699
      prompts — one per sentence. Each entry includes sentence_id,
      client_uid, doc_uid, question_id, category, and the full prompt text.

  - prompts/prompts_index.csv
      Index tracking all generated prompt files with creation date
      and context version. Used to skip already-generated criteria
      on re-runs.

Usage:
  python scripts/02_generate_prompts.py

Notes:
  - Skips criteria that already have a generated prompt file (incremental)
  - Special criteria marked with [SPC] use a custom prompt format
    instead of the shared contexte_analyse.txt template
  - Categories: ROI (Performance & Efficiency), NOT (User Experience
    & Brand Value), OBL (Obligation & Safety), GEN (General)

"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import hashlib
import sys
import re

# ========================
# PATHS AND DIRECTORIES
# ========================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "anonymized"
PROMPTS_DIR = ROOT_DIR / "prompts"
PROMPTS_DIR.mkdir(exist_ok=True)

sentences_path = DATA_DIR / "anonymized_sentences.csv"
context_path = PROMPTS_DIR / "contexte_analyse.txt"
questions_txt = PROMPTS_DIR / "liste_questions.txt"
index_path = PROMPTS_DIR / "prompts_index.csv"

# ========================
# LOAD BASE DATA
# ========================

if not sentences_path.exists():
    sys.exit(f"ERROR: {sentences_path} not found.")
sent_df = pd.read_csv(sentences_path)

if not context_path.exists():
    sys.exit(f"ERROR: {context_path} not found.")
CONTEXT_TEMPLATE = context_path.read_text(encoding="utf-8").strip()

if not questions_txt.exists():
    sys.exit(f"ERROR: {questions_txt} not found.")
# --- REMPLACEZ VOTRE BLOC DE LECTURE PAR CELUI-CI ---
raw_content = questions_txt.read_text(encoding="utf-8")
# On split, mais on ne garde QUE les blocs qui commencent réellement par '['
question_blocks = [b.strip() for b in re.split(r'\n(?=\[)', raw_content) if b.strip().startswith('[')]

q_data = []
for i, block in enumerate(question_blocks):
    # 1. Extraction ID Manuel (Priorité absolue)
    id_match = re.search(r'\[(q\d+[a-z]?)\]', block)
    if id_match:
        q_id = id_match.group(1)
        text_payload = block.replace(f"[{q_id}]", "").strip()
    else:
        q_id = f"q{i+1:02d}"
        text_payload = block
        print(f"⚠️ ATTENTION : Pas d'ID détecté. ID auto : {q_id}")

    # 2. Détection Catégorie
    cat = "GEN"
    if "[ROI]" in block: cat = "ROI"
    if "[NOT]" in block: cat = "NOT"
    if "[OBL]" in block: cat = "OBL"
    is_special = "[SPC]" in block

    # 3. Nettoyage strict (Purification)
    clean_text = re.sub(r'\[.*?\]', '', text_payload) 
    clean_text = re.sub(r'\(Ref:.*?\)', '', clean_text) 
    clean_text = clean_text.strip()

    # Sécurité q05a
    if "ATTRIBUE EXPLICITEMENT un statut" in clean_text:
        q_id = "q05a"

    # ON N'AJOUTE QUE SI LE TEXTE EST RÉELLEMENT UNE QUESTION
    if len(clean_text) > 5: 
        q_data.append({
            "question_id": q_id,
            "category": cat,
            "is_special": is_special,
            "question_text": clean_text
        })
# ---------------------------------------------------
q_df = pd.DataFrame(q_data)

# ========================
# LOAD OR CREATE INDEX
# ========================
if index_path.exists():
    index_df = pd.read_csv(index_path)
else:
    index_df = pd.DataFrame(columns=["question_id", "file_path", "n_sentences", "context_version", "date_created"])

# ========================
# GENERATE PROMPTS
# ========================

for _, q_row in q_df.iterrows():
    q_id = q_row["question_id"]
    cat = q_row["category"]
    q_text = q_row["question_text"]
    is_special = q_row["is_special"]
    out_path = PROMPTS_DIR / f"{q_id}_{cat}.json"

    if q_id in index_df["question_id"].values and out_path.exists():
        print(f"Skipping {q_id} ({cat}) — already generated.")
        continue

    print(f" Generating prompts for {q_id} ({cat})...")

    prompts = []
    if is_special:
        ctx_version, log_context = "v_custom_spc", "NONE_SPECIFIC_PROMPT"
    else:
        ctx_hash = hashlib.md5(CONTEXT_TEMPLATE.encode("utf-8")).hexdigest()[:6]
        ctx_version, log_context = f"v01_{ctx_hash}", CONTEXT_TEMPLATE

    for _, s_row in sent_df.iterrows():
        try:
            if is_special:
                full_prompt = q_text.format(sentence=s_row["sentence"])
            else:
                full_prompt = CONTEXT_TEMPLATE.format(
                    sentence=s_row["sentence"],
                    question_text=q_text
                )
        except KeyError as e:
            print(f"ERREUR Placeholder {{sentence}} manquant dans {q_id} : {e}")
            continue

        prompts.append({
            "sentence_id": int(s_row["sentence_id"]),
            "client_uid": s_row["client_uid"],
            "doc_uid": s_row["doc_uid"],
            "question_id": q_id,
            "category": cat,
            "context_version": ctx_version,
            "context": log_context,   
            "prompt": full_prompt      
        })

    if prompts:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)

        new_entry = pd.DataFrame([{
            "question_id": q_id,
            "file_path": str(out_path),
            "n_sentences": len(prompts),
            "context_version": ctx_version,
            "date_created": datetime.now().isoformat(timespec="seconds")
        }])
        index_df = pd.concat([index_df, new_entry], ignore_index=True)

index_df.drop_duplicates(subset="question_id", keep="last", inplace=True)
index_df.to_csv(index_path, index=False)
print(f"\nPrompt index updated at: {index_path}")
print("Done.")