"""
04_build_tensor.py
-------------------
Aggregates all LLM annotation responses (output of 03_run_llm_annotation.py)
into a single 3D binary tensor of shape (N_sentences, N_criteria, N_models).

For each (sentence, criterion, model) triple, the tensor contains:
  - 1  if the model answered "Oui" (criterion present)
  - 0  if the model answered "Non" (criterion absent)
  - -1 if the response is missing or could not be parsed

This tensor is the core input for the schema diagnostic (stability and
overlap analysis) described in Section 3 of the paper.

Inputs:
  - data/anonymized/sentences_anonymized.csv   (sentence metadata)
  - prompts/{q_id}_{category}.json             (prompt texts for metadata)
  - runs/model_{model_name}/**/q*.jsonl        (raw LLM responses, all runs)

Outputs (saved under data/tensors/run_{timestamp}_FULL_AGGREGATED/):
  - tensor_raw_4700.npy       (3D binary tensor, shape: 4700 x 11 x 6)
  - mapping_sentences.csv     (maps tensor row index to sentence_id, client_uid,
                               doc_uid, sentence text)
  - mapping.json              (metadata: criteria, models, prompt texts,
                               valid/missing entry counts)

Usage:
  python scripts/04_build_tensor.py

Notes:
  - Scans all runs for each model and merges responses (last write wins
    for duplicates)
  - Criteria q01-q09 correspond to the 9 PVE criteria in the paper
  - q10, q11 are refinement criteria from the iterative schema revision
  - Categories: ROI → c1, NOT → c2, OBL → c3 

"""

import numpy as np
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# =================================================================
# 1. CONFIGURATION
# =================================================================
SENTENCES_CSV  = Path("data/anonymized/sentences_anonymized.csv")
PROMPTS_DIR    = Path("prompts")
RUNS_DIR       = Path("runs")
BASE_OUTPUT_DIR = Path("data/tensors")

# q10 est l'ID article pour vos fichiers nommés q05a
ORDERED_IDS = ["q01", "q02", "q03", "q04", "q05", "q06", "q07", "q08", "q09", "q10", "q11"]
ID_TO_IDX = {q_id: i for i, q_id in enumerate(ORDERED_IDS)}

CATEGORIES = {
    "q01": "ROI", "q02": "ROI", "q03": "ROI", "q04": "NOT", "q05": "NOT", 
    "q06": "NOT", "q07": "OBL", "q08": "OBL", "q09": "OBL", "q10": "NOT_V2_Q05a", "q11": "NOT_V2_Q5b_Q6"
}

MODELS_LIST = [
    "model_google-gemini-2.0-flash-001", "model_gpt-4.1-mini", "model_openai-gpt-4.1",
    "model_openrouter-meta-llama-llama-3.3-70b-instruct", "model_openrouter-mistralai-mistral-large-2411",
    "model_openrouter-qwen-qwen-2.5-72b-instruct"
]

N_PHRASES = 4700

# =================================================================
# 2. GÉNÉRATION DU MAPPING DES PHRASES (CSV)
# =================================================================
def generate_sentence_mapping(output_path):
    print(f"Lecture du texte depuis {SENTENCES_CSV.name}...")
    try:
        df_source = pd.read_csv(SENTENCES_CSV)
        df_source = df_source[df_source['sentence_id'] <= N_PHRASES].copy()
        df_source['tensor_idx'] = df_source['sentence_id'] - 1
        df_final = df_source[['tensor_idx', 'sentence_id', 'client_uid', 'doc_uid', 'sentence']]
        df_final.to_csv(output_path, index=False, encoding="utf-8")
        print(f"mapping_sentences.csv généré ({len(df_final)} lignes).")
    except Exception as e:
        print(f"Erreur mapping : {e}")

# =================================================================
# 3. LOGIQUE TENSEUR & MÉTADONNÉES
# =================================================================
def get_prompts_from_json_source():
    prompts_text = {}
    mapping_files = {"q10": "q05a"} 
    for q_id in ORDERED_IDS:
        search_id = mapping_files.get(q_id, q_id)
        files = list(PROMPTS_DIR.glob(f"{search_id}*.json"))
        if files:
            with open(files[0], "r", encoding="utf-8") as f:
                data = json.load(f)
                prompts_text[q_id] = data[0].get("prompt", "Texte absent")
    return prompts_text

def normalize_response(resp):
    if resp is None: return None
    r = str(resp).strip().lower()
    oui_variants = {
        "oui", "oui.", "o", "o.", "1", "1.0", "true",
        "**oui**", "**oui", "oui**", "*oui*", "o/1","OUI"
    }
    
    non_variants = {
        "non", "non.", "n", "n.", "0", "0.0", "false",
        "**non**", "**non", "non**", "*non*","NON"
    }
    if r in oui_variants: return 1
    if r in non_variants: return 0
    return None




def generate_full_tensor():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = BASE_OUTPUT_DIR / f"run_{timestamp}_FULL_AGGREGATED"
    output_dir.mkdir(exist_ok=True, parents=True)

    generate_sentence_mapping(output_dir / "mapping_sentences.csv")
    source_texts = get_prompts_from_json_source()
    
    # Initialisation à -1 (Donnée manquante)
    tensor = np.full((N_PHRASES, len(ORDERED_IDS), len(MODELS_LIST)), -1, dtype=int)

    for m_idx, m_folder in enumerate(MODELS_LIST):
        model_path = RUNS_DIR / m_folder
        if not model_path.exists(): continue
        
        print(f"Analyse du modèle : {m_folder}")
        
        # On scanne TOUTE l'arborescence du modèle pour fusionner les différents runs
        all_files = list(model_path.rglob("q*.jsonl"))
        
        for file in all_files:
            q_file_id = file.stem.lower()
            if q_file_id == "q05a": q_file_id = "q10" # Mapping forcé pour le tenseur
            
            if q_file_id not in ID_TO_IDX: continue
            q_idx = ID_TO_IDX[q_file_id]

            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "error" in data: continue
                        s_idx = int(data["sentence_id"]) - 1
                        
                        if 0 <= s_idx < N_PHRASES:
                            val = normalize_response(data.get("response"))
                            if val is not None:
                                # On remplit la cellule
                                tensor[s_idx, q_idx, m_idx] = val
                    except: continue

    # Sauvegarde finale
    np.save(output_dir / "tensor_raw_4700.npy", tensor)
    
    metadata = {
        "date": timestamp,
        "questions_registry": {
            q_id: {"category": CATEGORIES[q_id], "prompt": source_texts[q_id]} for q_id in ORDERED_IDS
        },
        "models": MODELS_LIST,
        "summary": {
            "total_valid_entries": int(np.sum(tensor != -1)),
            "total_missing_entries": int(np.sum(tensor == -1))
        }
    }
    with open(output_dir / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"\n Agrégation terminée dans : {output_dir}")
    print(f" Entrées valides : {metadata['summary']['total_valid_entries']}")
    print(f" Entrées manquantes (-1) : {metadata['summary']['total_missing_entries']}")

if __name__ == "__main__":
    generate_full_tensor()