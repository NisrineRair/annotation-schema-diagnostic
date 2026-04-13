"""
03_run_llm_annotation.py
-------------------------
Runs LLM annotation for each criterion and sentence pair using the
generated prompts (output of 02_generate_prompts.py).
For each model and criterion, sends prompts to the LLM API and saves
binary Oui/Non responses with logprobs and cost tracking.

Inputs:
  - prompts/{q_id}_{category}.json     (output of 02_generate_prompts.py)
  - data/pricing.json                  (pricing per model in USD per million tokens)

Outputs (saved under runs/model_{model_name}/run_{timestamp}_s{seed}/):
  - raw_responses/{q_id}.jsonl         (one JSONL file per criterion with
                                        raw responses, logprobs, cost, timestamp)
  - analysis/reasoning_vectors.csv     (pivoted table: one row per sentence,
                                        one column per criterion with Oui/Non)
  - analysis/detailed_logprobs.csv     (full response table with logprobs)
  - metadata.json                      (run summary: model, cost, seed, errors)
  - progress.json                      (tracks completed criteria for resuming)
  - run.log                            (execution log)

Usage:
  python scripts/03_run_llm_annotation.py

Key Configuration (edit at top of script):
  - MODELS         : list of models to run (e.g. ["gpt-4.1-mini", "gpt-4.1"])
  - API            : "openai" or "openrouter"
  - SPECIFIC_QUESTION : set to a criterion ID (e.g. "q01") to run one criterion
                        only, or None to run all
  - NEW_RUN        : True = start fresh run, False = resume last run
  - SEED           : random seed for reproducibility (default: 2)
  - MAX_WORKERS    : number of parallel threads (default: 8)

Notes:
  - Supports both OpenAI and OpenRouter APIs
  - Crash-safe: saves each response immediately to JSONL
  - Resumes interrupted runs automatically (skips already completed criteria)
  - Deduplicates responses by (sentence_id, question_id) keeping last occurrence
  - Extracts logprobs for "Oui" and "Non" tokens for uncertainty analysis

Requirements:
  - OPENAI_API_KEY and/or OPENROUTER_API_KEY in .env file
  - pip install openai pandas tqdm python-dotenv
"""


import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, APIError, RateLimitError, Timeout
import pandas as pd
from tqdm import tqdm
import math

# === Load environment variables ===
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

# === CONFIGURATION ===
MODELS = ["gpt-4.1-mini"]  # add more later openai/gpt-4.1  openrouter/qwen/qwen-2.5-72b-instruct"
API = "openai"  # "openai" or "openrouter" change depending on the chosen API
MAX_WORKERS = 8
PROMPTS_DIR = Path("prompts")
PRICING_FILE = Path("data/pricing.json")
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True, parents=True)

# --- ADD THIS LINE ---
# Set to None to run all questions, or "q05a" to run only that specific question
SPECIFIC_QUESTION = "q11" #For specific adding questions 

# === RUN CONTROL FLAGS ===
NEW_RUN = False  # False = continue last run if exists; True = start fresh
RUN_LABEL = ""  # optional tag; leave "" if not needed
SEED = 2  # reproducibility

# === Load pricing ===
with open(PRICING_FILE, "r", encoding="utf-8") as f:
    pricing = json.load(f)

# === Initialize client ===
if API == "openai":
    client = OpenAI(api_key=OPENAI_KEY)
elif API == "openrouter":
    client = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1")
else:
    raise ValueError("Unsupported API")

# === Helper: estimate cost ===
def estimate_cost(model_name, input_tokens, output_tokens):
    normalized = model_name.lower().strip().replace(":", "-").replace("_", "-")
    found_key, pricing_data = None, None
    for key, value in pricing.items():
        key_norm = key.lower().strip().replace(":", "-").replace("_", "-")
        if normalized.startswith(key_norm) or key_norm in normalized:
            found_key, pricing_data = key, value
            break

    if not pricing_data:
        logging.warning(f"No pricing found for model: {model_name}")
        return None

    input_cost = (input_tokens / 1_000_000) * pricing_data["input_per_mtok_usd"]
    output_cost = (output_tokens / 1_000_000) * pricing_data["output_per_mtok_usd"]
    return round(input_cost + output_cost, 8)


# === Core prompt runner ===
def run_prompt(prompt_obj, model_full_name, client, output_file=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_full_name.replace("openai/", "").replace("openrouter/", ""),
                messages=[{"role": "user", "content": prompt_obj["prompt"]}],
                temperature=0.0,
                max_tokens=3,
                logprobs=True,
                top_logprobs=20,
            )

            message = response.choices[0].message.content.strip()
            in_t = response.usage.prompt_tokens
            out_t = response.usage.completion_tokens

            # === FIXED logprob extraction ===
            try:
                first_token_data = response.choices[0].logprobs.content[0]
                logprob_response = first_token_data.logprob  # This is always correct!
                
                logprob_oui = None
                logprob_non = None
                
                # Find the highest probability for each option
                for alt in first_token_data.top_logprobs:
                    token_clean = alt.token.strip().lower()
                    
                    if token_clean == "oui":
                        # Take the highest logprob for "oui"
                        if logprob_oui is None or alt.logprob > logprob_oui:
                            logprob_oui = alt.logprob
                            
                    elif token_clean == "non":
                        # Take the highest logprob for "non"  
                        if logprob_non is None or alt.logprob > logprob_non:
                            logprob_non = alt.logprob
                            
            except Exception as e:
                logging.warning(f"Logprob extraction failed: {e}")
                logprob_response = logprob_oui = logprob_non = None


            # === Build result record ===
            result = {
                "sentence_id": prompt_obj["sentence_id"],
                "client_uid": prompt_obj["client_uid"],
                "doc_uid": prompt_obj["doc_uid"],
                "question_id": prompt_obj["question_id"],
                "category": prompt_obj["category"],
                "response": message,
                "logprob_response": logprob_response,
                "logprob_oui": logprob_oui,
                "logprob_non": logprob_non,
                "tokens_prompt": in_t,
                "tokens_completion": out_t,
                "cost_usd": estimate_cost(model_full_name, in_t, out_t),
                "seed": SEED,
                "model": model_full_name,
                "timestamp": datetime.now().isoformat(),
            }

            #  Save immediately (crash safety)
            if output_file:
                with open(output_file, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            return result

        except (APIError, RateLimitError, Timeout) as e:
            wait = 2 ** attempt
            logging.warning(
                f"Retry {attempt + 1}/{max_retries} after error for sentence {prompt_obj.get('sentence_id')}: {e}"
            )
            time.sleep(wait)

        except Exception as e:
            logging.error(f"Unrecoverable error on sentence {prompt_obj.get('sentence_id')}: {e}")
            err = {
                "error": str(e),
                "sentence_id": prompt_obj.get("sentence_id"),
                "question_id": prompt_obj.get("question_id"),
                "model": model_full_name,
                "seed": SEED,
                "timestamp": datetime.now().isoformat(),
            }

            if output_file:
                with open(output_file, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(err, ensure_ascii=False) + "\n")

            return err

    # === All retries failed ===
    err = {
        "error": f"Failed after {max_retries} retries",
        "sentence_id": prompt_obj.get("sentence_id"),
        "question_id": prompt_obj.get("question_id"),
        "model": model_full_name,
        "seed": SEED,
        "timestamp": datetime.now().isoformat(),
    }

    if output_file:
        with open(output_file, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(err, ensure_ascii=False) + "\n")

    return err


# === MAIN LOOP ===
for model in MODELS:
    model_safe_name = model.replace("/", "-")
    model_base_dir = RUNS_DIR / f"model_{model_safe_name}"
    model_base_dir.mkdir(exist_ok=True, parents=True)

    run_pattern = f"run_*_s{SEED}" + (f"_{RUN_LABEL}" if RUN_LABEL else "")
    existing_runs = sorted(model_base_dir.glob(run_pattern), reverse=True)

    if not NEW_RUN and existing_runs:
        model_run_dir = existing_runs[0]
        print(f"Resuming existing run: {model_run_dir.name}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_id = f"run_{timestamp}_s{SEED}" + (f"_{RUN_LABEL}" if RUN_LABEL else "")
        model_run_dir = model_base_dir / run_id
        model_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Starting new run: {model_run_dir.name}")

    log_file = model_run_dir / "run.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    model_full_name = f"{API}/{model}" if not model.startswith(API + "/") else model

    # --- UPDATED LOGIC HERE ---
    if SPECIFIC_QUESTION:
        # Searches for files starting with your specific ID (e.g., q05a*.json)
        prompt_files = sorted(PROMPTS_DIR.glob(f"{SPECIFIC_QUESTION}*.json"))
        if not prompt_files:
            print(f"⚠️ Warning: No files found for question ID: {SPECIFIC_QUESTION}")
    else:
        # Default behavior: run all
        prompt_files = sorted(PROMPTS_DIR.glob("q*.json"))
    
    if not prompt_files:
        print("No prompt files found. Skipping model.")
        continue
    # --------------------------

    print(f"\n=== Running {model_full_name} on {len(prompt_files)} question files (Seed: {SEED}) ===")

    all_results = []

    progress_file = model_run_dir / "progress.json"
    if not progress_file.exists():
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump({"completed_questions": []}, f, indent=2)

    for file_path in prompt_files:
        question_id = file_path.stem.split("_")[0]
        with open(progress_file, "r", encoding="utf-8") as f:
            progress = json.load(f)

        if question_id in progress["completed_questions"] and not NEW_RUN:
            print(f" Skipping already completed {question_id}")
            continue

        print(f"\n Processing {question_id} ({file_path.name})")
        with open(file_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        raw_output_dir = model_run_dir / "raw_responses"
        raw_output_dir.mkdir(exist_ok=True, parents=True)
        output_file = raw_output_dir / f"{question_id}.jsonl"

        existing_records = {}
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        sid = record.get("sentence_id")
                        if sid and "error" not in record:
                            existing_records[sid] = record
                    except json.JSONDecodeError:
                        continue

        existing_ids = set(existing_records.keys())
        pending_prompts = [p for p in prompts if p["sentence_id"] not in existing_ids]
        print(f"   {len(pending_prompts)} remaining sentences to process.")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(run_prompt, p, model_full_name, client, output_file) for p in pending_prompts]
            for future in tqdm(as_completed(futures), total=len(futures), desc=question_id):
                result = future.result()
                all_results.append(result)

        progress["completed_questions"].append(question_id)
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

   
    # === MERGE ALL PARTIAL JSONL FILES SAFELY ===
    raw_output_dir = model_run_dir / "raw_responses"
    analysis_dir = model_run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True, parents=True)

    all_jsonl = sorted(raw_output_dir.glob("q*.jsonl"))
    records = []

    for file in all_jsonl:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "error" not in rec:
                        records.append(rec)
                except json.JSONDecodeError:
                    continue

    if not records:
        print(" No valid records found; skipping aggregation.")
        continue

    df = pd.DataFrame(records)
    df = df.sort_values(by=["client_uid", "doc_uid", "sentence_id", "question_id"]).reset_index(drop=True)

    # === Deduplicate by (sentence_id, question_id) keeping last occurrence ===
    df = df.drop_duplicates(subset=["sentence_id", "question_id"], keep="last")

    # === Pivot to reasoning vectors ===
    agg = df.pivot_table(
        index=["sentence_id", "client_uid", "doc_uid"],
        columns="question_id",
        values="response",
        aggfunc="first"
    ).reset_index()

    agg = agg.sort_values(by=["client_uid", "doc_uid", "sentence_id"]).reset_index(drop=True)

    # Save consistent merged outputs
    agg.to_csv(analysis_dir / "reasoning_vectors.csv", index=False)
    df.to_csv(analysis_dir / "detailed_logprobs.csv", index=False)

    print(f" Aggregated {len(df)} individual responses into reasoning_vectors.csv")


    total_cost = round(sum(r.get("cost_usd", 0) for r in all_results if r.get("cost_usd")), 6)
    meta = {
        "run_id": model_run_dir.name,
        "model": model_full_name,
        "api": API,
        "seed": SEED,
        "temperature": 0.0,
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(prompt_files),
        "num_sentences": df["sentence_id"].nunique() if not df.empty else 0,
        "total_cost_usd": total_cost,
        "num_errors": len([r for r in all_results if "error" in r]),
        "new_run": NEW_RUN,
        "run_label": RUN_LABEL or None,
        "max_workers": MAX_WORKERS,
        "prompts_dir": str(PROMPTS_DIR),
    }

    with open(model_run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n Done for {model_full_name}")
    print(f"   Results saved in: {model_run_dir}")
    print(f"   Estimated total cost: ${total_cost}")
    print(f"   Seed: {SEED}")
