"""
01_anonymize_sentences.py
--------------------------
Reads the prepared sentences CSV (output of 00_prepare_sentences.py),
anonymizes and translates each sentence to English using GPT-4.1-mini:
  - Replaces people names, locations, companies, products with generic terms
  - Translates French (or any language) to English
  - Preserves sentence_id, client_uid, doc_uid columns unchanged

Input:
  - data/original/original_sentences.csv

Output:
  - data/anonymized/sentences_anonymized.csv   (safe to share)

Usage:
  python scripts/01_anonymize_sentences.py            # full run
  
Requirements:
  - OPENAI_API_KEY in .env file or environment variable
  - pip install openai pandas tqdm python-dotenv
"""


import pandas as pd
import openai
import os
from tqdm import tqdm
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV      = "data/original/original_sentences.csv"
OUTPUT_CSV     = "data/anonymized/sentences_anonymized.csv"
MODEL          = "gpt-4.1-mini"
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert editor specializing in data privacy and content anonymization.
Task: Rewrite the provided sentence to remove all specific identifiers while preserving the core meaning, tone, and intent.
Anonymization Rules:

1. People: Replace specific names with general roles (e.g., "John Doe" becomes "the client" or "the stakeholder").
2. Locations: Replace specific cities, offices, or addresses with general categories (e.g., "the London branch" becomes "the regional office").
3. Products/Brands: Replace specific brand or product names with generic descriptions (e.g., "iPhone 15" becomes "the mobile device").
4. Links/Digital Data: Replace URLs, emails, or file paths with placeholders like [URL], [Email], or [File Path].
5. Companies: Replace specific company names with "the organization" or "the vendor."
6. Language: Translate the output to English, regardless of the input language. Ensure the translation is natural and fluent.

Constraint: Ensure the resulting sentence remains grammatically correct and flows naturally.
Do not change the underlying message.
Return ONLY the rewritten sentence, no explanations."""


def anonymize_sentence(sentence: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sentence}
        ],
        temperature=0,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def main(test: bool = False):
    df = pd.read_csv(INPUT_CSV)

    if test:
        df = df.head(10)
        print("Running in TEST mode on 10 sentences...")

    anonymized_sentences = []
    for idx, sentence in enumerate(tqdm(df["sentence"], desc="Anonymizing")):
        try:
            anon = anonymize_sentence(sentence)
        except Exception as e:
            print(f"Error on sentence {idx}: {e}")
            anon = sentence  # fallback: keep original
        anonymized_sentences.append(anon)

    df["sentence"] = anonymized_sentences

    output_path = OUTPUT_CSV.replace(".csv", "_test.csv") if test else OUTPUT_CSV
    df.to_csv(output_path, index=False)
    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run on 10 sentences only")
    args = parser.parse_args()
    main(test=args.test)