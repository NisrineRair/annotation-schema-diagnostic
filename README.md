# Annotation-Schema-Diagnostic

*Beyond Black-Box Labels: Interpretable Criteria for Diagnosing Subjective NLP Tasks*  (accepted at **ACL Findings 2026**).  

We propose a schema-level diagnostic for auditing expert-designed annotation schemas prior to gold-label commitment. The diagnostic separates two failure modes: **criterion instability** (criteria with hard-to-operationalize boundaries) and **criterion overlap** (criteria that systematically co-activate and blur category boundaries). Applied to Persuasive Value Extraction in B2B commercial documents, we show that disagreement is structured: instability concentrates in a few criteria, while nearly half of covered sentences activate multiple categories.

For full details, see the paper: [link to be added upon publication]

## Data

The annotation tensor and anonymized human annotations are released in this repository. Raw source documents are not released due to client confidentiality. All diagnostic procedures are fully specified and applicable to other datasets.

## Structure

- `data/anonymized/` — anonymized human annotations (`human_annotations_merged.csv`).
- `data/original/` — placeholder (not released — confidential).
- `data/tensors/` — LLM annotation tensor (`tensor_raw_4700.npy`, shape `4700 × 11 × 6`) and metadata (`mapping.json`).
- `notebooks/` — analysis notebooks reproducing all paper results.
  - `00_inspect_tensor.ipynb` — tensor explorer.
  - `01_instability_analysis.ipynb` — Table 1, Figure 2.
  - `02_overlap_analysis.ipynb` — Table 2, Figure 3.
  - `03_human_validation.ipynb` — Table 3.
  - `04_model_sensitivity.ipynb` — Appendix K.
  - `05_refinement_analysis.ipynb` — Appendix L.
- `scripts/` — data preparation and annotation pipeline.
  - `00_prepare_sentences.py` — Excel → anonymized CSV.
  - `01_anonymize_sentences.py` — anonymization + EN translation.
  - `02_generate_prompts.py` — per-criterion JSON prompts.
  - `03_run_llm_annotation.py` — LLM annotation via API.
  - `04_build_tensor.py` — aggregate responses → tensor.
  - `05_prepare_human_annotations.py` — merge expert annotation files.
- `requirements.txt` — Python dependencies.

## Citation

Citation will be added upon publication.

