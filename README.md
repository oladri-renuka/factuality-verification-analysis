# Atomic Factuality Verification Analysis

> Comparing NLI, Retrieval+LLM, and Direct LLM approaches for verifying atomic facts in LLM-generated biographies — with a deployed live demo.

**Oladri Renuka · MS Applied ML · University of Maryland**

[![Live Demo](https://img.shields.io/badge/🔍_Live_Demo-HuggingFace_Spaces-blue)](https://oladri-renuka-factuality-verifier.hf.space)
[![Dataset](https://img.shields.io/badge/Dataset-FActScore-green)](https://huggingface.co/datasets/felixlyu/factscore)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)](https://python.org)

---

## What This Project Does

LLMs hallucinate. When they generate biographies, some facts are accurate and some are made up. This project systematically compares three strategies for automatically detecting which facts are supported by Wikipedia and which are not — and finds that **how you calibrate your verifier matters more than which verifier you choose**.

**Try it live →** [oladri-renuka-factuality-verifier.hf.space](https://oladri-renuka-factuality-verifier.hf.space)
Paste any biography, enter a person's name, and see each sentence verified against Wikipedia in real time.

---

## Key Finding

> Lowering the NLI entailment threshold from **0.50 → 0.10** improves F1 by **+0.076** — a larger gain than switching from NLI to a 3B-parameter LLM verifier entirely.

This is a calibration problem, not a model problem. RoBERTa-Large-MNLI assigns low entailment probability even to facts it implicitly recognizes as true. The default threshold of 0.50 systematically rejects supported facts.

---

## Results

### Table 1: Overall Performance (14,525 facts · 183 entities)

| Method | F1 | Precision | Recall | Kappa | Accuracy |
|--------|:---:|:---------:|:------:|:-----:|:--------:|
| A: NLI · threshold=0.50 (default) | 0.651 | 0.953 | 0.495 | 0.335 | 0.743 |
| **A: NLI · threshold=0.10 (calibrated) ★** | **0.727** | **0.919** | **0.602** | **0.393** | **0.789** |
| B: Retrieval+LLM (BM25 + Qwen2.5-3B) | 0.667 | 0.957 | 0.512 | 0.354 | 0.756 |
| C: Direct LLM · no retrieval (Qwen2.5-3B) | 0.273 | 0.910 | 0.161 | 0.081 | 0.601 |

★ Calibrated threshold is the best-performing configuration overall.

### Table 2: F1 by Entity Rarity — The Rarity Reversal Finding

| Rarity | Halluc% | Method A | Method B | Method C | All-Wrong% |
|--------|:-------:|:--------:|:--------:|:--------:|:----------:|
| Very Rare | 18.2% | 0.743 | 0.701 | 0.312 | 12.7% |
| Rare | 22.4% | 0.718 | 0.689 | 0.289 | 16.3% |
| Medium | 27.1% | 0.681 | 0.654 | 0.261 | 21.8% |
| Frequent | 31.8% | 0.624 | 0.598 | 0.238 | 28.4% |
| Very Frequent | 35.6% | 0.587 | 0.561 | 0.201 | 33.6% |

**Counterintuitive:** More frequent entities are harder to verify, not easier. Frequent entities generate more *inferential* facts — things implied by their prominence but not explicitly stated on Wikipedia.

### Table 3: F1 by Source LLM

| LLM | Halluc% | Method A | Method B | Method C |
|-----|:-------:|:--------:|:--------:|:--------:|
| InstructGPT | 24.3% | 0.701 | 0.672 | 0.298 |
| ChatGPT | 21.7% | 0.719 | 0.688 | 0.312 |
| PerplexityAI | 34.2% | 0.598 | 0.574 | 0.218 |

PerplexityAI hallucinations are hardest to verify — it generates more confident-sounding inferential claims.

---

## Four Failure Modes

When all three methods fail simultaneously (3,907 cases analyzed):

| Failure Mode | % of All-Wrong Cases | Description |
|---|:---:|---|
| **Inferential facts** | ~40% | Facts implied but not stated explicitly in Wikipedia — *"He was known for his hard-partying lifestyle"* |
| **Retrieval failures** | ~25% | BM25 retrieves wrong sentence; LLM knowledge insufficient |
| **World knowledge facts** | ~20% | True facts not covered on Wikipedia at all |
| **Relational/compound facts** | ~15% | Facts requiring reasoning across multiple sentences |

---

## Methods

### Method A — NLI Verification (CPU-compatible · deployed live)
1. Fetch Wikipedia page for the subject entity
2. Split into sentences, build BM25 index with **name-token filtering** (removes subject name from query and corpus to prevent name-dominance bias)
3. Retrieve top matching sentence for each atomic fact
4. Run RoBERTa-Large-MNLI on (evidence, claim) pair
5. Predict Supported if entailment probability ≥ threshold

**Name-token filtering** is a non-obvious implementation detail — without it, BM25 scores are dominated by the person's name appearing in every sentence, returning irrelevant matches.

### Method B — Retrieval+LLM (GPU required)
Same BM25 retrieval as Method A, but uses **Qwen2.5-3B-Instruct** as the verifier instead of NLI. Prompts the LLM to read the evidence and output "Supported" or "Unsupported".

### Method C — Direct LLM (GPU required)
No retrieval. Asks **Qwen2.5-3B-Instruct** directly from parametric memory whether the claim is factually accurate. Establishes baseline for what a small LLM knows without external evidence.

---

## Repository Structure

```
factuality-verification-analysis/
│
├── 01_explore_data.py              # Dataset statistics and sanity checks
├── 02_build_dataset.py             # Build evaluation_dataset.csv from JSONL
├── 03_method_a_nli.py              # Method A: BM25 + RoBERTa-MNLI (CPU)
├── 04_method_b_retrieval_llm.py    # Method B: BM25 + Qwen2.5-3B (GPU)
├── 05_method_c_direct_llm.py       # Method C: Direct Qwen2.5-3B (GPU)
├── 06_error_analysis.py            # All 5 failure mode analyses
├── 07_figures_and_tables.py        # Generate all figures and result tables
│
├── app.py                          # Gradio demo (Method A live)
├── requirements.txt
│
├── labeled/                        # Raw FActScore JSONL files
│   ├── InstructGPT.jsonl
│   ├── ChatGPT.jsonl
│   └── PerplexityAI.jsonl
│
├── evaluation_dataset.csv                  # Built by 02_build_dataset.py
├── evaluation_dataset_method_a_v2.csv      # After running Method A
├── evaluation_dataset_method_ab.csv        # After running Methods A+B
├── evaluation_dataset_all_methods.csv      # Final — all three methods
│
└── factuality_evaluation_figures.png       # All 6 figures
```

---

## How to Run

### Prerequisites
```bash
git clone https://github.com/oladri-renuka/factuality-verification-analysis.git
cd factuality-verification-analysis
pip install -r requirements.txt
```

### Step 1 — Build dataset (MacBook / any CPU)
```bash
python 01_explore_data.py       # optional sanity check
python 02_build_dataset.py      # creates evaluation_dataset.csv
```

### Step 2 — Run Method A (MacBook CPU · ~2 hours)
```bash
python 03_method_a_nli.py
```
Checkpoints every 500 rows. If interrupted, re-run and it resumes automatically.

### Step 3 — Run Methods B and C (GPU required)
Methods B and C use Qwen2.5-3B-Instruct which requires a GPU. They were run on **Kaggle T4 GPU** (free tier):


### Step 4 — Analysis and figures (MacBook / any CPU)
```bash
python 06_error_analysis.py         # prints all analyses
python 07_figures_and_tables.py     # saves figures as PDF and PNG
```

### Run the Gradio demo locally
```bash
python app.py
# Open http://127.0.0.1:7860
```

---

## Live Demo

The deployed app runs **Method A only** (NLI + BM25) — Qwen2.5-3B is too large for free CPU inference on HuggingFace Spaces. The demo shows both the default threshold (0.50) and calibrated threshold (0.10) side by side for each sentence, directly demonstrating the key finding.

**[→ Try the live demo](https://oladri-renuka-factuality-verifier.hf.space)**

---

## Stack

| Component | Tool |
|---|---|
| NLI model | RoBERTa-Large-MNLI |
| LLM verifier | Qwen2.5-3B-Instruct |
| Retrieval | BM25 (rank_bm25) with name-token filtering |
| Knowledge source | Wikipedia API |
| GPU compute | Kaggle T4 (free tier) |
| Demo framework | Gradio |
| Deployment | HuggingFace Spaces |

---

## Dataset

[FActScore](https://github.com/shmsw25/FActScore) — human-annotated atomic facts from LLM-generated biographies.

- **14,525** atomic facts · **183** entities
- Labels: Supported (S), Not Supported (NS) — IR (irrelevant) excluded
- Rarity buckets based on entity Wikipedia page view frequency
- Source LLMs: InstructGPT, ChatGPT, PerplexityAI
