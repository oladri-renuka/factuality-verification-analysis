"""
Step 3: Method A — BM25 retrieval + RoBERTa-Large-MNLI verification.
Input:  evaluation_dataset.csv
Output: evaluation_dataset_method_a_v2.csv

Key implementation detail: removes subject name tokens from BM25 query
and corpus to prevent name-dominance bias in biography retrieval.

Run on Kaggle T4 GPU. Expected time: ~20 minutes.
"""
# ── Install ───────────────────────────────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "transformers", "torch",
                "wikipedia-api", "rank_bm25", "-q"])

import pandas as pd
import numpy as np
import torch
import re
import wikipediaapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rank_bm25 import BM25Okapi
from tqdm import tqdm

print("Libraries loaded")
print(f"GPU available: {torch.cuda.is_available()}")

# ── Load NLI model ────────────────────────────────────────────────────────────
print("\nLoading RoBERTa-Large-MNLI...")
model_name = "roberta-large-mnli"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
nli_model  = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nli_model = nli_model.to(device)
nli_model.eval()
print(f"Model loaded on {device}")
print(f"Label mapping: {nli_model.config.id2label}")

# ── Wikipedia retrieval ───────────────────────────────────────────────────────
wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="FactualityEval/1.0 (research@umd.edu)"
)
wiki_cache = {}

def get_full_wiki_sentences(topic):
    """Fetch full Wikipedia article and split into sentences."""
    if topic in wiki_cache:
        return wiki_cache[topic]

    page = wiki.page(topic)
    if not page.exists():
        wiki_cache[topic] = []
        return []

    # Use full text, not just summary
    text = page.text
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    wiki_cache[topic] = sentences
    return sentences


def preprocess_for_bm25(text, topic_to_remove=None):
    """Tokenize for BM25, removing topic name tokens to prevent name-dominance bias."""
    stopwords = {
        "the","a","an","is","was","are","were","has","have","had",
        "in","on","at","to","of","and","or","but","he","she","his",
        "her","it","its","they","their","who","which","that","this",
        "with","for","as","by","be","been","being","also","from","not","no"
    }
    name_tokens = set(topic_to_remove.lower().split()) if topic_to_remove else set()
    tokens = [
        t for t in text.lower().split()
        if t not in stopwords
        and t not in name_tokens
        and len(t) > 2
    ]
    return tokens


def get_bm25_evidence(topic, atomic_fact):
    """Retrieve most relevant Wikipedia sentence using BM25 with name filtering."""
    sentences = get_full_wiki_sentences(topic)
    if not sentences:
        return None

    tokenized_corpus = [
        preprocess_for_bm25(s, topic_to_remove=topic) for s in sentences
    ]
    valid_indices = [i for i, t in enumerate(tokenized_corpus) if t]
    if not valid_indices:
        return sentences[0]

    valid_corpus    = [tokenized_corpus[i] for i in valid_indices]
    valid_sentences = [sentences[i] for i in valid_indices]

    tokenized_query = preprocess_for_bm25(atomic_fact, topic_to_remove=topic)
    if not tokenized_query:
        return sentences[0]

    bm25   = BM25Okapi(valid_corpus)
    scores = bm25.get_scores(tokenized_query)
    best   = np.argmax(scores)

    return valid_sentences[best] if scores[best] > 0 else valid_sentences[0]


# ── NLI classification ────────────────────────────────────────────────────────
def classify_nli(premise, hypothesis):
    """
    Returns (prediction, entailment_prob, neutral_prob, contradiction_prob).
    prediction: 1=Supported (entailment > 0.5), 0=Not Supported
    """
    inputs = tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # roberta-large-mnli: 0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT
    con, neu, ent = float(probs[0]), float(probs[1]), float(probs[2])
    pred = 1 if ent > 0.5 else 0
    return pred, ent, neu, con


# ── Run Method A ──────────────────────────────────────────────────────────────
print("\nRunning Method A (BM25 + NLI)...")
df = pd.read_csv("evaluation_dataset.csv")

method_a_preds         = []
method_a_confidence    = []
method_a_entailment    = []
method_a_neutral       = []
method_a_contradiction = []
retrieved_evidences    = []
retrieval_found        = []

checkpoint_file = "method_a_v2_checkpoint.csv"
start_idx = 0

try:
    ckpt = pd.read_csv(checkpoint_file)
    start_idx              = len(ckpt)
    method_a_preds         = ckpt["method_a_pred"].tolist()
    method_a_confidence    = ckpt["method_a_confidence"].tolist()
    method_a_entailment    = ckpt["method_a_entailment"].tolist()
    method_a_neutral       = ckpt["method_a_neutral"].tolist()
    method_a_contradiction = ckpt["method_a_contradiction"].tolist()
    retrieved_evidences    = ckpt["retrieved_evidence"].tolist()
    retrieval_found        = ckpt["retrieval_found"].tolist()
    print(f"Resuming from checkpoint at row {start_idx}")
except Exception:
    print("Starting fresh")

for idx in tqdm(range(start_idx, len(df))):
    row   = df.iloc[idx]
    topic = row["topic"]
    fact  = row["atomic_fact"]

    evidence = get_bm25_evidence(topic, fact)

    if evidence is None:
        method_a_preds.append(0)
        method_a_confidence.append(0.0)
        method_a_entailment.append(0.0)
        method_a_neutral.append(0.0)
        method_a_contradiction.append(0.0)
        retrieved_evidences.append("NO_WIKI_PAGE")
        retrieval_found.append(False)
    else:
        pred, ent, neu, con = classify_nli(evidence, fact)
        method_a_preds.append(pred)
        method_a_confidence.append(ent)
        method_a_entailment.append(ent)
        method_a_neutral.append(neu)
        method_a_contradiction.append(con)
        retrieved_evidences.append(evidence)
        retrieval_found.append(True)

    if (idx + 1) % 500 == 0:
        tmp = df.iloc[:idx+1].copy()
        tmp["method_a_pred"]         = method_a_preds
        tmp["method_a_confidence"]   = method_a_confidence
        tmp["method_a_entailment"]   = method_a_entailment
        tmp["method_a_neutral"]      = method_a_neutral
        tmp["method_a_contradiction"]= method_a_contradiction
        tmp["retrieved_evidence"]    = retrieved_evidences
        tmp["retrieval_found"]       = retrieval_found
        tmp.to_csv(checkpoint_file, index=False)
        print(f"  Checkpoint saved at row {idx+1}")

df["method_a_pred"]          = method_a_preds
df["method_a_confidence"]    = method_a_confidence
df["method_a_entailment"]    = method_a_entailment
df["method_a_neutral"]       = method_a_neutral
df["method_a_contradiction"] = method_a_contradiction
df["retrieved_evidence"]     = retrieved_evidences
df["retrieval_found"]        = retrieval_found
df.to_csv("evaluation_dataset_method_a_v2.csv", index=False)
print("\nSaved evaluation_dataset_method_a_v2.csv")

# ── Evaluate ──────────────────────────────────────────────────────────────────
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              cohen_kappa_score, accuracy_score)

y_true = df["human_label"].values
y_pred = np.array(method_a_preds)

print("\n" + "="*50)
print("METHOD A (FIXED BM25) — FULL RESULTS")
print("="*50)
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
print(f"F1:        {f1_score(y_true, y_pred):.3f}")
print(f"Precision: {precision_score(y_true, y_pred):.3f}")
print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
print(f"Kappa:     {cohen_kappa_score(y_true, y_pred):.3f}")
print(f"Wiki found: {np.mean(retrieval_found):.1%}")

rarity_order = ["very rare", "rare", "medium", "freq", "very freq"]
print("\nF1 by rarity:")
for rarity in rarity_order:
    mask = df["rarity"] == rarity
    if mask.sum() > 0:
        f1   = f1_score(y_true[mask], y_pred[mask])
        prec = precision_score(y_true[mask], y_pred[mask])
        rec  = recall_score(y_true[mask], y_pred[mask])
        print(f"  {rarity:12s}: F1={f1:.3f} P={prec:.3f} R={rec:.3f} (n={mask.sum()})")

print("\nF1 by LLM:")
for llm in ["InstructGPT", "ChatGPT", "PerplexityAI"]:
    mask = df["llm"] == llm
    if mask.sum() > 0:
        f1   = f1_score(y_true[mask], y_pred[mask])
        prec = precision_score(y_true[mask], y_pred[mask])
        rec  = recall_score(y_true[mask], y_pred[mask])
        print(f"  {llm:15s}: F1={f1:.3f} P={prec:.3f} R={rec:.3f}")

ent_scores = np.array(method_a_entailment)
print(f"\nEntailment score distribution:")
print(f"  Mean:   {ent_scores.mean():.3f}")
print(f"  Median: {np.median(ent_scores):.3f}")
print(f"  >0.9:   {(ent_scores > 0.9).mean():.1%}")
print(f"  >0.5:   {(ent_scores > 0.5).mean():.1%}")
print(f"  <0.1:   {(ent_scores < 0.1).mean():.1%}")

fp = ((y_pred == 1) & (y_true == 0)).sum()
fn = ((y_pred == 0) & (y_true == 1)).sum()
print(f"\nFalse positives (predicted S, actually NS): {fp}")
print(f"False negatives (predicted NS, actually S): {fn}")
