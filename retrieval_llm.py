"""
Step 4: Method B — BM25 retrieval + Qwen2.5-3B-Instruct verification.
Input:  evaluation_dataset_method_a_v2.csv
Output: evaluation_dataset_method_ab.csv

Uses same BM25 retrieval as Method A but LLM as verifier instead of NLI.
Run on Kaggle T4 GPU. Expected time: ~90 minutes.
"""
import pandas as pd
import numpy as np
import torch
import re
import wikipediaapi
from transformers import AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              cohen_kappa_score, accuracy_score)

print(f"GPU available: {torch.cuda.is_available()}")

# ── Load Qwen ─────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
print(f"Loading {MODEL_NAME}...")

tokenizer_llm = AutoTokenizer.from_pretrained(MODEL_NAME)
model_llm = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model_llm.eval()
print(f"Model loaded. Device: {next(model_llm.parameters()).device}")

# ── Wikipedia + BM25 (same as Method A) ──────────────────────────────────────
wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="FactualityEval/1.0 (research@umd.edu)"
)
wiki_cache = {}

def get_full_wiki_sentences(topic):
    if topic in wiki_cache:
        return wiki_cache[topic]
    page = wiki.page(topic)
    if not page.exists():
        wiki_cache[topic] = []
        return []
    sentences = re.split(r'(?<=[.!?])\s+', page.text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    wiki_cache[topic] = sentences
    return sentences

def preprocess_for_bm25(text, topic_to_remove=None):
    stopwords = {
        "the","a","an","is","was","are","were","has","have","had",
        "in","on","at","to","of","and","or","but","he","she","his",
        "her","it","its","they","their","who","which","that","this",
        "with","for","as","by","be","been","being","also","from","not","no"
    }
    name_tokens = set(topic_to_remove.lower().split()) if topic_to_remove else set()
    return [
        t for t in text.lower().split()
        if t not in stopwords and t not in name_tokens and len(t) > 2
    ]

def get_bm25_evidence(topic, atomic_fact):
    sentences = get_full_wiki_sentences(topic)
    if not sentences:
        return None
    tokenized_corpus = [preprocess_for_bm25(s, topic) for s in sentences]
    valid_indices = [i for i, t in enumerate(tokenized_corpus) if t]
    if not valid_indices:
        return sentences[0]
    valid_corpus    = [tokenized_corpus[i] for i in valid_indices]
    valid_sentences = [sentences[i] for i in valid_indices]
    query = preprocess_for_bm25(atomic_fact, topic)
    if not query:
        return sentences[0]
    scores = BM25Okapi(valid_corpus).get_scores(query)
    best   = np.argmax(scores)
    return valid_sentences[best] if scores[best] > 0 else valid_sentences[0]

# ── Method B verification ─────────────────────────────────────────────────────
def verify_with_retrieval_llm(atomic_fact, evidence, tokenizer, model):
    if evidence is None or evidence == "NO_WIKI_PAGE":
        return 0, "no_evidence"

    prompt = (
        f"You are a fact-checker. Read the evidence carefully, then determine "
        f"if the claim is supported by that evidence.\n\n"
        f"Evidence: {evidence}\n\n"
        f"Claim: {atomic_fact}\n\n"
        f"Answer with exactly one word — Supported or Unsupported:"
    )
    messages = [
        {"role": "system", "content": "You are a precise fact-checker. "
                                       "Answer with exactly one word: "
                                       "Supported or Unsupported."},
        {"role": "user", "content": prompt}
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
            temperature=1.0, pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

    if "unsupported" in response:
        return 0, response
    elif "supported" in response:
        return 1, response
    else:
        return 0, f"AMBIGUOUS:{response}"

# ── Quick test ────────────────────────────────────────────────────────────────
print("\nTesting Method B on 3 examples...")
test_cases = [
    ("Doug Sheehan is an American actor.",
     "Douglas Stuart Sheehan was an American actor who played Ben Gibson in Knots Landing."),
    ("He appeared in Beverly Hills, 90210.",
     "He also appeared on Sabrina the Teenage Witch as Sabrina's father."),
    ("He was born on August 24, 1956.",
     "Douglas Stuart Sheehan (April 27, 1949 – June 29, 2024) was an American actor."),
]
for fact, evidence in test_cases:
    pred, resp = verify_with_retrieval_llm(fact, evidence, tokenizer_llm, model_llm)
    print(f"Fact:     {fact}")
    print(f"Pred:     {'Supported' if pred else 'Unsupported'} ({resp})")
    print()

# ── Run Method B ──────────────────────────────────────────────────────────────
df = pd.read_csv("evaluation_dataset_method_a_v2.csv")

method_b_preds     = []
method_b_responses = []

checkpoint_b = "method_b_checkpoint.csv"
start_idx    = 0

try:
    ckpt          = pd.read_csv(checkpoint_b)
    start_idx     = len(ckpt)
    method_b_preds     = ckpt["method_b_pred"].tolist()
    method_b_responses = ckpt["method_b_response"].tolist()
    print(f"Resuming Method B from row {start_idx}")
except Exception:
    print("Starting Method B fresh")

for idx in tqdm(range(start_idx, len(df))):
    row      = df.iloc[idx]
    fact     = row["atomic_fact"]
    topic    = row["topic"]
    evidence = get_bm25_evidence(topic, fact)

    pred, response = verify_with_retrieval_llm(fact, evidence, tokenizer_llm, model_llm)
    method_b_preds.append(pred)
    method_b_responses.append(response)

    if (idx + 1) % 500 == 0:
        tmp = df.iloc[:idx+1].copy()
        tmp["method_b_pred"]     = method_b_preds
        tmp["method_b_response"] = method_b_responses
        tmp.to_csv(checkpoint_b, index=False)
        print(f"  Checkpoint saved at row {idx+1}")

df["method_b_pred"]     = method_b_preds
df["method_b_response"] = method_b_responses
df.to_csv("evaluation_dataset_method_ab.csv", index=False)
print("\nSaved evaluation_dataset_method_ab.csv")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_true   = df["human_label"].values
y_pred_a = df["method_a_pred"].values
y_pred_b = np.array(method_b_preds)

ambiguous = [r for r in method_b_responses if r.startswith("AMBIGUOUS")]
print(f"Ambiguous responses: {len(ambiguous)} ({len(ambiguous)/len(df):.1%})")

print("\n" + "="*60)
print("METHOD A vs METHOD B")
print("="*60)
print(f"\n{'Metric':<12} {'Method A':>10} {'Method B':>10} {'Delta':>10}")
print("-"*45)
for name, fn in [("Accuracy", accuracy_score), ("F1", f1_score),
                  ("Precision", precision_score), ("Recall", recall_score),
                  ("Kappa", cohen_kappa_score)]:
    a = fn(y_true, y_pred_a)
    b = fn(y_true, y_pred_b)
    d = b - a
    print(f"{name:<12} {a:>10.3f} {b:>10.3f} {d:>+10.3f} {'↑' if d>0 else '↓'}")

rarity_order = ["very rare", "rare", "medium", "freq", "very freq"]
print(f"\nF1 by rarity:")
for rarity in rarity_order:
    mask = df["rarity"] == rarity
    if mask.sum() > 0:
        a = f1_score(y_true[mask], y_pred_a[mask])
        b = f1_score(y_true[mask], y_pred_b[mask])
        print(f"  {rarity:<12}: A={a:.3f} B={b:.3f} Δ={b-a:+.3f}")

# Disagreement analysis
disagree = y_pred_a != y_pred_b
print(f"\nDisagreement: {disagree.sum()} ({disagree.mean():.1%})")

mask_a_ns_b_s = (y_pred_a == 0) & (y_pred_b == 1)
if mask_a_ns_b_s.sum() > 0:
    print(f"A=NS, B=S → actually S: {(y_true[mask_a_ns_b_s]==1).mean():.1%}")

mask_a_s_b_ns = (y_pred_a == 1) & (y_pred_b == 0)
if mask_a_s_b_ns.sum() > 0:
    print(f"A=S, B=NS → actually S: {(y_true[mask_a_s_b_ns]==1).mean():.1%}")
