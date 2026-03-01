"""
Step 5: Method C — Direct LLM verification (no retrieval).
Input:  evaluation_dataset_method_ab.csv
Output: evaluation_dataset_all_methods.csv

Qwen2.5-3B-Instruct with parametric knowledge only.
Run on Kaggle T4 GPU. Expected time: ~40 minutes.
NOTE: Assumes model_llm and tokenizer_llm already loaded from 04_method_b.
If running fresh, the model loading block below will run automatically.
"""
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              cohen_kappa_score, accuracy_score)

# ── Load model if not already in memory ──────────────────────────────────────
try:
    model_llm
    tokenizer_llm
    print("Using already-loaded Qwen model.")
except NameError:
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    print(f"Loading {MODEL_NAME}...")
    tokenizer_llm = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model_llm.eval()
    print("Model loaded.")

# ── Method C verification ─────────────────────────────────────────────────────
def verify_direct_llm(atomic_fact, tokenizer, model):
    """Verify claim using parametric knowledge only — no retrieval."""
    prompt = (
        f"You are a fact-checker. Based on your own knowledge, "
        f"determine if the following claim is factually accurate.\n\n"
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
                       truncation=True, max_length=256).to(model.device)

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

# ── Run Method C ──────────────────────────────────────────────────────────────
print("Running Method C (Direct LLM — no retrieval)...")
df = pd.read_csv("evaluation_dataset_method_ab.csv")

method_c_preds     = []
method_c_responses = []

checkpoint_c = "method_c_checkpoint.csv"
start_idx    = 0

try:
    ckpt               = pd.read_csv(checkpoint_c)
    start_idx          = len(ckpt)
    method_c_preds     = ckpt["method_c_pred"].tolist()
    method_c_responses = ckpt["method_c_response"].tolist()
    print(f"Resuming Method C from row {start_idx}")
except Exception:
    print("Starting Method C fresh")

for idx in tqdm(range(start_idx, len(df))):
    fact = df.iloc[idx]["atomic_fact"]
    pred, response = verify_direct_llm(fact, tokenizer_llm, model_llm)
    method_c_preds.append(pred)
    method_c_responses.append(response)

    if (idx + 1) % 500 == 0:
        tmp = df.iloc[:idx+1].copy()
        tmp["method_c_pred"]     = method_c_preds
        tmp["method_c_response"] = method_c_responses
        tmp.to_csv(checkpoint_c, index=False)
        print(f"  Checkpoint saved at row {idx+1}")

df["method_c_pred"]     = method_c_preds
df["method_c_response"] = method_c_responses
df.to_csv("evaluation_dataset_all_methods.csv", index=False)
print("\nSaved evaluation_dataset_all_methods.csv")

# ── Three-way evaluation ──────────────────────────────────────────────────────
y_true   = df["human_label"].values
y_pred_a = df["method_a_pred"].values
y_pred_b = df["method_b_pred"].values
y_pred_c = np.array(method_c_preds)

ambiguous_c = [r for r in method_c_responses if r.startswith("AMBIGUOUS")]
print(f"Ambiguous responses: {len(ambiguous_c)} ({len(ambiguous_c)/len(df):.1%})")

print("\n" + "="*65)
print("THREE-WAY COMPARISON: METHOD A vs B vs C")
print("="*65)
print(f"\n{'Metric':<12} {'Method A':>10} {'Method B':>10} {'Method C':>10}")
print("-"*50)

for name, fn in [("Accuracy", accuracy_score), ("F1", f1_score),
                  ("Precision", precision_score), ("Recall", recall_score),
                  ("Kappa", cohen_kappa_score)]:
    a = fn(y_true, y_pred_a)
    b = fn(y_true, y_pred_b)
    c = fn(y_true, y_pred_c)
    best = max(a, b, c)
    print(f"{name:<12} "
          f"{'*' if a==best else ' '}{a:.3f}{'':>4}"
          f"{'*' if b==best else ' '}{b:.3f}{'':>4}"
          f"{'*' if c==best else ' '}{c:.3f}")

rarity_order = ["very rare", "rare", "medium", "freq", "very freq"]
print(f"\nF1 by rarity:")
print(f"{'Rarity':<12} {'A':>8} {'B':>8} {'C':>8} {'Best':>6}")
print("-"*45)
for rarity in rarity_order:
    mask = df["rarity"] == rarity
    if mask.sum() > 0:
        a = f1_score(y_true[mask], y_pred_a[mask])
        b = f1_score(y_true[mask], y_pred_b[mask])
        c = f1_score(y_true[mask], y_pred_c[mask])
        best = ["A","B","C"][[a,b,c].index(max(a,b,c))]
        print(f"{rarity:<12} {a:>8.3f} {b:>8.3f} {c:>8.3f} {best:>6}")

print(f"\nPrediction rate comparison:")
print(f"  {'Method':<12} {'%Supported':>12}")
for name, preds in [("Method A", y_pred_a), ("Method B", y_pred_b),
                     ("Method C", y_pred_c), ("Human", y_true)]:
    print(f"  {name:<12} {(preds==1).mean():>12.1%}")

all_wrong = (y_pred_a != y_true) & (y_pred_b != y_true) & (y_pred_c != y_true)
c_only    = (y_pred_c == y_true) & (y_pred_a != y_true) & (y_pred_b != y_true)
print(f"\nAll three wrong: {all_wrong.sum()} ({all_wrong.mean():.1%})")
print(f"Only C correct:  {c_only.sum()}")
