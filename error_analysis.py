"""
Step 6: Error analysis — all five analyses from the paper.
Input:  evaluation_dataset_all_methods.csv
Output: Prints all analysis results (no new files)

Analyses:
  1. Rarity vs hallucination vs verification accuracy
  2. All-methods failure cases (3,907 hard cases)
  3. Where direct LLM beats retrieval-based methods
  4. Method A vs B disagreement deep dive
  5. NLI confidence threshold calibration
"""
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

df = pd.read_csv("evaluation_dataset_all_methods.csv")
y_true   = df["human_label"].values
y_pred_a = df["method_a_pred"].values
y_pred_b = df["method_b_pred"].values
y_pred_c = df["method_c_pred"].values

rarity_order = ["very rare", "rare", "medium", "freq", "very freq"]

# ── Analysis 1: Rarity vs hallucination vs verification ───────────────────────
print("="*65)
print("ANALYSIS 1: RARITY vs HALLUCINATION vs VERIFICATION ACCURACY")
print("="*65)
print(f"\n{'Rarity':<12} {'Halluc%':>8} {'A F1':>8} {'B F1':>8} {'C F1':>8} {'n':>6}")
print("-"*55)
for rarity in rarity_order:
    mask = df["rarity"] == rarity
    halluc = (y_true[mask] == 0).mean()
    a = f1_score(y_true[mask], y_pred_a[mask])
    b = f1_score(y_true[mask], y_pred_b[mask])
    c = f1_score(y_true[mask], y_pred_c[mask])
    print(f"{rarity:<12} {halluc:>8.1%} {a:>8.3f} {b:>8.3f} {c:>8.3f} {mask.sum():>6}")

# ── Analysis 2: All-methods failure cases ─────────────────────────────────────
print(f"\n{'='*65}")
print("ANALYSIS 2: ALL-METHODS FAILURE CASES")
print("="*65)
all_wrong_mask = (y_pred_a != y_true) & (y_pred_b != y_true) & (y_pred_c != y_true)
all_wrong_df   = df[all_wrong_mask].copy()
print(f"\nTotal all-wrong cases: {all_wrong_mask.sum()}")

print(f"\nBy rarity:")
for rarity in rarity_order:
    n_rarity = (df["rarity"] == rarity).sum()
    n_wrong  = (all_wrong_df["rarity"] == rarity).sum()
    print(f"  {rarity:<12}: {n_wrong:>5} ({n_wrong/n_rarity:.1%} of {rarity} facts)")

print(f"\nBy true label:")
print(f"  Actually S  (all predict NS): {(all_wrong_df['human_label']==1).sum()}")
print(f"  Actually NS (all predict S):  {(all_wrong_df['human_label']==0).sum()}")

print(f"\nSample of all-wrong cases (20 examples):")
print("-"*65)
sample = all_wrong_df.sample(min(20, len(all_wrong_df)), random_state=42)
for _, row in sample.iterrows():
    label = "S" if row["human_label"] == 1 else "NS"
    print(f"Topic: {row['topic']} [{row['rarity']}]")
    print(f"Fact:  {row['atomic_fact']}")
    print(f"True:  {label} | A:{int(row['method_a_pred'])} B:{int(row['method_b_pred'])} C:{int(row['method_c_pred'])}")
    print(f"Evid:  {str(row['retrieved_evidence'])[:100]}...")
    print()

# ── Analysis 3: Where Method C beats A and B ──────────────────────────────────
print(f"{'='*65}")
print("ANALYSIS 3: WHERE DIRECT LLM BEATS RETRIEVAL-BASED METHODS")
print("="*65)
c_only_mask = (y_pred_c == y_true) & (y_pred_a != y_true) & (y_pred_b != y_true)
c_only_df   = df[c_only_mask].copy()
print(f"\nOnly C correct: {c_only_mask.sum()}")
print(f"\nBy true label:")
print(f"  Actually S  (C correct S, A+B say NS): {(c_only_df['human_label']==1).sum()}")
print(f"  Actually NS (C correct NS, A+B say S): {(c_only_df['human_label']==0).sum()}")

print(f"\nSample — C correct, A+B wrong (15 examples):")
sample_c = c_only_df[c_only_df["human_label"]==1].sample(
    min(15, (c_only_df["human_label"]==1).sum()), random_state=42
)
for _, row in sample_c.iterrows():
    print(f"Topic: {row['topic']} [{row['rarity']}]")
    print(f"Fact:  {row['atomic_fact']}")
    print(f"Evid:  {str(row['retrieved_evidence'])[:100]}...")
    print()

# ── Analysis 4: Method A vs B disagreement ───────────────────────────────────
print(f"{'='*65}")
print("ANALYSIS 4: METHOD A vs B DISAGREEMENT DEEP DIVE")
print("="*65)
b_recovers_mask = (y_pred_a == 0) & (y_pred_b == 1) & (y_true == 1)
b_recovers_df   = df[b_recovers_mask].copy()
print(f"\nB recovers (A=NS → B=S, True=S): {b_recovers_mask.sum()}")

sample_b = b_recovers_df.sample(min(10, len(b_recovers_df)), random_state=42)
for _, row in sample_b.iterrows():
    print(f"Topic: {row['topic']} [{row['rarity']}]")
    print(f"Fact:  {row['atomic_fact']}")
    print(f"Evid:  {str(row['retrieved_evidence'])[:100]}...")
    print()

b_overcorrects = ((y_pred_a == 1) & (y_pred_b == 0) & (y_true == 1)).sum()
print(f"B overcorrects (A=S → B=NS, True=S): {b_overcorrects}")

# ── Analysis 5: NLI threshold calibration ────────────────────────────────────
print(f"\n{'='*65}")
print("ANALYSIS 5: NLI CONFIDENCE THRESHOLD CALIBRATION")
print("="*65)

entailment_scores = df["method_a_entailment"].values
best_f1        = 0
best_threshold = 0.5

print(f"\n{'Threshold':>10} {'F1':>8} {'Precision':>10} {'Recall':>8} {'n_S':>8}")
print("-"*50)
for t in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
          0.60, 0.70, 0.80, 0.90]:
    preds = (entailment_scores >= t).astype(int)
    if preds.sum() == 0:
        continue
    f1   = f1_score(y_true, preds)
    prec = precision_score(y_true, preds)
    rec  = recall_score(y_true, preds)
    marker = " ← BEST" if f1 > best_f1 else ""
    print(f"{t:>10.2f} {f1:>8.3f} {prec:>10.3f} {rec:>8.3f} "
          f"{preds.sum():>8}{marker}")
    if f1 > best_f1:
        best_f1        = f1
        best_threshold = t

default_f1 = f1_score(y_true, (entailment_scores >= 0.5).astype(int))
print(f"\nBest threshold:   {best_threshold}  (F1={best_f1:.3f})")
print(f"Default (0.50):   F1={default_f1:.3f}")
print(f"Improvement:      +{best_f1-default_f1:.3f}")
