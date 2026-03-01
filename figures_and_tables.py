"""
Step 7: Generate all six figures and three result tables.
Input:  evaluation_dataset_all_methods.csv
Output: factuality_evaluation_figures.pdf
        factuality_evaluation_figures.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              accuracy_score, cohen_kappa_score)

df = pd.read_csv("evaluation_dataset_all_methods.csv")
y_true   = df["human_label"].values
y_pred_a = df["method_a_pred"].values
y_pred_b = df["method_b_pred"].values
y_pred_c = df["method_c_pred"].values

rarity_order = ["very rare", "rare", "medium", "freq", "very freq"]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    "Error Analysis and Failure Mode Comparison\n"
    "of Atomic Factuality Verification Methods",
    fontsize=14, fontweight="bold", y=1.02
)

# ── Figure 1: Overall metrics ─────────────────────────────────────────────────
ax1 = axes[0, 0]
metrics_names = ["Accuracy", "F1", "Precision", "Recall", "Kappa"]
fns = [accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score]
vals_a = [fn(y_true, y_pred_a) for fn in fns]
vals_b = [fn(y_true, y_pred_b) for fn in fns]
vals_c = [fn(y_true, y_pred_c) for fn in fns]

x = np.arange(len(metrics_names))
w = 0.25
for bars, vals, label, color in [
    (ax1.bar(x-w, vals_a, w, color="#2196F3", alpha=0.85), vals_a, "Method A (NLI)", "#2196F3"),
    (ax1.bar(x,   vals_b, w, color="#4CAF50", alpha=0.85), vals_b, "Method B (Retrieval+LLM)", "#4CAF50"),
    (ax1.bar(x+w, vals_c, w, color="#FF9800", alpha=0.85), vals_c, "Method C (Direct LLM)", "#FF9800"),
]:
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x()+bar.get_width()/2, v+0.01,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=7)

ax1.set_title("Figure 1: Overall Performance Comparison")
ax1.set_xticks(x); ax1.set_xticklabels(metrics_names)
ax1.set_ylim(0, 1.15); ax1.grid(axis="y", alpha=0.3)
ax1.legend(["Method A (NLI)", "Method B (Ret+LLM)", "Method C (Direct)"],
           fontsize=7, loc="upper right")

# ── Figure 2: F1 by rarity ────────────────────────────────────────────────────
ax2 = axes[0, 1]
halluc_rates, a_f1s, b_f1s, c_f1s = [], [], [], []
for rarity in rarity_order:
    mask = df["rarity"] == rarity
    halluc_rates.append((y_true[mask] == 0).mean())
    a_f1s.append(f1_score(y_true[mask], y_pred_a[mask]))
    b_f1s.append(f1_score(y_true[mask], y_pred_b[mask]))
    c_f1s.append(f1_score(y_true[mask], y_pred_c[mask]))

x = np.arange(len(rarity_order))
ax2.plot(x, a_f1s, "o-", color="#2196F3", lw=2, ms=8, label="Method A")
ax2.plot(x, b_f1s, "s-", color="#4CAF50", lw=2, ms=8, label="Method B")
ax2.plot(x, c_f1s, "^-", color="#FF9800", lw=2, ms=8, label="Method C")
ax2_twin = ax2.twinx()
ax2_twin.bar(x, halluc_rates, alpha=0.2, color="red")
ax2_twin.set_ylabel("Hallucination Rate", color="red", fontsize=9)
ax2_twin.tick_params(axis="y", labelcolor="red")
ax2_twin.set_ylim(0, 1.0)
ax2.set_title("Figure 2: F1 by Entity Rarity vs Hallucination Rate")
ax2.set_xticks(x)
ax2.set_xticklabels(["Very\nRare","Rare","Medium","Freq","Very\nFreq"], fontsize=8)
ax2.legend(fontsize=8, loc="lower left")
ax2.set_ylim(0, 1.0); ax2.grid(alpha=0.3)

# ── Figure 3: NLI threshold calibration ──────────────────────────────────────
ax3 = axes[0, 2]
ent_scores = df["method_a_entailment"].values
thresholds = np.arange(0.05, 0.96, 0.05)
f1s, precs, recs = [], [], []
for t in thresholds:
    preds = (ent_scores >= t).astype(int)
    if preds.sum() == 0:
        f1s.append(0); precs.append(0); recs.append(0); continue
    f1s.append(f1_score(y_true, preds))
    precs.append(precision_score(y_true, preds))
    recs.append(recall_score(y_true, preds))

best_t  = thresholds[np.argmax(f1s)]
best_f1 = max(f1s)
ax3.plot(thresholds, f1s,   "b-",  lw=2.5, label="F1")
ax3.plot(thresholds, precs, "g--", lw=1.5, label="Precision")
ax3.plot(thresholds, recs,  "r--", lw=1.5, label="Recall")
ax3.axvline(0.5,    color="gray", ls=":", lw=1.5, label="Default (0.5)")
ax3.axvline(best_t, color="blue", ls=":", lw=1.5, label=f"Optimal ({best_t:.2f})")
ax3.scatter([best_t], [best_f1], color="blue", s=100, zorder=5)
ax3.set_title("Figure 3: NLI Threshold Calibration (Method A)")
ax3.legend(fontsize=8); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.grid(alpha=0.3)

# ── Figure 4: F1 by source LLM ───────────────────────────────────────────────
ax4 = axes[1, 0]
llms = ["InstructGPT", "ChatGPT", "PerplexityAI"]
llm_h, a_llm, b_llm, c_llm = [], [], [], []
for llm in llms:
    mask = df["llm"] == llm
    llm_h.append((y_true[mask] == 0).mean())
    a_llm.append(f1_score(y_true[mask], y_pred_a[mask]))
    b_llm.append(f1_score(y_true[mask], y_pred_b[mask]))
    c_llm.append(f1_score(y_true[mask], y_pred_c[mask]))

x = np.arange(len(llms)); w = 0.25
ax4.bar(x-w, a_llm, w, color="#2196F3", alpha=0.85, label="Method A")
ax4.bar(x,   b_llm, w, color="#4CAF50", alpha=0.85, label="Method B")
ax4.bar(x+w, c_llm, w, color="#FF9800", alpha=0.85, label="Method C")
for i, h in enumerate(llm_h):
    ax4.text(i, 0.02, f"Halluc:\n{h:.1%}", ha="center",
             fontsize=7.5, color="darkred", fontweight="bold")
ax4.set_title("Figure 4: F1 by Source LLM")
ax4.set_xticks(x); ax4.set_xticklabels(llms)
ax4.legend(fontsize=8); ax4.set_ylim(0, 0.85); ax4.grid(axis="y", alpha=0.3)

# ── Figure 5: All-methods failure rate ───────────────────────────────────────
ax5 = axes[1, 1]
all_wrong = (y_pred_a != y_true) & (y_pred_b != y_true) & (y_pred_c != y_true)
aw_rates, aw_counts = [], []
for rarity in rarity_order:
    mask = df["rarity"] == rarity
    aw_rates.append((all_wrong & mask).sum() / mask.sum())
    aw_counts.append((all_wrong & mask).sum())

bars = ax5.bar(rarity_order, aw_rates,
               color=["#e74c3c","#e67e22","#f1c40f","#2ecc71","#3498db"],
               alpha=0.85, edgecolor="black", lw=0.5)
for bar, count in zip(bars, aw_counts):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f"n={count}", ha="center", va="bottom", fontsize=8)
ax5.set_title("Figure 5: All-Methods Failure Rate by Entity Rarity")
ax5.set_xticklabels(["Very\nRare","Rare","Medium","Freq","Very\nFreq"])
ax5.set_ylim(0, 0.45); ax5.grid(axis="y", alpha=0.3)

# ── Figure 6: Prediction bias ─────────────────────────────────────────────────
ax6 = axes[1, 2]
categories     = ["Method A\n(NLI)", "Method B\n(Ret+LLM)",
                  "Method C\n(Direct)", "Human"]
supported_rates = [(y_pred_a==1).mean(), (y_pred_b==1).mean(),
                   (y_pred_c==1).mean(), (y_true==1).mean()]
unsupported     = [1-r for r in supported_rates]
x = np.arange(len(categories))
ax6.bar(x, supported_rates,  label="Supported",   color="#27ae60", alpha=0.85)
ax6.bar(x, unsupported, bottom=supported_rates,
        label="Unsupported", color="#e74c3c", alpha=0.85)
for i, r in enumerate(supported_rates):
    ax6.text(i, r/2, f"{r:.1%}", ha="center", va="center",
             fontsize=9, color="white", fontweight="bold")
ax6.axhline((y_true==1).mean(), color="green", ls="--", lw=1.5, alpha=0.7)
ax6.set_title("Figure 6: Prediction Bias")
ax6.set_xticks(x); ax6.set_xticklabels(categories, fontsize=8)
ax6.legend(fontsize=8, loc="lower right"); ax6.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig("factuality_evaluation_figures.pdf", bbox_inches="tight", dpi=300)
plt.savefig("factuality_evaluation_figures.png", bbox_inches="tight", dpi=300)
plt.show()
print("Saved factuality_evaluation_figures.pdf and .png")

# ── Tables ─────────────────────────────────────────────────────────────────────
print("\nTABLE 1: MAIN RESULTS")
print("="*70)
print(f"{'Method':<28} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Kappa':>7} {'Acc':>6}")
print("-"*70)
for name, preds in [("A: NLI (BM25+RoBERTa)", y_pred_a),
                     ("B: Retrieval+LLM (BM25+Qwen)", y_pred_b),
                     ("C: Direct LLM (Qwen)", y_pred_c)]:
    print(f"{name:<28} "
          f"{f1_score(y_true,preds):>6.3f} "
          f"{precision_score(y_true,preds):>6.3f} "
          f"{recall_score(y_true,preds):>6.3f} "
          f"{cohen_kappa_score(y_true,preds):>7.3f} "
          f"{accuracy_score(y_true,preds):>6.3f}")
opt = (df["method_a_entailment"].values >= 0.1).astype(int)
print(f"{'A: NLI (threshold=0.10)':<28} "
      f"{f1_score(y_true,opt):>6.3f} "
      f"{precision_score(y_true,opt):>6.3f} "
      f"{recall_score(y_true,opt):>6.3f} "
      f"{cohen_kappa_score(y_true,opt):>7.3f} "
      f"{accuracy_score(y_true,opt):>6.3f}  ← calibrated")

print("\nTABLE 2: F1 BY ENTITY RARITY")
print("="*70)
print(f"{'Rarity':<12} {'Halluc%':>8} {'Method A':>9} "
      f"{'Method B':>9} {'Method C':>9} {'All-Wrong%':>11}")
print("-"*70)
for rarity in rarity_order:
    mask = df["rarity"] == rarity
    aw = ((y_pred_a!=y_true)&(y_pred_b!=y_true)&(y_pred_c!=y_true)&mask).sum()/mask.sum()
    print(f"{rarity:<12} "
          f"{(y_true[mask]==0).mean():>8.1%} "
          f"{f1_score(y_true[mask],y_pred_a[mask]):>9.3f} "
          f"{f1_score(y_true[mask],y_pred_b[mask]):>9.3f} "
          f"{f1_score(y_true[mask],y_pred_c[mask]):>9.3f} "
          f"{aw:>11.1%}")

print("\nTABLE 3: F1 BY SOURCE LLM")
print("="*65)
print(f"{'LLM':<15} {'Halluc%':>8} {'Method A':>9} "
      f"{'Method B':>9} {'Method C':>9}")
print("-"*65)
for llm in ["InstructGPT", "ChatGPT", "PerplexityAI"]:
    mask = df["llm"] == llm
    print(f"{llm:<15} "
          f"{(y_true[mask]==0).mean():>8.1%} "
          f"{f1_score(y_true[mask],y_pred_a[mask]):>9.3f} "
          f"{f1_score(y_true[mask],y_pred_b[mask]):>9.3f} "
          f"{f1_score(y_true[mask],y_pred_c[mask]):>9.3f}")
