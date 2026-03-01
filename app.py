import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

df = pd.read_csv("evaluation_dataset_all_methods.csv")
y_true = df["human_label"].values
y_pred_a = df["method_a_pred"].values
y_pred_b = df["method_b_pred"].values
y_pred_c = df["method_c_pred"].values

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Error Analysis and Failure Mode Comparison\nof Atomic Factuality Verification Methods", 
             fontsize=14, fontweight='bold', y=1.02)

colors = {'Method A\n(NLI)': '#2196F3', 
          'Method B\n(Retrieval+LLM)': '#4CAF50', 
          'Method C\n(Direct LLM)': '#FF9800'}
method_labels = list(colors.keys())
method_preds = [y_pred_a, y_pred_b, y_pred_c]

# ============================================================
# FIGURE 1: Overall metrics comparison — main results table
# ============================================================
ax1 = axes[0, 0]
metrics_names = ['Accuracy', 'F1', 'Precision', 'Recall', 'Kappa']
from sklearn.metrics import accuracy_score, cohen_kappa_score

method_a_vals = [accuracy_score(y_true, y_pred_a), f1_score(y_true, y_pred_a),
                 precision_score(y_true, y_pred_a), recall_score(y_true, y_pred_a),
                 cohen_kappa_score(y_true, y_pred_a)]
method_b_vals = [accuracy_score(y_true, y_pred_b), f1_score(y_true, y_pred_b),
                 precision_score(y_true, y_pred_b), recall_score(y_true, y_pred_b),
                 cohen_kappa_score(y_true, y_pred_b)]
method_c_vals = [accuracy_score(y_true, y_pred_c), f1_score(y_true, y_pred_c),
                 precision_score(y_true, y_pred_c), recall_score(y_true, y_pred_c),
                 cohen_kappa_score(y_true, y_pred_c)]

x = np.arange(len(metrics_names))
width = 0.25
bars_a = ax1.bar(x - width, method_a_vals, width, 
                  label='Method A\n(NLI)', color='#2196F3', alpha=0.85)
bars_b = ax1.bar(x, method_b_vals, width, 
                  label='Method B\n(Retrieval+LLM)', color='#4CAF50', alpha=0.85)
bars_c = ax1.bar(x + width, method_c_vals, width, 
                  label='Method C\n(Direct LLM)', color='#FF9800', alpha=0.85)

for bars in [bars_a, bars_b, bars_c]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=7)

ax1.set_xlabel('Metric')
ax1.set_ylabel('Score')
ax1.set_title('Figure 1: Overall Performance\nComparison Across Methods')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_names)
ax1.legend(loc='upper right', fontsize=7)
ax1.set_ylim(0, 1.15)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
ax1.grid(axis='y', alpha=0.3)

# ============================================================
# FIGURE 2: F1 by rarity — the central finding
# ============================================================
ax2 = axes[0, 1]
rarity_order = ["very rare", "rare", "medium", "freq", "very freq"]
halluc_rates = []
a_f1s, b_f1s, c_f1s = [], [], []

for rarity in rarity_order:
    mask = df["rarity"] == rarity
    halluc_rates.append((y_true[mask] == 0).mean())
    a_f1s.append(f1_score(y_true[mask], y_pred_a[mask]))
    b_f1s.append(f1_score(y_true[mask], y_pred_b[mask]))
    c_f1s.append(f1_score(y_true[mask], y_pred_c[mask]))

x = np.arange(len(rarity_order))
ax2.plot(x, a_f1s, 'o-', color='#2196F3', linewidth=2, 
         markersize=8, label='Method A (NLI)', zorder=3)
ax2.plot(x, b_f1s, 's-', color='#4CAF50', linewidth=2, 
         markersize=8, label='Method B (Retrieval+LLM)', zorder=3)
ax2.plot(x, c_f1s, '^-', color='#FF9800', linewidth=2, 
         markersize=8, label='Method C (Direct LLM)', zorder=3)

ax2_twin = ax2.twinx()
ax2_twin.bar(x, halluc_rates, alpha=0.2, color='red', label='Hallucination Rate')
ax2_twin.set_ylabel('Hallucination Rate', color='red', fontsize=9)
ax2_twin.tick_params(axis='y', labelcolor='red')
ax2_twin.set_ylim(0, 1.0)

ax2.set_xlabel('Entity Rarity')
ax2.set_ylabel('F1 Score')
ax2.set_title('Figure 2: F1 by Entity Rarity\nvs Hallucination Rate')
ax2.set_xticks(x)
ax2.set_xticklabels(['Very\nRare', 'Rare', 'Medium', 'Freq', 'Very\nFreq'], fontsize=8)
ax2.legend(loc='lower left', fontsize=7)
ax2.set_ylim(0, 1.0)
ax2.grid(alpha=0.3)

# ============================================================
# FIGURE 3: NLI threshold calibration
# ============================================================
ax3 = axes[0, 2]
entailment_scores = df["method_a_entailment"].values
thresholds = np.arange(0.05, 0.96, 0.05)
f1s, precs, recs = [], [], []

for t in thresholds:
    preds = (entailment_scores >= t).astype(int)
    if preds.sum() == 0:
        f1s.append(0); precs.append(0); recs.append(0)
        continue
    f1s.append(f1_score(y_true, preds))
    precs.append(precision_score(y_true, preds))
    recs.append(recall_score(y_true, preds))

ax3.plot(thresholds, f1s, 'b-', linewidth=2.5, label='F1', zorder=3)
ax3.plot(thresholds, precs, 'g--', linewidth=1.5, label='Precision', zorder=2)
ax3.plot(thresholds, recs, 'r--', linewidth=1.5, label='Recall', zorder=2)

best_t = thresholds[np.argmax(f1s)]
best_f1 = max(f1s)
ax3.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, label='Default (0.5)')
ax3.axvline(x=best_t, color='blue', linestyle=':', linewidth=1.5, 
            label=f'Optimal ({best_t:.2f})')
ax3.scatter([best_t], [best_f1], color='blue', s=100, zorder=5)
ax3.scatter([0.5], [f1_score(y_true, (entailment_scores>=0.5).astype(int))], 
            color='gray', s=100, zorder=5)

ax3.set_xlabel('Entailment Probability Threshold')
ax3.set_ylabel('Score')
ax3.set_title('Figure 3: NLI Threshold Calibration\n(Method A)')
ax3.legend(fontsize=8)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.grid(alpha=0.3)

# ============================================================
# FIGURE 4: F1 by LLM source
# ============================================================
ax4 = axes[1, 0]
llms = ["InstructGPT", "ChatGPT", "PerplexityAI"]
llm_halluc = []
a_llm, b_llm, c_llm = [], [], []

for llm in llms:
    mask = df["llm"] == llm
    llm_halluc.append((y_true[mask] == 0).mean())
    a_llm.append(f1_score(y_true[mask], y_pred_a[mask]))
    b_llm.append(f1_score(y_true[mask], y_pred_b[mask]))
    c_llm.append(f1_score(y_true[mask], y_pred_c[mask]))

x = np.arange(len(llms))
width = 0.25
ax4.bar(x - width, a_llm, width, label='Method A (NLI)', 
        color='#2196F3', alpha=0.85)
ax4.bar(x, b_llm, width, label='Method B (Retrieval+LLM)', 
        color='#4CAF50', alpha=0.85)
ax4.bar(x + width, c_llm, width, label='Method C (Direct LLM)', 
        color='#FF9800', alpha=0.85)

for i, (h_rate) in enumerate(llm_halluc):
    ax4.text(i, 0.02, f'Halluc:\n{h_rate:.1%}', ha='center', 
             fontsize=7.5, color='darkred', fontweight='bold')

ax4.set_xlabel('Source LLM')
ax4.set_ylabel('F1 Score')
ax4.set_title('Figure 4: F1 by Source LLM\n(with hallucination rates)')
ax4.set_xticks(x)
ax4.set_xticklabels(llms, fontsize=9)
ax4.legend(fontsize=8)
ax4.set_ylim(0, 0.85)
ax4.grid(axis='y', alpha=0.3)

# ============================================================
# FIGURE 5: Failure mode taxonomy — all wrong cases
# ============================================================
ax5 = axes[1, 1]
all_wrong_mask = (y_pred_a != y_true) & (y_pred_b != y_true) & (y_pred_c != y_true)

all_wrong_by_rarity = []
total_by_rarity = []
for rarity in rarity_order:
    r_mask = df["rarity"] == rarity
    all_wrong_by_rarity.append((all_wrong_mask & r_mask).sum())
    total_by_rarity.append(r_mask.sum())

all_wrong_rates = [a/t for a, t in zip(all_wrong_by_rarity, total_by_rarity)]

bars = ax5.bar(rarity_order, all_wrong_rates, 
               color=['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db'],
               alpha=0.85, edgecolor='black', linewidth=0.5)

for bar, count in zip(bars, all_wrong_by_rarity):
    ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
             f'n={count}', ha='center', va='bottom', fontsize=8)

ax5.set_xlabel('Entity Rarity')
ax5.set_ylabel('Rate of All-Methods Failure')
ax5.set_title('Figure 5: All-Methods Failure Rate\nby Entity Rarity')
ax5.set_xticklabels(['Very\nRare', 'Rare', 'Medium', 'Freq', 'Very\nFreq'])
ax5.set_ylim(0, 0.45)
ax5.grid(axis='y', alpha=0.3)

# ============================================================
# FIGURE 6: Prediction bias comparison
# ============================================================
ax6 = axes[1, 2]
categories = ['Method A\n(NLI)', 'Method B\n(Retrieval\n+LLM)', 
              'Method C\n(Direct LLM)', 'Human\nAnnotation']
supported_rates = [
    (y_pred_a == 1).mean(),
    (y_pred_b == 1).mean(),
    (y_pred_c == 1).mean(),
    (y_true == 1).mean()
]
unsupported_rates = [1 - r for r in supported_rates]

x = np.arange(len(categories))
bars1 = ax6.bar(x, supported_rates, label='Predicted Supported', 
                 color='#27ae60', alpha=0.85)
bars2 = ax6.bar(x, unsupported_rates, bottom=supported_rates, 
                 label='Predicted Unsupported', color='#e74c3c', alpha=0.85)

ax6.axhline(y=(y_true == 1).mean(), color='green', linestyle='--', 
            linewidth=1.5, alpha=0.7, label=f'Human S rate ({(y_true==1).mean():.1%})')

for bar, rate in zip(bars1, supported_rates):
    ax6.text(bar.get_x() + bar.get_width()/2., rate/2,
             f'{rate:.1%}', ha='center', va='center', 
             fontsize=9, color='white', fontweight='bold')

ax6.set_xlabel('Method')
ax6.set_ylabel('Proportion')
ax6.set_title('Figure 6: Prediction Bias\n(Supported vs Unsupported Rate)')
ax6.set_xticks(x)
ax6.set_xticklabels(categories, fontsize=8)
ax6.legend(fontsize=8, loc='lower right')
ax6.set_ylim(0, 1.0)
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("factuality_evaluation_figures.pdf", bbox_inches='tight', 
            dpi=300, format='pdf')
plt.savefig("factuality_evaluation_figures.png", bbox_inches='tight', dpi=300)
plt.show()
print("Figures saved as PDF and PNG")

print("TABLE 1: MAIN RESULTS")
print("="*70)
print(f"{'Method':<25} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Kappa':>7} {'Acc':>6}")
print("-"*70)

results = [
    ("A: NLI (BM25+RoBERTa)", y_pred_a),
    ("B: Retrieval+LLM (BM25+Qwen)", y_pred_b),
    ("C: Direct LLM (Qwen)", y_pred_c),
]

for name, preds in results:
    print(f"{name:<25} "
          f"{f1_score(y_true, preds):>6.3f} "
          f"{precision_score(y_true, preds):>6.3f} "
          f"{recall_score(y_true, preds):>6.3f} "
          f"{cohen_kappa_score(y_true, preds):>7.3f} "
          f"{accuracy_score(y_true, preds):>6.3f}")

print("-"*70)
print(f"\nOptimal NLI (threshold=0.1):")
opt_preds = (df["method_a_entailment"].values >= 0.1).astype(int)
print(f"{'A: NLI (threshold=0.1)':<25} "
      f"{f1_score(y_true, opt_preds):>6.3f} "
      f"{precision_score(y_true, opt_preds):>6.3f} "
      f"{recall_score(y_true, opt_preds):>6.3f} "
      f"{cohen_kappa_score(y_true, opt_preds):>7.3f} "
      f"{accuracy_score(y_true, opt_preds):>6.3f}")

print("\nTABLE 2: F1 BY ENTITY RARITY")
print("="*70)
print(f"{'Rarity':<12} {'Halluc%':>8} {'Method A':>9} "
      f"{'Method B':>9} {'Method C':>9} {'All-Wrong%':>11}")
print("-"*70)
for rarity in rarity_order:
    mask = df["rarity"] == rarity
    aw = ((y_pred_a != y_true) & (y_pred_b != y_true) & 
          (y_pred_c != y_true) & mask).sum() / mask.sum()
    print(f"{rarity:<12} "
          f"{(y_true[mask]==0).mean():>8.1%} "
          f"{f1_score(y_true[mask], y_pred_a[mask]):>9.3f} "
          f"{f1_score(y_true[mask], y_pred_b[mask]):>9.3f} "
          f"{f1_score(y_true[mask], y_pred_c[mask]):>9.3f} "
          f"{aw:>11.1%}")

print("\nTABLE 3: F1 BY SOURCE LLM")
print("="*70)
print(f"{'LLM':<15} {'Halluc%':>8} {'Method A':>9} "
      f"{'Method B':>9} {'Method C':>9}")
print("-"*70)
for llm in ["InstructGPT", "ChatGPT", "PerplexityAI"]:
    mask = df["llm"] == llm
    print(f"{llm:<15} "
          f"{(y_true[mask]==0).mean():>8.1%} "
          f"{f1_score(y_true[mask], y_pred_a[mask]):>9.3f} "
          f"{f1_score(y_true[mask], y_pred_b[mask]):>9.3f} "
          f"{f1_score(y_true[mask], y_pred_c[mask]):>9.3f}")