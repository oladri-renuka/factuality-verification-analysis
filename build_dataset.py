"""
Step 2: Build evaluation dataset from all three LLM JSONL files.
Input:  InstructGPT.jsonl, ChatGPT.jsonl, PerplexityAI.jsonl
Output: evaluation_dataset.csv
"""
import json
import pandas as pd

base_path = "labeled/"
files = {
    "InstructGPT": base_path + "InstructGPT.jsonl",
    "ChatGPT":     base_path + "ChatGPT.jsonl",
    "PerplexityAI": base_path + "PerplexityAI.jsonl"
}

rows  = []
stats = {}

for llm_name, file_path in files.items():
    total = skipped_records = skipped_annotations = fact_count = 0

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            total += 1

            if not data.get("annotations"):
                skipped_records += 1
                continue

            topic  = data["topic"]
            rarity = data.get("cat", ["unknown"])[0] if data.get("cat") else "unknown"

            for annotation in data["annotations"]:
                if not annotation.get("human-atomic-facts"):
                    skipped_annotations += 1
                    continue

                for haf in annotation["human-atomic-facts"]:
                    if haf["label"] == "IR":
                        continue

                    fact_count += 1
                    rows.append({
                        "llm":                  llm_name,
                        "topic":                topic,
                        "rarity":               rarity,
                        "sentence":             annotation["text"],
                        "atomic_fact":          haf["text"],
                        "human_label":          1 if haf["label"] == "S" else 0,
                        "method_a_pred":        None,
                        "method_b_pred":        None,
                        "method_c_pred":        None,
                        "retrieved_evidence":   None,
                        "method_a_confidence":  None,
                        "method_b_confidence":  None,
                        "method_c_confidence":  None,
                    })

    stats[llm_name] = {
        "total_biographies":  total,
        "skipped_records":    skipped_records,
        "skipped_annotations": skipped_annotations,
        "usable_facts":       fact_count,
    }

df = pd.DataFrame(rows)

# ── Print summary ─────────────────────────────────────────────────────────────
print("=" * 50)
print("PER-LLM STATISTICS")
print("=" * 50)
for llm, s in stats.items():
    print(f"\n{llm}:")
    print(f"  Biographies:         {s['total_biographies']}")
    print(f"  Skipped records:     {s['skipped_records']}")
    print(f"  Skipped annotations: {s['skipped_annotations']}")
    print(f"  Usable facts:        {s['usable_facts']}")

print("\n" + "=" * 50)
print("COMBINED DATAFRAME")
print("=" * 50)
print(f"Total rows: {len(df)}")

print(f"\nLabel distribution:")
print(df["human_label"].value_counts())

print(f"\nRarity distribution:")
print(df["rarity"].value_counts())

print(f"\nLLM distribution:")
print(df["llm"].value_counts())

rarity_order = ["very rare", "rare", "medium", "freq", "very freq"]
print(f"\nHallucination rate (NS%) by rarity:")
for rarity in rarity_order:
    subset = df[df["rarity"] == rarity]
    if len(subset) > 0:
        rate = (subset["human_label"] == 0).mean()
        print(f"  {rarity:12s}: {rate:.1%} hallucinated ({len(subset)} facts)")

print(f"\nHallucination rate by LLM:")
for llm in ["InstructGPT", "ChatGPT", "PerplexityAI"]:
    subset = df[df["llm"] == llm]
    if len(subset) > 0:
        rate = (subset["human_label"] == 0).mean()
        print(f"  {llm:15s}: {rate:.1%} hallucinated ({len(subset)} facts)")

df.to_csv("evaluation_dataset.csv", index=False)
print(f"\nSaved evaluation_dataset.csv  shape={df.shape}")
