"""
Step 1: Explore FActScore dataset
Run this first to understand the data structure.
Input:  FActScore labeled JSONL files
Output: Prints statistics only (no files saved)
"""
import json

file_path = "labeled/InstructGPT.jsonl"

total_records = 0
total_human_facts = 0
label_counts = {"S": 0, "NS": 0, "IR": 0}
rarity_counts = {}
skipped_annotations = 0
skipped_records = 0

with open(file_path, "r") as f:
    for line in f:
        data = json.loads(line)
        total_records += 1

        cat = data.get("cat", ["unknown"])
        cat = cat[0] if cat else "unknown"
        rarity_counts[cat] = rarity_counts.get(cat, 0) + 1

        if not data["annotations"]:
            skipped_records += 1
            continue

        for annotation in data["annotations"]:
            if not annotation.get("human-atomic-facts"):
                skipped_annotations += 1
                continue

            for haf in annotation["human-atomic-facts"]:
                total_human_facts += 1
                label_counts[haf["label"]] += 1

print(f"Total biographies:                        {total_records}")
print(f"Skipped records (annotations=None):       {skipped_records}")
print(f"Skipped annotations (haf=None):           {skipped_annotations}")
print(f"Total human atomic facts:                 {total_human_facts}")
print(f"Label distribution:                       {label_counts}")
print(f"Rarity distribution:                      {rarity_counts}")
