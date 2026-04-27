"""
Sample 200 examples from qasa_eval.json (seed 42) and write to gold_questions.json.
Adds a "gold_passages" field containing the full doc entries whose id_suffix
matches the gold_ctxs list.
"""

import json
import random
import numpy as np

INPUT_FILE = "data/qasa_eval.json"
OUTPUT_FILE = "gold_questions.json"
N = 200
SEED = 42

with open(INPUT_FILE) as f:
    data = json.load(f)

rng_eval = np.random.default_rng(SEED)
eval_ids = rng_eval.choice(len(data), N, replace=False)
sample = [data[int(idx)] for idx in eval_ids]

for item in sample:
    gold_suffixes = {str(g) for g in item.get("gold_ctxs", [])}
    gold_passages = []
    for doc in item.get("docs", []):
        # id_suffix is display rank after prepare_qasa; recover original suffix from stable id
        doc_id = str(doc.get("id", ""))
        original_suffix = doc_id.rsplit("_", 1)[-1] if "_" in doc_id else doc_id
        if original_suffix in gold_suffixes:
            gold_passages.append(doc)
    item["gold_passages"] = gold_passages

with open(OUTPUT_FILE, "w") as f:
    json.dump(sample, f, indent=2)

print(f"Wrote {len(sample)} examples to {OUTPUT_FILE}")
missing = sum(1 for item in sample if not item["gold_passages"])
print(f"Examples with no gold passages found: {missing}")