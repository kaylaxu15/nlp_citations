import json
import os
import re
import copy
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from transformers import pipeline

# -----------------------------
# CONFIG
# -----------------------------
RESULTS_DIR = "/content/drive/MyDrive/results"

# -----------------------------
# LOAD NLI MODEL
# -----------------------------
nli = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-large",
    device=0
)

# -----------------------------
# HELPERS
# -----------------------------
def run_nli(premise, hypothesis, threshold=0.5):
    result = nli(f"{premise} </s> {hypothesis}")[0]
    return 1 if (result["label"] == "ENTAILMENT" and result["score"] >= threshold) else 0


def extract_citations(sentence):
    """Extract [1][2] → [0,1]"""
    refs = re.findall(r"\[(\d+)\]", sentence)
    return [int(r) - 1 for r in refs]


def remove_citations(text):
    return re.sub(r"\[\d+\]", "", text).strip()


def format_doc(doc):
    return f"{doc.get('title', '')}\n{doc.get('text', '')}"


# -----------------------------
# MAIN EVAL FUNCTION
# -----------------------------
def evaluate_file(data):

    recall_scores = []
    precision_scores = []

    for item in tqdm(data):
        output = item["output"]
        docs = item.get("docs", [])

        sentences = sent_tokenize(output)

        sent_recalls = []
        sent_precisions = []

        for sent in sentences:
            clean_sent = remove_citations(sent)
            refs = extract_citations(sent)

            # -----------------------------
            # RECALL
            # -----------------------------
            if len(refs) == 0 or any(r >= len(docs) for r in refs):
                sent_recalls.append(0)
                sent_precisions.append(0)
                continue

            joint_passage = "\n".join([format_doc(docs[r]) for r in refs])

            recall = run_nli(joint_passage, clean_sent)
            sent_recalls.append(recall)

            # -----------------------------
            # PRECISION
            # -----------------------------
            if recall == 0:
                sent_precisions.append(0)
                continue

            correct_citations = 0

            for r in refs:
                # Check if single doc supports
                single_passage = format_doc(docs[r])
                single_entail = run_nli(single_passage, clean_sent)

                if single_entail:
                    correct_citations += 1
                    continue

                # Check if removing it still works
                subset = copy.deepcopy(refs)
                subset.remove(r)

                if len(subset) == 0:
                    continue

                subset_passage = "\n".join([format_doc(docs[s]) for s in subset])
                subset_entail = run_nli(subset_passage, clean_sent)

                if not subset_entail:
                    # necessary
                    correct_citations += 1
                # else: irrelevant → do not count

            precision = correct_citations / len(refs)
            sent_precisions.append(precision)

        # Aggregate per example
        if len(sentences) > 0:
            recall_scores.append(np.mean(sent_recalls))
            precision_scores.append(np.mean(sent_precisions))

    return {
        "citation_recall": 100 * np.mean(recall_scores),
        "citation_precision": 100 * np.mean(precision_scores)
    }


# -----------------------------
# RUN OVER FOLDER
# -----------------------------
def main():
    results = {}

    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(RESULTS_DIR, filename)

        with open(path) as f:
            raw = json.load(f)
            data = raw["data"] if isinstance(raw, dict) else raw

        print(f"\nEvaluating {filename}...")

        scores = evaluate_file(data)
        results[filename] = scores

        print(scores)

    with open("/content/drive/MyDrive/citation_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to citation_scores.json")


if __name__ == "__main__":
    main()