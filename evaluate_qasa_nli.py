import json
import os
import re
import copy
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import nltk
nltk.download('punkt_tab')

# -----------------------------
# CONFIG
# -----------------------------
RESULTS_DIR = "/content/drive/MyDrive/results"
N_EXAMPLES = 5  # how many good/bad examples to save per file

# -----------------------------
# LOAD NLI MODEL
# -----------------------------
nli = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    device=-1,
    truncation=True,
    max_length=1024,
)

# -----------------------------
# HELPERS
# -----------------------------
def run_nli(premise, hypothesis, threshold=0.5):
    result = nli(f"{premise} {hypothesis}")[0]
    return 1 if (result["label"].lower() == "entailment" and result["score"] >= threshold) else 0


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
    per_item_scores = []  # track per-item for example extraction

    for item in tqdm(data):
        output = item["output"]

        if not output or not output.strip():
          continue 
        
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
            item_recall = float(np.mean(sent_recalls))
            item_precision = float(np.mean(sent_precisions))
            recall_scores.append(item_recall)
            precision_scores.append(item_precision)
            per_item_scores.append({
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "output": output,
                "docs": [{"title": d.get("title", ""), "text": d.get("text", "")} for d in docs],
                "citation_recall": item_recall,
                "citation_precision": item_precision,
            })

    # Sort by recall for good/bad examples
    sorted_by_recall = sorted(per_item_scores, key=lambda x: x["citation_recall"])
    bad_examples = sorted_by_recall[:N_EXAMPLES]
    good_examples = sorted_by_recall[-N_EXAMPLES:][::-1]

    return {
        "citation_recall": 100 * np.mean(recall_scores),
        "citation_precision": 100 * np.mean(precision_scores),
        "good_examples": good_examples,
        "bad_examples": bad_examples,
    }


# -----------------------------
# RUN OVER FOLDER
# -----------------------------
def main():
    results = {}
    examples_out = {}

    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(RESULTS_DIR, filename)

        with open(path) as f:
            raw = json.load(f)
            data = raw["data"] if isinstance(raw, dict) else raw

        print(f"\nEvaluating {filename}...")

        scores = evaluate_file(data)

        # Separate scores from examples for the scores file
        results[filename] = {
            "citation_recall": scores["citation_recall"],
            "citation_precision": scores["citation_precision"],
        }
        examples_out[filename] = {
            "good_examples": scores["good_examples"],
            "bad_examples": scores["bad_examples"],
        }

        print({
            "citation_recall": scores["citation_recall"],
            "citation_precision": scores["citation_precision"],
        })

    with open("/content/drive/MyDrive/citation_scores_bart.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("/content/drive/MyDrive/citation_examples_bart.json", "w") as f:
        json.dump(examples_out, f, indent=2, ensure_ascii=False)

    print("\nSaved scores to citation_scores_bart.json")
    print("Saved examples to citation_examples_bart.json")


if __name__ == "__main__":
    main()