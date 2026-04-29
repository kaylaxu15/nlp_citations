"""
Post-hoc citation for generations that were produced without citing passages (e.g. closed-book).

ALCE (Gao et al.) defines closed-book + post-hoc as: generate without retrieval, then attach citations.
They report strong correctness but weaker citation quality vs retrieval-augmented generation.

This script follows the repo pattern: segment the answer into sentences; for each sentence without
citations, score it against each candidate passage with GTR (or TF-IDF) and prefix the argmax passage as [k].

Closed-book result JSON rows often have docs=[]. Pass the corresponding retrieval-augmented run
(same eval order / questions), or any JSON with aligned rows and non-empty docs, via --external_docs / --docs_from.
"""

import json
import argparse
import sys
from tqdm import tqdm
from nltk import sent_tokenize
import re
import torch
from searcher import SearcherWithinDocs


def _effective_retriever_device(requested: str) -> str:
    """Use cuda only when PyTorch was built with CUDA and a device is available."""
    req = (requested or "cpu").strip().lower()
    if req.startswith("cuda") and not torch.cuda.is_available():
        print(
            "post_hoc_cite: CUDA requested but unavailable (CPU-only PyTorch or no GPU); using cpu.",
            file=sys.stderr,
        )
        return "cpu"
    return requested if requested else "cpu"


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def _load_data_rows(path):
    raw = json.load(open(path))
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    if isinstance(raw, list):
        return raw
    raise ValueError(f"{path}: expected JSON list or object with 'data' array")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Add citations to model outputs by sentence-level similarity to passages in doc_list. "
            "For closed-book runs with empty docs, pass --docs_from path to an aligned JSON that "
            "contains the same questions in the same order with retrieved passages (e.g. vanilla ndoc=3)."
        )
    )
    parser.add_argument("--f", type=str, required=True, help="Result JSON from run.py (list or {data: [...]})")
    parser.add_argument("--retriever", type=str, default="gtr-t5-large", help="`tfidf` or `gtr-t5-large`")
    parser.add_argument(
        "--retriever_device",
        type=str,
        default="cpu",
        help="Where to run GTR (dense retriever). Options: `cpu` (default), `cuda`",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing [n] citations per sentence")
    parser.add_argument(
        "--external_docs",
        "--docs_from",
        dest="external_docs",
        default=None,
        help="Aligned JSON with same rows as --f; each row's docs used as the passage pool (required if docs empty)",
    )

    args = parser.parse_args()
    device = _effective_retriever_device(args.retriever_device)

    raw_file = json.load(open(args.f))
    if isinstance(raw_file, dict) and "data" in raw_file:
        data_rows = raw_file["data"]
        preserve_wrapper = True
    elif isinstance(raw_file, list):
        data_rows = raw_file
        preserve_wrapper = False
    else:
        raise ValueError(f"{args.f}: expected JSON list or object with 'data' array")

    external_rows = _load_data_rows(args.external_docs) if args.external_docs else None
    if external_rows is not None and len(external_rows) != len(data_rows):
        raise ValueError(
            f"external_docs has {len(external_rows)} rows but --f has {len(data_rows)}; must align"
        )

    gtr_model = None
    if "gtr" in args.retriever:
        from sentence_transformers import SentenceTransformer

        gtr_model = SentenceTransformer(f"sentence-transformers/{args.retriever}", device=device)

    new_data = []
    for idx, item in enumerate(tqdm(data_rows)):
        doc_list = item.get("docs") or []
        if external_rows is not None:
            ext = external_rows[idx]
            if ext["question"] != item["question"]:
                raise ValueError(
                    f"Row {idx}: question mismatch between --f and external_docs:\n"
                    f"  {item['question'][:120]!r}\nvs\n  {ext['question'][:120]!r}"
                )
            doc_list = ext.get("docs") or []
        if not doc_list:
            raise ValueError(
                f"Row {idx}: no passages to cite (docs empty). "
                "Pass --docs_from / --external_docs to a JSON with aligned retrieved passages "
                "(e.g. vanilla shot1 ndoc3 run on the same quick_test indices)."
            )

        searcher = SearcherWithinDocs(doc_list, args.retriever, model=gtr_model, device=device)

        output = item["output"].replace("<|im_end|>", "")
        if "qampari" in args.f:
            sents = [item["question"] + " " + x.strip() for x in item["output"].rstrip(".").split(",")]
        else:
            sents = sent_tokenize(output)

        new_output = ""
        for sent in sents:
            original_ref = [int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)]

            if len(original_ref) == 0 or args.overwrite:
                sent_clean = remove_citations(sent)
                best_doc_id = searcher.search(sent_clean)
                sent = f"[{best_doc_id + 1}] " + sent_clean
                if "qampari" in args.f:
                    new_output += sent.replace(item["question"], "").strip() + ", "
                else:
                    new_output += sent + " "
            else:
                if "qampari" in args.f:
                    new_output += sent.replace(item["question"], "").strip() + ", "
                else:
                    new_output += sent + " "

        item["output"] = new_output.rstrip().rstrip(",")
        item["docs"] = doc_list
        new_data.append(item)

    if preserve_wrapper:
        raw_file["data"] = new_data
        out_obj = raw_file
    else:
        out_obj = new_data

    tag = f".{args.retriever}"
    if args.overwrite:
        tag += "-overwrite"
    if args.external_docs is not None:
        tag += "-external"

    out_path = args.f + f".post_hoc_cite{tag}"
    json.dump(out_obj, open(out_path, "w"), indent=4)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
