#!/usr/bin/env python3
"""
Rank passages *within one paper* by similarity to a question.

Note: ``retrieval.py`` ``gtr_wiki_retrieval`` searches English Wikipedia (DPR TSV),
not chunks of an arXiv paper. For QASA demos you want this script (or SearcherWithinDocs
extended to top-k) instead.

Uses the same encoding convention as ``searcher.py``: title + ". " + text,
with sentence-transformers GTR when available; falls back to TF-IDF.

Example:
  python tools/qasa_intrapaper_topk.py \\
    --chunks_json data/highway_paper_chunks.json \\
    --question "What are the differences between plain networks and highway networks?" \\
    --k 5 \\
    --gold-global-indices 17,23,25
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Sequence, Tuple


def doc_to_text_dense(doc: Dict[str, Any]) -> str:
    return doc["title"] + ". " + doc["text"]


def rank_gtr(
    question: str,
    chunks: Sequence[Dict[str, Any]],
    model_name: str,
    device: str | None,
    model=None,
) -> Tuple[List[int], Any]:
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = SentenceTransformer(model_name, device=device)
    texts = [doc_to_text_dense(c) for c in chunks]
    with torch.inference_mode():
        q_emb = model.encode(
            [question],
            device=device,
            convert_to_numpy=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        d_emb = model.encode(
            texts,
            device=device,
            convert_to_numpy=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=8,
            show_progress_bar=len(texts) > 32,
        )
        scores = torch.matmul(d_emb, q_emb.t()).squeeze(1).detach().cpu().numpy()
    order = list(range(len(chunks)))
    order.sort(key=lambda i: float(scores[i]), reverse=True)
    return order, scores


def rank_tfidf(question: str, chunks: Sequence[Dict[str, Any]]) -> Tuple[List[int], Any]:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [doc_to_text_dense(c) for c in chunks]
    vec = TfidfVectorizer(max_features=16384)
    doc_m = vec.fit_transform(texts + [question])
    qv = doc_m[-1]
    dv = doc_m[:-1]
    sim = cosine_similarity(dv, qv).ravel()
    order = list(np.argsort(-sim))
    return order, sim


def main() -> None:
    p = argparse.ArgumentParser(description="Top-k passages within one document set.")
    p.add_argument("--chunks_json", type=str, required=True, help="JSON array of {id,title,text}")
    p.add_argument("--question", type=str, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument(
        "--gold-global-indices",
        type=str,
        default=None,
        help="Comma-separated 0-based indices into chunks_json array for gold passages",
    )
    p.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/gtr-t5-large",
        help="sentence-transformers model (try gtr-t5-large to match interactive search defaults)",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--tfidf", action="store_true", help="Use TF-IDF instead of GTR")
    args = p.parse_args()

    with open(args.chunks_json) as f:
        chunks = json.load(f)
    if not isinstance(chunks, list):
        sys.exit("chunks_json must be a JSON array")

    gold_idx: List[int] = []
    if args.gold_global_indices:
        gold_idx = [int(x.strip()) for x in args.gold_global_indices.split(",")]

    if args.tfidf:
        order, scores = rank_tfidf(args.question, chunks)
        method = "tfidf"
    else:
        try:
            order, scores = rank_gtr(args.question, chunks, args.model, args.device, model=None)
            method = args.model
        except Exception as e:
            print(f"# GTR failed ({e}); falling back to TF-IDF", file=sys.stderr)
            order, scores = rank_tfidf(args.question, chunks)
            method = "tfidf (fallback)"

    topk = order[: args.k]
    print(f"# rank_method: {method}", file=sys.stderr)
    print("# rank  global_idx  score  id", file=sys.stderr)
    for r, gi in enumerate(topk):
        sc = float(scores[gi])
        cid = chunks[gi].get("id", "")
        print(f"# {r+1:3d}  {gi:5d}  {sc:.5f}  {cid}", file=sys.stderr)

    # Map old global index -> new 1-based citation in prompt order (top-k list order)
    remap: Dict[int, int] = {}
    for new_pos, gi in enumerate(topk):
        remap[gi] = new_pos + 1

    print("\n# ALCE-style docs (top-k, title + text only):", file=sys.stderr)
    out_docs = [{"title": chunks[i]["title"], "text": chunks[i]["text"]} for i in topk]
    print(json.dumps(out_docs, indent=2))

    if gold_idx:
        print("\n# Gold global indices -> new citation brackets:", file=sys.stderr)
        for g in gold_idx:
            new_c = remap.get(g)
            if new_c is None:
                print(f"#   global {g}: NOT IN TOP-{args.k} — expand k or merge gold-first", file=sys.stderr)
            else:
                print(f"#   global {g} -> [{new_c}]", file=sys.stderr)


if __name__ == "__main__":
    main()
