import argparse
import json
import os
import random
import re
from typing import Any, Dict, List, Set

from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_INSTRUCTION = (
    "Instruction: Write an accurate and concise answer for the given question using only the "
    "provided search results. Always cite factual claims with bracket citations such as [1][2]. "
    "Use at least one citation per sentence and at most three citations per sentence."
)


def normalize_answer_text(text: str, ctxs: List[Dict[str, Any]] = None, docs: List[Dict[str, str]] = None) -> str:
    """
    Preserves citations from the gold answer, remapping their indices to match
    the new document ordering in `docs`. If no mapping info is provided, citations
    are stripped as before.
    """
    if ctxs is None or docs is None:
        return re.sub(r"\[\d+\]", "", text).replace("  ", " ").strip()

    # Build a mapping from original ctx position (1-indexed) -> new doc position (1-indexed)
    # ctxs is the original ordered list; docs is the reordered subset used in the prompt.
    original_titles = [str(c.get("title", "")).strip() or "Untitled" for c in ctxs]
    new_titles = [d["title"] for d in docs]

    index_map: Dict[int, int] = {}
    for old_idx, title in enumerate(original_titles, start=1):
        if title in new_titles:
            new_idx = new_titles.index(title) + 1  # 1-indexed
            index_map[old_idx] = new_idx

    def remap(match: re.Match) -> str:
        old_idx = int(match.group(1))
        new_idx = index_map.get(old_idx)
        return f"[{new_idx}]" if new_idx is not None else ""  # drop if doc was not retained

    return re.sub(r"\[(\d+)\]", remap, text).replace("  ", " ").strip()


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True).strip()


def is_gold_ctx(ctx_id: str, gold_ctx_ids: Set[str]) -> bool:
    suffix = ctx_id.rsplit("_", 1)[-1] if "_" in ctx_id else ctx_id
    return ctx_id in gold_ctx_ids or suffix in gold_ctx_ids


def build_docs(
    ctxs: List[Dict[str, Any]],
    gold_ctx_ids: Set[str],
    ndoc_total: int,
    put_gold_first: bool,
    rng: random.Random,  # kept for compatibility; unused
) -> List[Dict[str, str]]:
    cleaned = []
    for c in ctxs:
        cid = str(c.get("id", ""))
        title = str(c.get("title", "")).strip() or "Untitled"
        text = str(c.get("text", "")).strip()
        if not text:
            continue

        # QASA gold ids are often short numeric ids (e.g., "16"),
        # while ctx ids may be full ids (e.g., "1707.01083_all_16").
        cid_suffix = cid.rsplit("_", 1)[-1] if "_" in cid else cid
        is_gold = (cid in gold_ctx_ids) or (cid_suffix in gold_ctx_ids)

        cleaned.append(
            {
                "id": cid,
                "id_suffix": cid_suffix,
                "title": title,
                "text": text,
                "is_gold": is_gold,
            }
        )

    if not cleaned:
        return []

    if put_gold_first:
        # Preserve original order within each subset.
        gold = [c for c in cleaned if c["is_gold"]]
        non_gold = [c for c in cleaned if not c["is_gold"]]
        ordered = gold + non_gold
    else:
        # Keep exact incoming order (retrieval order).
        ordered = cleaned

    return [{"title": c["title"], "text": c["text"]} for c in ordered[:ndoc_total]]


def build_item(
    row: Dict[str, Any],
    ndoc_total: int,
    put_gold_first: bool,
    rng: random.Random,
    tokenizer,
    doc_token_budget: int,
    max_tokens_per_doc: int,
) -> Dict[str, Any]:
    question = str(row.get("input", "")).strip()
    raw_ctxs = row.get("ctxs") or []

    gold_ctx_ids = {str(x) for x in (row.get("gold_ctxs") or [])}
    docs = build_docs(
        ctxs=raw_ctxs,
        gold_ctx_ids=gold_ctx_ids,
        ndoc_total=ndoc_total,
        put_gold_first=put_gold_first,
        rng=rng,
    )

    # Pass raw_ctxs and docs so citations can be remapped to the new ordering
    answer = normalize_answer_text(
        str(row.get("answer", "")).strip(),
        ctxs=raw_ctxs,
        docs=docs,
    )

    return {
        "question": question,
        "answer": answer,
        "docs": docs,
        "gold_ctxs": sorted(list(gold_ctx_ids)),
    }


def build_prompt_data(demos: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "instruction": DEFAULT_INSTRUCTION,
        "demo_sep": "\n\n\n",
        "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
        "doc_prompt": "Document [{ID}](Title: {T}): {P}\n",
        "demos": demos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ALCE-compatible prompt/eval JSON from QASA")
    parser.add_argument("--dataset_name", type=str, default="LinerAI/QASA")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_demos", type=int, default=8)
    parser.add_argument("--max_eval_examples", type=int, default=None)

    parser.add_argument("--ndoc_total", type=int, default=5, help="Max docs retained per sample")
    parser.add_argument("--doc_token_budget", type=int, default=320, help="Token budget for all docs in one sample")
    parser.add_argument("--max_tokens_per_doc", type=int, default=140, help="Token cap per individual document")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")

    parser.add_argument(
        "--put_gold_first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Place gold contexts first before adding distractors.",
    )

    parser.add_argument("--prompt_output", type=str, default="data/qasa_prompt.json")
    parser.add_argument("--eval_output", type=str, default="data/qasa_eval.json")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name)[args.split]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)
    rng = random.Random(args.seed)

    items = []
    for row in dataset:
        item = build_item(
            row=row,
            ndoc_total=args.ndoc_total,
            put_gold_first=args.put_gold_first,
            rng=rng,
            tokenizer=tokenizer,
            doc_token_budget=args.doc_token_budget,
            max_tokens_per_doc=args.max_tokens_per_doc,
        )
        if not item["question"] or not item["answer"] or len(item["docs"]) == 0:
            continue
        items.append(item)

    if len(items) <= args.num_demos:
        raise ValueError(f"Need more than {args.num_demos} valid items, found {len(items)}")

    rng.shuffle(items)
    demos = items[: args.num_demos]
    eval_data = items[args.num_demos :]
    if args.max_eval_examples is not None:
        eval_data = eval_data[: args.max_eval_examples]

    os.makedirs(os.path.dirname(args.prompt_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.eval_output), exist_ok=True)

    with open(args.prompt_output, "w") as f:
        json.dump(build_prompt_data(demos), f, indent=2)
    with open(args.eval_output, "w") as f:
        json.dump(eval_data, f, indent=2)

    print(
        f"Wrote {len(demos)} demos to {args.prompt_output} and "
        f"{len(eval_data)} eval examples to {args.eval_output}"
    )


if __name__ == "__main__":
    main()