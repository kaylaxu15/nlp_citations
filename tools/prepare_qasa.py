import argparse
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Set

from datasets import load_dataset
from transformers import AutoTokenizer

# When run as `python tools/prepare_qasa.py`, allow sibling import
_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

try:
    from qasa_intrapaper_topk import rank_gtr, rank_tfidf
except ImportError:
    from tools.qasa_intrapaper_topk import rank_gtr, rank_tfidf  # type: ignore

DEFAULT_INSTRUCTION = (
    "Instruction: Write an accurate and concise answer for the given question using only the "
    "provided search results. Always cite factual claims with bracket citations such as [1][2]. "
    "Use at least one citation per sentence and at most three citations per sentence."
)

def has_too_many_citations(answer: str, max_citations: int = 3) -> bool:
    return len(re.findall(r"\[\d+\]", answer)) > max_citations


def normalize_answer_text(
    text: str, ctxs: Optional[List[Dict[str, Any]]] = None, docs: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Remap [k] citations to 1..len(docs) in **display order**.
    QASA references may use either (a) 1-based index into full ctxs ``source_idx`` space,
    or (b) passage id suffix (e.g. ``..._all_12`` → ``12``). Try source_idx first, then suffix.
    """
    if ctxs is None or docs is None:
        return re.sub(r"\[\d+\]", "[NA]", text).replace("  ", " ").strip()

    if docs and isinstance(docs[0], dict) and "source_idx" in docs[0]:
        by_source_idx: Dict[int, int] = {}
        by_suffix: Dict[str, int] = {}
        for new_i, doc in enumerate(docs, start=1):
            si = doc.get("source_idx")
            if si is not None:
                by_source_idx[int(si)] = new_i
            suf = doc.get("id_suffix")
            if suf is not None and str(suf).strip():
                by_suffix[str(suf).strip()] = new_i

        def remap(match: re.Match) -> str:
            raw = match.group(1)
            old_idx = int(raw)
            new_idx = by_source_idx.get(old_idx)
            if new_idx is not None:
                return f"[{new_idx}]"
            new_idx = by_suffix.get(str(old_idx))
            if new_idx is not None:
                return f"[{new_idx}]"
            new_idx = by_suffix.get(raw)
            if new_idx is not None:
                return f"[{new_idx}]"
            return "[NA]"

        return re.sub(r"\[(\d+)\]", remap, text).replace("  ", " ").strip()

    # Legacy: match by title
    original_titles = [str(c.get("title", "")).strip() or "Untitled" for c in ctxs]
    new_titles = [d["title"] for d in docs]

    index_map = {}
    for old_idx, title in enumerate(original_titles, start=1):
        if title in new_titles:
            new_idx = new_titles.index(title) + 1
            index_map[old_idx] = new_idx

    def remap(match: re.Match) -> str:
        old_idx = int(match.group(1))
        new_idx = index_map.get(old_idx)
        return f"[{new_idx}]" if new_idx is not None else "[NA]"

    return re.sub(r"\[(\d+)\]", remap, text).replace("  ", " ").strip()


def _doc_row_from_cleaned(c: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": c["title"],
        "text": c["text"],
        "source_idx": c["source_idx"],
        "id": c["id"],
        "id_suffix": c["id_suffix"],
        "is_gold": c["is_gold"],
    }


def _pad_docs_to_k(cur: List[Dict[str, Any]], cleaned: List[Dict[str, Any]], k: int) -> None:
    """Append unused passages from ``cleaned`` until ``cur`` has length ``k`` (small paper / short rank)."""
    present = {d["source_idx"] for d in cur}
    for c in cleaned:
        if len(cur) >= k:
            return
        if c["source_idx"] not in present:
            cur.append(_doc_row_from_cleaned(c))
            present.add(c["source_idx"])


def ensure_gold_passages_in_topk(
    docs: List[Dict[str, Any]],
    cleaned: List[Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    """
    After dense retrieval, gold passages cited in ``gold_ctxs`` may fall outside the top-k.
    Swap in any missing gold passages by evicting non-gold from the tail (worst-ranked slots).
    Keeps list length at most k.
    """
    if not docs or k <= 0:
        return docs

    gold_rows = [c for c in cleaned if c["is_gold"]]
    if not gold_rows:
        out = [dict(d) for d in docs[:k]]
        _pad_docs_to_k(out, cleaned, k)
        return out

    cur: List[Dict[str, Any]] = [dict(d) for d in docs[:k]]
    _pad_docs_to_k(cur, cleaned, k)
    present = {d["source_idx"] for d in cur}

    def add_missing(g: Dict[str, Any]) -> None:
        nonlocal cur, present
        row = _doc_row_from_cleaned(g)
        if row["source_idx"] in present:
            return
        if len(cur) < k:
            cur.append(row)
            present.add(row["source_idx"])
            return
        for j in range(len(cur) - 1, -1, -1):
            if not cur[j].get("is_gold", False):
                old_sid = cur[j]["source_idx"]
                cur[j] = row
                present.discard(old_sid)
                present.add(row["source_idx"])
                return

    for g in gold_rows:
        add_missing(g)

    return cur[:k]


def apply_display_rank_suffixes(docs: List[Dict[str, Any]]) -> None:
    """Set ``id_suffix`` to 1..n (display/citation rank); keep stable ``id`` unchanged."""
    for i, d in enumerate(docs, start=1):
        d["id_suffix"] = str(i)


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True).strip()


def collect_cleaned_ctxs(ctxs: List[Dict[str, Any]], gold_ctx_ids: Set[str]) -> List[Dict[str, Any]]:
    """
    One entry per non-empty passage. ``source_idx`` = 1-based index in the original ``ctxs`` list
    (matches typical QASA bracket citations).
    """
    cleaned = []
    for pos, c in enumerate(ctxs, start=1):
        cid = str(c.get("id", ""))
        title = str(c.get("title", "")).strip() or "Untitled"
        text = str(c.get("text", "")).strip()
        if not text:
            continue

        cid_suffix = cid.rsplit("_", 1)[-1] if "_" in cid else cid
        is_gold = (cid in gold_ctx_ids) or (cid_suffix in gold_ctx_ids)

        cleaned.append(
            {
                "id": cid,
                "id_suffix": cid_suffix,
                "title": title,
                "text": text,
                "is_gold": is_gold,
                "source_idx": pos,
            }
        )
    return cleaned


def strip_internal_doc_fields(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ALCE-style docs on disk: title + text + optional stable QASA passage ids (like prompt demos)."""
    out: List[Dict[str, Any]] = []
    for d in docs:
        row: Dict[str, Any] = {"title": d["title"], "text": d["text"]}
        cid = d.get("id")
        if cid:
            row["id"] = str(cid)
        suf = d.get("id_suffix")
        if suf not in (None, ""):
            row["id_suffix"] = str(suf)
        out.append(row)
    return out


def rank_select_topk(
    question: str,
    cleaned: List[Dict[str, Any]],
    topk: int,
    ranker: str,
    gtr_model_name: str,
    gtr_device: Optional[str],
    gtr_model=None,
) -> List[Dict[str, Any]]:
    """Sort all passages by relevance to ``question``; keep the top-``topk`` (with metadata)."""
    if len(cleaned) <= topk:
        return [_doc_row_from_cleaned(c) for c in cleaned]

    chunks = [{"title": c["title"], "text": c["text"]} for c in cleaned]

    if ranker == "tfidf":
        order, _scores = rank_tfidf(question, chunks)
    elif ranker == "gtr":
        order, _scores = rank_gtr(
            question, chunks, gtr_model_name, gtr_device, model=gtr_model
        )
    else:
        raise ValueError(f"Unknown ranker: {ranker}")

    return [_doc_row_from_cleaned(cleaned[i]) for i in order[:topk]]


def build_docs_ordered(
    cleaned: List[Dict[str, Any]],
    gold_ctx_ids: Set[str],
    ndoc_total: int,
    put_gold_first: bool,
) -> List[Dict[str, Any]]:
    if not cleaned:
        return []

    if put_gold_first:
        gold = [c for c in cleaned if c["is_gold"]]
        non_gold = [c for c in cleaned if not c["is_gold"]]
        ordered = gold + non_gold
    else:
        ordered = cleaned

    selected = ordered[:ndoc_total]
    return [_doc_row_from_cleaned(c) for c in selected]


def build_item(
    row: Dict[str, Any],
    ndoc_total: int,
    put_gold_first: bool,
    rng: random.Random,
    tokenizer,
    doc_token_budget: int,
    max_tokens_per_doc: int,
    *,
    rank_passages: bool,
    ranker: str,
    gtr_model_name: str,
    gtr_device: Optional[str],
    gtr_model=None,
    ensure_gold_in_topk: bool = True,
) -> Dict[str, Any]:
    question = str(row.get("input", "")).strip()
    raw_ctxs = row.get("ctxs") or []

    gold_ctx_ids = {str(x) for x in (row.get("gold_ctxs") or [])}
    cleaned = collect_cleaned_ctxs(raw_ctxs, gold_ctx_ids)

    if rank_passages:
        docs_full = rank_select_topk(
            question,
            cleaned,
            ndoc_total,
            ranker=ranker,
            gtr_model_name=gtr_model_name,
            gtr_device=gtr_device,
            gtr_model=gtr_model,
        )
    else:
        docs_full = build_docs_ordered(
            cleaned,
            gold_ctx_ids,
            ndoc_total,
            put_gold_first,
        )

    if ensure_gold_in_topk:
        docs_full = ensure_gold_passages_in_topk(docs_full, cleaned, ndoc_total)

    answer = normalize_answer_text(
        str(row.get("answer", "")).strip(),
        ctxs=raw_ctxs,
        docs=docs_full,
    )

    apply_display_rank_suffixes(docs_full)

    docs_out = strip_internal_doc_fields(docs_full)

    return {
        "question": question,
        "answer": answer,
        "docs": docs_out,
        "gold_ctxs": sorted(list(gold_ctx_ids)),
    }


def build_prompt_data(demos: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "instructions": [DEFAULT_INSTRUCTION],
        "instruction": DEFAULT_INSTRUCTION,
        "demo_sep": "\n\n\n",
        "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
        "doc_prompt": "Document [{ID}](Title: {T}): {P}\n",
        "demos": demos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare ALCE-compatible prompt/eval JSON from QASA. "
            "By default, ranks all paper passages by GTR similarity to the question and keeps top-k."
        )
    )
    parser.add_argument("--dataset_name", type=str, default="LinerAI/QASA")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_demos", type=int, default=8)
    parser.add_argument(
        "--max_eval_examples",
        type=int,
        default=None,
        help=(
            "Cap eval JSON size (non-eval_only: keeps first num_demos shuffle slots for demos). "
            "Also stops scanning the HF split early once enough valid rows are collected "
            "(num_demos + max_eval_examples in prompt+eval mode; max_eval_examples in --eval_only)."
        ),
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="How many passages to keep per example after ranking (same role as former ndoc_total). Default 3.",
    )
    parser.add_argument("--ndoc_total", type=int, default=None, help="Alias for --topk (backward compat).")
    parser.add_argument(
        "--max_gold_ctxs",
        type=int,
        default=3,
        help="Skip HF rows whose gold_ctxs list has more than this many passage ids (default 3).",
    )
    parser.add_argument(
        "--ensure_gold_in_topk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After ranking, swap in missing gold passages by dropping non-gold tail slots (default: on).",
    )
    parser.add_argument("--doc_token_budget", type=int, default=320, help="Reserved for future truncation.")
    parser.add_argument("--max_tokens_per_doc", type=int, default=140, help="Reserved for future truncation.")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")

    parser.add_argument(
        "--rank_passages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rank passages with GTR/TF-IDF over full ctxs and keep top-k (default: true).",
    )
    parser.add_argument(
        "--ranker",
        type=str,
        choices=("gtr", "tfidf"),
        default="gtr",
        help="Dense GTR (sentence-transformers) or TF-IDF fallback (CPU, fast).",
    )
    parser.add_argument(
        "--gtr_model",
        type=str,
        default="sentence-transformers/gtr-t5-large",
        help="sentence-transformers model id when --ranker gtr.",
    )
    parser.add_argument(
        "--gtr_device",
        type=str,
        default=None,
        help="cuda / cpu / cuda:0 (default: cuda if available else cpu).",
    )

    parser.add_argument(
        "--put_gold_first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only used when --no-rank-passages: order gold passages before others, then truncate.",
    )

    parser.add_argument(
        "--max_dataset_examples",
        type=int,
        default=None,
        help="Process only the first N rows from the HF split (debug / faster iteration).",
    )

    parser.add_argument("--prompt_output", type=str, default="data/qasa_prompt.json")
    parser.add_argument("--eval_output", type=str, default="data/qasa_eval.json")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only write eval JSON (no prompt file). Demos come from your --prompt_file at run time. "
        "Uses --max_eval_examples examples after shuffle (default 10 if unset).",
    )
    args = parser.parse_args()

    topk = args.ndoc_total if args.ndoc_total is not None else args.topk

    dataset = load_dataset(args.dataset_name)[args.split]
    if args.max_dataset_examples is not None:
        dataset = dataset.select(range(min(args.max_dataset_examples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)
    rng = random.Random(args.seed)

    gtr_model = None
    if args.rank_passages and args.ranker == "gtr":
        import torch
        from sentence_transformers import SentenceTransformer

        dev = args.gtr_device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading GTR encoder {args.gtr_model} on {dev}...")
        gtr_model = SentenceTransformer(args.gtr_model, device=dev)

    collect_cap: Optional[int] = None
    if args.eval_only:
        if args.max_eval_examples is not None:
            collect_cap = args.max_eval_examples
    elif args.max_eval_examples is not None:
        collect_cap = args.num_demos + args.max_eval_examples

    items = []
    for row in dataset:
        gcs = row.get("gold_ctxs") or []
        if len(gcs) > args.max_gold_ctxs:
            continue
        item = build_item(
            row=row,
            ndoc_total=topk,
            put_gold_first=args.put_gold_first,
            rng=rng,
            tokenizer=tokenizer,
            doc_token_budget=args.doc_token_budget,
            max_tokens_per_doc=args.max_tokens_per_doc,
            rank_passages=args.rank_passages,
            ranker=args.ranker,
            gtr_model_name=args.gtr_model,
            gtr_device=args.gtr_device,
            gtr_model=gtr_model,
            ensure_gold_in_topk=args.ensure_gold_in_topk,
        )
        if not item["question"] or not item["answer"] or len(item["docs"]) == 0:
            continue
        if has_too_many_citations(item["answer"]):
            continue
        items.append(item)
        if collect_cap is not None and len(items) >= collect_cap:
            print(
                f"Reached --max_eval_examples pipeline cap ({collect_cap} valid rows); stopping HF scan early.",
                file=sys.stderr,
            )
            break

    if args.eval_only:
        n_eval = args.max_eval_examples if args.max_eval_examples is not None else 10
        if len(items) < n_eval:
            raise ValueError(
                f"--eval_only needs at least {n_eval} valid items after filtering, found {len(items)}"
            )
        rng.shuffle(items)
        eval_data = items[:n_eval]
        demos = []
    else:
        if args.max_eval_examples is not None:
            needed = args.num_demos + args.max_eval_examples
            if len(items) < needed:
                raise ValueError(
                    f"--max_eval_examples={args.max_eval_examples} requires at least "
                    f"{needed} valid items (num_demos={args.num_demos} + eval cap), found {len(items)}."
                )
        elif len(items) <= args.num_demos:
            raise ValueError(f"Need more than {args.num_demos} valid items, found {len(items)}")

        rng.shuffle(items)
        demos = items[: args.num_demos]
        eval_data = items[args.num_demos :]
        if args.max_eval_examples is not None:
            eval_data = eval_data[: args.max_eval_examples]

    os.makedirs(os.path.dirname(args.eval_output) or ".", exist_ok=True)

    if not args.eval_only:
        os.makedirs(os.path.dirname(args.prompt_output) or ".", exist_ok=True)
        with open(args.prompt_output, "w") as f:
            json.dump(build_prompt_data(demos), f, indent=2)

    with open(args.eval_output, "w") as f:
        json.dump(eval_data, f, indent=2)

    mode = f"rank={args.rank_passages} ({args.ranker}, topk={topk})" if args.rank_passages else "ordered (no rank)"
    if args.eval_only:
        print(f"Wrote {len(eval_data)} eval examples to {args.eval_output} [{mode}] (eval-only; use your own prompt JSON for ICL)")
    else:
        print(
            f"Wrote {len(demos)} demos to {args.prompt_output} and "
            f"{len(eval_data)} eval examples to {args.eval_output} [{mode}]"
        )


if __name__ == "__main__":
    main()
