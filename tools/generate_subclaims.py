#!/usr/bin/env python3
"""
Build subclaim-generation prompts from prompts/subclaims_generation.json and optional OpenRouter calls.

Examples:
  # Print prompts only (no API)
  .venv/bin/python tools/generate_subclaims.py \\
    --eval_file data/qasa_eval_top5_n10.json --sample 3 --seed 42 --dry_run

  # Fixed example indices (into the eval list)
  .venv/bin/python tools/generate_subclaims.py \\
    --eval_file data/qasa_eval_top5_n10.json --indices 0,2,7 --dry_run

  # Call OpenRouter (needs OPENROUTER_API_KEY)
  .venv/bin/python tools/generate_subclaims.py \\
    --eval_file data/qasa_eval_top5_n10.json --sample 2 --call_openrouter \\
    --model google/gemma-4-31b-it:free --out_json data/subclaims_generated.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_dotenv() -> None:
    path = _ROOT / ".env"
    if not path.is_file():
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                    val = val[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass


def passage_from_item(item: Dict[str, Any], mode: str) -> str:
    docs = item.get("docs") or []
    if not docs:
        return ""
    if mode == "first":
        return str(docs[0].get("text", "")).strip()
    parts = []
    for d in docs:
        t = str(d.get("text", "")).strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def build_user_prompt(prompt_data: Dict[str, Any], question: str, passage: str, continuation: bool = True) -> str:
    intro = prompt_data["task_description"].strip()
    fmt = prompt_data.get("response_format", "").strip()
    if fmt:
        intro = intro + "\n\n" + fmt

    blocks: List[str] = []
    for ex in prompt_data["few_shot_examples"]:
        blocks.append(
            f"Original question: {ex['original_question']}\n\n"
            f"Passage: {ex['passage']}\n\n"
            f"Claim 1: {ex['claim_1']}\n"
            f"Claim 2: {ex['claim_2']}\n"
            f"Claim 3: {ex['claim_3']}"
        )
    few_shot = "\n\n".join(blocks)

    tail = (
        f"Original question: {question}\n\nPassage: {passage}\n\n"
        + ("Claim 1:" if continuation else "")
    )
    return f"{intro}\n\n{few_shot}\n\n---\n\n{tail}"


def parse_claims_from_completion(text: str) -> List[str]:
    """Extract three claims from model output (handles leading 'Claim 1:' or raw continuation)."""
    if not text:
        return []
    t = text.strip()
    # If model continued after we ended with 'Claim 1:', prepend label for regex
    if not re.search(r"(?i)claim\s*1\s*:", t[:200]):
        t = "Claim 1: " + t
    out: List[str] = []
    for i in range(1, 4):
        m = re.search(
            rf"(?is)Claim\s*{i}\s*:\s*(.*?)(?=Claim\s*{i + 1}\s*:|$)" if i < 3 else rf"(?is)Claim\s*{i}\s*:\s*(.*)",
            t,
        )
        if m:
            out.append(re.sub(r"\s+", " ", m.group(1).strip()))
    return out[:3]


def openrouter_complete(
    prompt: str,
    *,
    model: str,
    api_key: str,
    temperature: float = 0.3,
    max_tokens: int = 512,
    url: str = "https://openrouter.ai/api/v1/chat/completions",
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
        "X-Title": os.environ.get("OPENROUTER_APP_NAME", "nlp_citations-subclaims"),
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content", "")
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content).strip()


def main() -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(description="Generate 3 subclaims per eval row (prompt + optional OpenRouter).")
    p.add_argument("--prompt_file", type=str, default="prompts/subclaims_generation.json")
    p.add_argument("--eval_file", type=str, required=True, help="JSON array of items with question + docs")
    p.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated 0-based indices into eval list (e.g. 0,3,9). Overrides --sample.",
    )
    p.add_argument("--sample", type=int, default=None, help="Randomly choose this many examples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--passage_mode", choices=("concat", "first"), default="concat")
    p.add_argument("--dry_run", action="store_true", help="Only print prompts / skip API")
    p.add_argument("--call_openrouter", action="store_true", help="POST to OpenRouter")
    p.add_argument("--model", type=str, default=os.environ.get("OPENROUTER_MODEL", "google/gemma-4-31b-it:free"))
    p.add_argument("--out_json", type=str, default=None, help="Write results list to this path")
    args = p.parse_args()

    prompt_path = _ROOT / args.prompt_file
    eval_path = _ROOT / args.eval_file
    with open(prompt_path, encoding="utf-8") as f:
        prompt_data = json.load(f)
    with open(eval_path, encoding="utf-8") as f:
        eval_data = json.load(f)
    if not isinstance(eval_data, list):
        raise ValueError("--eval_file must be a JSON array")

    n = len(eval_data)
    if args.indices is not None:
        idxs = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
        for i in idxs:
            if i < 0 or i >= n:
                raise ValueError(f"Index {i} out of range [0,{n})")
    elif args.sample is not None:
        rng = random.Random(args.seed)
        k = min(args.sample, n)
        idxs = sorted(rng.sample(range(n), k=k))
    else:
        k = min(3, n)
        print(f"No --indices or --sample: defaulting to first {k} examples (0..{k-1}).", file=sys.stderr)
        idxs = list(range(k))

    results: List[Dict[str, Any]] = []
    api_key = os.environ.get("OPENROUTER_API_KEY") if args.call_openrouter else None
    if args.call_openrouter and not api_key:
        raise ValueError("OPENROUTER_API_KEY required when --call_openrouter")

    for idx in idxs:
        item = eval_data[idx]
        question = str(item.get("question", "")).strip()
        passage = passage_from_item(item, args.passage_mode)
        user_prompt = build_user_prompt(prompt_data, question, passage, continuation=True)

        row: Dict[str, Any] = {
            "eval_index": idx,
            "question": question,
            "passage_excerpt_chars": len(passage),
            "prompt_char_len": len(user_prompt),
        }

        print(f"\n{'=' * 60}\n# eval_index={idx}\n{'=' * 60}")
        if args.dry_run or not args.call_openrouter:
            print(user_prompt[:12000])
            if len(user_prompt) > 12000:
                print(f"\n... [{len(user_prompt)} chars total]\n")

        raw = ""
        if args.call_openrouter and not args.dry_run:
            raw = openrouter_complete(user_prompt, model=args.model, api_key=api_key)
            claims = parse_claims_from_completion(raw)
            row["claims"] = claims
            row["raw_completion"] = raw
            print(raw[:4000])
        elif args.dry_run:
            row["claims"] = []
            row["raw_completion"] = ""

        results.append(row)

    if args.out_json:
        out_path = _ROOT / args.out_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "prompt_file": args.prompt_file,
            "eval_file": args.eval_file,
            "indices": idxs,
            "passage_mode": args.passage_mode,
            "model": args.model if args.call_openrouter else None,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
