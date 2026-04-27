"""
Convert existing qasa_eval.json and qasa_prompt.json to closed-book versions.
Strips bracket citations from answers and removes docs, preserving example order/identity.

Usage:
    python tools/qasa_convert_to_closed_book.py \
        --eval_input  data/qasa_eval.json \
        --prompt_input data/qasa_prompt.json \
        --eval_output  data/qasa_closedbook_eval.json \
        --prompt_output data/qasa_closedbook_prompt.json
"""

import argparse
import json
import os
import re


CLOSEDBOOK_INSTRUCTION = (
    "Instruction: Write an accurate and concise answer for the given question."
)


def strip_citations(text: str) -> str:
    cleaned = re.sub(r"\s*\[\d+\]", "", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def convert_item(item: dict) -> dict:
    return {
        "question": item["question"],
        "answer": strip_citations(item["answer"]),
        "docs": [],
        "gold_ctxs": item.get("gold_ctxs", []),
    }


def convert_eval(eval_input: str, eval_output: str) -> None:
    with open(eval_input) as f:
        data = json.load(f)

    converted = [convert_item(item) for item in data]

    os.makedirs(os.path.dirname(eval_output) or ".", exist_ok=True)
    with open(eval_output, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"Wrote {len(converted)} eval examples to {eval_output}")


def convert_prompt(prompt_input: str, prompt_output: str) -> None:
    with open(prompt_input) as f:
        data = json.load(f)

    converted_demos = [convert_item(d) for d in data.get("demos", [])]

    out = {
        "instructions": [CLOSEDBOOK_INSTRUCTION],
        "instruction": CLOSEDBOOK_INSTRUCTION,
        "demo_sep": data.get("demo_sep", "\n\n\n"),
        "demo_prompt": "{INST}\n\nQuestion: {Q}\n\nAnswer: {A}",
        "doc_prompt": data.get("doc_prompt", "Document [{ID}](Title: {T}): {P}\n"),
        "demos": converted_demos,
    }

    os.makedirs(os.path.dirname(prompt_output) or ".", exist_ok=True)
    with open(prompt_output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(converted_demos)} demos to {prompt_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert qasa_eval.json / qasa_prompt.json to closed-book (no docs, no citations)."
    )
    parser.add_argument("--eval_input",    type=str, default="data/qasa_eval.json")
    parser.add_argument("--prompt_input",  type=str, default="data/qasa_prompt.json")
    parser.add_argument("--eval_output",   type=str, default="data/qasa_closedbook_eval.json")
    parser.add_argument("--prompt_output", type=str, default="data/qasa_closedbook_prompt.json")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only convert the eval file; skip prompt conversion.",
    )
    args = parser.parse_args()

    convert_eval(args.eval_input, args.eval_output)
    if not args.eval_only:
        convert_prompt(args.prompt_input, args.prompt_output)


if __name__ == "__main__":
    main()
