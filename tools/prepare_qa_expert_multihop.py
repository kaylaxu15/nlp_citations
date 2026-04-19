import argparse
import json
import os

from datasets import load_dataset


DEFAULT_INSTRUCTION = (
    "Instruction: Write an accurate, engaging, and concise answer for the given question "
    "using only the provided contexts and cite them properly. Use an unbiased and journalistic "
    "tone. Always cite for any factual claim. When citing several contexts, use [1][2][3]. "
    "Cite at least one document and at most three documents in each sentence."
)


def extract_final_answer(row):
    answer_text = (row.get("final_answer") or row.get("answer") or "").strip()
    if not answer_text:
        return ""
    if "Answer:" in answer_text:
        # Keep only the explicit final answer span.
        answer_text = answer_text.split("Answer:")[-1].strip()
    return answer_text


def build_docs(sub_questions, max_docs):
    docs = []
    seen_paragraphs = set()

    for sub_question in sub_questions:
        paragraph = (sub_question.get("paragraph") or "").strip()
        if not paragraph:
            continue
        if paragraph in seen_paragraphs:
            continue

        seen_paragraphs.add(paragraph)
        title = (sub_question.get("question") or "").strip() or f"Sub-question {len(docs) + 1}"
        docs.append({"title": title, "text": paragraph})
        if len(docs) >= max_docs:
            break

    return docs


def build_item(row, max_docs):
    question = (row.get("question") or "").strip()
    answer = extract_final_answer(row)
    sub_questions = row.get("sub_questions") or []
    docs = build_docs(sub_questions, max_docs=max_docs)
    return {
        "question": question,
        "answer": answer,
        "docs": docs,
        "multihop": bool(row.get("multihop", False)),
    }


def build_prompt_data(demos):
    return {
        "instruction": DEFAULT_INSTRUCTION,
        "demo_sep": "\n\n\n",
        "demo_prompt": "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}",
        "doc_prompt": "Document [{ID}](Title: {T}): {P}\n",
        "demos": demos,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare ALCE-formatted files from QA-Expert multi-hop dataset")
    parser.add_argument("--dataset_name", type=str, default="khaimaitien/qa-expert-multi-hop-qa-V1.0")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--num_demos", type=int, default=8, help="Number of in-context demos in prompt file")
    parser.add_argument("--max_docs", type=int, default=5, help="Maximum contexts per example")
    parser.add_argument("--multihop_only", action="store_true", help="Keep only multihop examples")
    parser.add_argument("--prompt_output", type=str, default="data/qa_expert_multihop_prompt.json")
    parser.add_argument("--eval_output", type=str, default="data/qa_expert_multihop_eval.json")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name)
    train_rows = dataset[args.train_split]
    eval_rows = dataset[args.eval_split]

    demos = []
    for row in train_rows:
        if args.multihop_only and not row.get("multihop", False):
            continue
        item = build_item(row, max_docs=args.max_docs)
        if not item["question"] or not item["answer"] or len(item["docs"]) == 0:
            continue
        demos.append(item)
        if len(demos) >= args.num_demos:
            break

    if len(demos) < args.num_demos:
        raise ValueError(
            f"Could only build {len(demos)} demos; requested {args.num_demos}. "
            "Try disabling --multihop_only or reducing --num_demos."
        )

    eval_data = []
    for row in eval_rows:
        if args.multihop_only and not row.get("multihop", False):
            continue
        item = build_item(row, max_docs=args.max_docs)
        if not item["question"] or not item["answer"] or len(item["docs"]) == 0:
            continue
        eval_data.append(item)

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
