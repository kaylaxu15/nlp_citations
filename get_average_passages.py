from datasets import load_dataset

dataset = load_dataset("LinerAI/QASA")["test"]

ctx_lengths = [len(row.get("ctxs") or []) for row in dataset]

print(f"Questions:    {len(ctx_lengths)}")
print(f"Total ctxs:   {sum(ctx_lengths)}")
print(f"Average ctxs: {sum(ctx_lengths) / len(ctx_lengths):.3f}")