import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
import json
import re
import os
import string
import time

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def strip_cot_round1_echoed_final_answer(s: str) -> str:
    """Remove text after a second blank-line 'Answer:' block in round-1 CoT output.

    The round-1 template ends with 'Answer:'; the model should write only passage
    relevance lines and the 'Most critical documents' line, but often repeats
    '\\n\\nAnswer:\\n' and emits round-2-style prose. That should not appear in
    cot_output or in {COT} for round 2.
    """
    if not s:
        return s
    s = s.rstrip()
    # Match a new paragraph starting with "Answer:" (not the line the model continues right after the prompt).
    # We only strip when that header is set off by a blank line so we do not cut mid-sentence on rare "Answer:" in quotes.
    m = re.search(r"(?i)\n\s*\n\s*Answer:\s*\n", s)
    if m is not None:
        return s[: m.start()].rstrip()
    return s


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


def _doc_prompt_display_id(doc, doc_id):
    """Prefer QASA passage index (id_suffix / id after '_') so CoT matches demo style 'Passage [23]'; else 1-based rank."""
    suf = doc.get("id_suffix")
    if suf is not None and str(suf).strip():
        return str(suf).strip()
    cid = doc.get("id")
    if cid:
        s = str(cid).strip()
        if "_" in s:
            return s.rsplit("_", 1)[-1]
        return s
    return str(doc_id + 1)


def make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=None):
    # For doc prompt:
    # - {ID}: display rank for citations (QASA id_suffix when set), else 1-based position in the shown list
    # - {RAW}: stable passage key from eval JSON (e.g. 1512.02325_all_36); empty if missing
    # - {T}: title
    # - {P}: text
    # use_shorter: None, "summary", or "extraction"

    text = doc['text']
    if use_shorter is not None:
        text = doc[use_shorter]
    disp = _doc_prompt_display_id(doc, doc_id)
    raw = str(doc.get("id", "")).strip()
    return (
        doc_prompt.replace("{T}", doc["title"])
        .replace("{P}", text)
        .replace("{ID}", disp)
        .replace("{RAW}", raw)
    )


def get_shorter_text(item, docs, ndoc, key):
    doc_list = []
    for item_id, item in enumerate(docs):
        if key not in item:
            if len(doc_list) == 0:
                # If there aren't any document, at least provide one (using full text)
                item[key] = item['text']
                doc_list.append(item)
            logger.warn(f"No {key} found in document. It could be this data do not contain {key} or previous documents are not relevant. This is document {item_id}. This question will only have {len(doc_list)} documents.")
            break
        if "irrelevant" in item[key] or "Irrelevant" in item[key]:
            continue
        doc_list.append(item)
        if len(doc_list) >= ndoc:
            break
    return doc_list


DEFAULT_DEMO_PROMPT_ROUND2 = (
    "{INST}\n\nQuestion: {Q}\n\n{D}\n\nPrior analysis:\n{COT}\n\nAnswer: {A}"
)


def normalize_instructions(prompt_data):
    """Return a list of instruction strings. Supports `instructions` (str or list) or legacy `instruction`."""
    if "instructions" in prompt_data:
        inst = prompt_data["instructions"]
        if isinstance(inst, str):
            return [inst]
        if isinstance(inst, list):
            return inst
        raise TypeError("'instructions' must be a string or a list of strings")
    if "instruction" in prompt_data:
        return [prompt_data["instruction"]]
    raise KeyError("Prompt file must contain 'instructions' (or legacy 'instruction')")


def _coerce_instruction_list(instruction):
    if instruction is None:
        raise ValueError("instruction / instructions is required")
    if isinstance(instruction, str):
        return [instruction]
    return list(instruction)


def _make_demo_single(item, prompt, ndoc, doc_prompt, inst_text, use_shorter, test):
    prompt = prompt.replace("{INST}", inst_text).replace("{Q}", item["question"])
    if "{D}" in prompt:
        if ndoc == 0:
            prompt = prompt.replace("{D}\n", "")
        else:
            doc_list = get_shorter_text(item, item["docs"], ndoc, use_shorter) if use_shorter is not None else item["docs"][:ndoc]
            text = "".join([make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=use_shorter) for doc_id, doc in enumerate(doc_list)])
            prompt = prompt.replace("{D}", text)

    if not test:
        answer = "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
        prompt = prompt.replace("{A}", "").rstrip() + answer
    else:
        prompt = prompt.replace("{A}", "").rstrip()

    return prompt


def _apply_demo_template(
    item, template, ndoc, doc_prompt, inst_text, use_shorter, test, answer_suffix=None, cot_text=None
):
    """Fill {INST},{Q},{D},{A}; optionally {COT} when cot_text is set."""
    prompt = template.replace("{INST}", inst_text).replace("{Q}", item["question"])
    if "{D}" in prompt:
        if ndoc == 0:
            prompt = prompt.replace("{D}\n", "")
        else:
            doc_list = get_shorter_text(item, item["docs"], ndoc, use_shorter) if use_shorter is not None else item["docs"][:ndoc]
            text = "".join([make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=use_shorter) for doc_id, doc in enumerate(doc_list)])
            prompt = prompt.replace("{D}", text)
    if cot_text is not None and "{COT}" in prompt:
        prompt = prompt.replace("{COT}", cot_text)

    if not test:
        assert answer_suffix is not None
        prompt = prompt.replace("{A}", "").rstrip() + answer_suffix
    else:
        prompt = prompt.replace("{A}", "").rstrip()

    return prompt


def _make_demo_cot(
    item,
    prompt,
    ndoc,
    doc_prompt,
    instructions,
    use_shorter,
    test,
    demo_prompt_round2,
    cot_demo_inner_sep,
    cot_round,
    prior_cot,
    cot_include_demo_round2=True,
):
    inst0, inst1 = instructions[0], instructions[1]
    round2_template = demo_prompt_round2 or DEFAULT_DEMO_PROMPT_ROUND2

    if not test:
        ans = item["answer"]
        if not (isinstance(ans, list) and len(ans) == 2 and all(isinstance(x, str) for x in ans)):
            raise ValueError(
                "Chain-of-thought mode (multiple instructions) expects each demo's 'answer' to be "
                "a list of two strings: [prior_analysis, final_answer]."
            )
        cot_s, final_s = ans[0], ans[1]
        r1 = _apply_demo_template(
            item,
            prompt,
            ndoc,
            doc_prompt,
            inst0,
            use_shorter,
            test=False,
            answer_suffix="\n" + cot_s,
        )
        r2 = _apply_demo_template(
            item,
            round2_template,
            ndoc,
            doc_prompt,
            inst1,
            use_shorter,
            test=False,
            answer_suffix="\n" + final_s,
            cot_text=cot_s,
        )
        if not cot_include_demo_round2:
            return r1
        return r1 + cot_demo_inner_sep + r2

    if cot_round == 1:
        return _apply_demo_template(
            item, prompt, ndoc, doc_prompt, inst0, use_shorter, test=True
        )
    if cot_round == 2:
        if prior_cot is None:
            raise ValueError("cot_round=2 requires prior_cot (model output from round 1)")
        return _apply_demo_template(
            item,
            round2_template,
            ndoc,
            doc_prompt,
            inst1,
            use_shorter,
            test=True,
            cot_text=prior_cot,
        )
    raise ValueError("cot_round must be 1 or 2 when using chain-of-thought prompts")


def make_demo(
    item,
    prompt,
    ndoc=None,
    doc_prompt=None,
    instruction=None,
    use_shorter=None,
    test=False,
    demo_prompt_round2=None,
    cot_demo_inner_sep="\n\n\n",
    cot_round=1,
    prior_cot=None,
    cot_include_demo_round2=True,
):
    # For demo prompt
    # - {INST}: the instruction
    # - {D}: the documents
    # - {Q}: the question
    # - {A}: the answers
    # ndoc: number of documents to put in context
    # use_shorter: None, "summary", or "extraction"
    # Multiple instructions: two-round chain-of-thought (pass instruction as list of 2+ strings; first two are used).

    instructions = _coerce_instruction_list(instruction)
    if len(instructions) == 1:
        return _make_demo_single(
            item, prompt, ndoc, doc_prompt, instructions[0], use_shorter, test
        )

    return _make_demo_cot(
        item,
        prompt,
        ndoc,
        doc_prompt,
        instructions,
        use_shorter,
        test,
        demo_prompt_round2,
        cot_demo_inner_sep,
        cot_round,
        prior_cot,
        cot_include_demo_round2=cot_include_demo_round2,
    )


def load_model(model_name_or_path, dtype=torch.float16, int8=False, reserve_memory=10):
    # Load a huggingface model and tokenizer
    # dtype: torch.float16 or torch.bfloat16
    # int8: whether to use int8 quantization
    # reserve_memory: how much memory to reserve for the model on each gpu (in GB)

    # Load the FP16 model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading {model_name_or_path} in {dtype}...")
    if int8:
        logger.warn("Use LLM.int8")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        torch_dtype=dtype,
        max_memory=get_max_memory(),
        load_in_8bit=int8,
    )
    logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    # Fix OPT bos token problem in HF
    if "opt" in model_name_or_path:
        tokenizer.bos_token = "<s>"
    tokenizer.padding_side = "left"

    return model, tokenizer
