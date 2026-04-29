#!/usr/bin/env bash
# Post-hoc GTR citation for QASA closed-book runs (ndoc=0), using passages from aligned vanilla (ndoc=3) runs.
# Same quick_test + indices as closed-book → row order matches.
#
# Usage from repo root:
#   bash scripts/post_hoc_closedbook_qasa.sh
# GPU: DEVICE=cuda bash scripts/post_hoc_closedbook_qasa.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"

"$PYTHON" post_hoc_cite.py \
  --f result/qasa-gemma-4-26b-a4b-it-openrouter_closedbook-shot2-ndoc0-42-quick_test200.json \
  --docs_from result/vanilla_qasa-gemma-4-26b-a4b-it-openrouter-shot1-ndoc3-42-quick_test200.json \
  --retriever gtr-t5-large \
  --retriever_device "$DEVICE"

"$PYTHON" post_hoc_cite.py \
  --f result/qasa-llama-3.3-70b-instruct-openrouter_closedbook-shot2-ndoc0-42-quick_test200.json \
  --docs_from result/vanilla_qasa-llama-3.3-70b-instruct-openrouter-shot1-ndoc3-42-quick_test200.json \
  --retriever gtr-t5-large \
  --retriever_device "$DEVICE"

echo "Done. Outputs end with .post_hoc_cite.gtr-t5-large-external"
