"""
Phase 6 — Fine-Tune Dataset Preparation
Converts processed Q&A pairs into the instruction-following format
needed by Unsloth / TRL for LoRA fine-tuning.
"""

import json
from pathlib import Path

from src.agents.prompts import FINETUNE_TEMPLATE

PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "finetune_dataset.json"


def prepare():
    qa_pairs = json.loads((PROCESSED_DIR / "qa_pairs_clean.json").read_text())
    articles = {a["id"]: a for a in json.loads((PROCESSED_DIR / "articles_clean.json").read_text())}

    dataset = []
    for qa in qa_pairs:
        # Build context from referenced articles
        context_parts = [
            articles[cid]["content"]
            for cid in qa.get("context_ids", [])
            if cid in articles
        ]
        context = "\n\n".join(context_parts) if context_parts else "No additional context."

        formatted = FINETUNE_TEMPLATE.format(
            instruction=qa["question"],
            context=context,
            response=qa["answer"],
        )
        dataset.append({"text": formatted, "source_id": qa["id"]})

    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2))
    print(f"✅ Fine-tune dataset prepared: {len(dataset)} examples → {OUTPUT_PATH}")
    return dataset


if __name__ == "__main__":
    prepare()
