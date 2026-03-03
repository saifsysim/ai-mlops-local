"""
Phase 6 — Local Fine-Tuning with Unsloth + LoRA
Adapts LLaMA 3 (8B, 4-bit) to your domain data using parameter-efficient LoRA.

Prerequisites (install separately before running):
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install torch torchvision torchaudio

Hardware: Requires 16 GB+ RAM on Mac (M1/M2/M3 with MPS backend).
Time: ~10-20 minutes for 3 Q&A examples (increase dataset for real training).

After training, import to Ollama:
    ollama create ai-mlops-model -f src/training/Modelfile
"""

import json
from pathlib import Path

DATASET_PATH = Path("data/processed/finetune_dataset.json")
OUTPUT_DIR = Path("src/training/checkpoints")
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"   # 4-bit: fits in 8-10 GB RAM


def finetune():
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
    except ImportError:
        print("❌ Unsloth not installed. See README for install instructions.")
        return

    # ── Load model with 4-bit quantisation ────────────────────────────────────
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # ── Add LoRA adapters (train only ~1% of params) ────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
    )

    # ── Load dataset ───────────────────────────────────────────────────────────
    raw = json.loads(DATASET_PATH.read_text())
    dataset = Dataset.from_list(raw)
    print(f"Dataset: {len(dataset)} training examples")

    # ── Train ──────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=1,
            save_strategy="epoch",
            fp16=False,
            bf16=False,
        ),
    )
    trainer.train()

    # ── Save as GGUF for Ollama ─────────────────────────────────────────────────
    gguf_path = OUTPUT_DIR / "model.gguf"
    model.save_pretrained_gguf(str(gguf_path), tokenizer)
    print(f"✅ Model saved → {gguf_path}")
    print("   To use in Ollama: ollama create ai-mlops-model -f src/training/Modelfile")


if __name__ == "__main__":
    finetune()
