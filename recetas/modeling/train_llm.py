import argparse
import json
import gc
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

from recetas.config import DATA, MODELS, BASE_LLM_NAME

PARQUET     = DATA.raw_parquet
LORA_OUT    = MODELS.lora_dir
MERGED_OUT  = MODELS.merged_dir
BASE_MODEL  = BASE_LLM_NAME
MAX_SEQ_LEN = 512


def build_prompt(row) -> str:
    title       = row.get("recipe_title", "")
    ingredients = row.get("ingredient_text", "")
    directions  = row.get("directions_text", "") or row.get("directions", "")
    dietary     = row.get("dietary_profile", "")
    cuisine     = row.get("cuisine_list", "")

    ctx_parts = []
    if dietary:
        ctx_parts.append(f"Dieta: {dietary}")
    if cuisine:
        try:
            cuisines = json.loads(cuisine) if isinstance(cuisine, str) else cuisine
            if cuisines:
                ctx_parts.append(f"Cocina: {', '.join(cuisines[:2])}")
        except Exception:
            pass
    ctx = "  |  ".join(ctx_parts)

    system = "Eres un chef experto. Crea instrucciones claras y apetitosas para preparar la receta."
    user   = f"Receta: {title}\nIngredientes: {ingredients}"
    if ctx:
        user += f"\n{ctx}"

    if isinstance(directions, str):
        try:
            steps = json.loads(directions)
            directions = " ".join(steps) if isinstance(steps, list) else directions
        except Exception:
            pass
    if not directions or len(str(directions).strip()) < 20:
        return None

    directions = str(directions).strip()[:800]

    return (
        f"<|system|>\n{system}</s>\n"
        f"<|user|>\n{user}</s>\n"
        f"<|assistant|>\n{directions}</s>"
    )


def load_dataset(max_samples: int | None) -> Dataset:
    print(f"Loading {PARQUET.name}…")
    df = pd.read_parquet(PARQUET)
    if max_samples:
        df = df.sample(min(max_samples, len(df)), random_state=42)

    print(f"Building prompts ({len(df):,} recipes)…")
    prompts = []
    for _, row in df.iterrows():
        p = build_prompt(row)
        if p:
            prompts.append({"text": p})

    print(f"Valid prompts: {len(prompts):,}")
    print(f"\nSample:\n{prompts[0]['text'][:400]}\n{'─'*40}\n")
    return Dataset.from_list(prompts)


def load_model_and_tokenizer(device_map="auto"):
    print(f"Loading {BASE_MODEL} in 4-bit…")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_cfg,
        device_map=device_map,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--grad-accum",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--max-samples", type=int,   default=None,
                        help="Limit number of recipes (None = all)")
    parser.add_argument("--skip-merge",  action="store_true",
                        help="Save LoRA adapters only, skip merge step")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Warning: no GPU detected. Training will be slow.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")

    dataset = load_dataset(args.max_samples)
    model, tokenizer = load_model_and_tokenizer()

    training_args = SFTConfig(
        output_dir=str(LORA_OUT),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=0,
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("\nStarting QLoRA training…")
    trainer.train()

    print(f"\nSaving LoRA adapters → {LORA_OUT}")
    trainer.save_model(str(LORA_OUT))
    tokenizer.save_pretrained(str(LORA_OUT))

    if not args.skip_merge:
        print("\nMerging LoRA with base model…")
        del model
        gc.collect()
        torch.cuda.empty_cache()

        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.bfloat16,
            device_map="cpu",
        )
        merged = PeftModel.from_pretrained(base, str(LORA_OUT))
        merged = merged.merge_and_unload()
        merged.save_pretrained(str(MERGED_OUT), safe_serialization=True)
        tokenizer.save_pretrained(str(MERGED_OUT))
        print(f"Merged model → {MERGED_OUT}")

    print("\nDone.")
    print(f"  LoRA adapters: {LORA_OUT}")
    if not args.skip_merge:
        print(f"  Merged model:  {MERGED_OUT}")
        print("\nNext — convert to GGUF:")
        print(f"  python llama.cpp/convert_hf_to_gguf.py {MERGED_OUT} --outtype q4_k_m")


if __name__ == "__main__":
    main()
