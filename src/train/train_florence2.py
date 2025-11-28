# ================================
# Florence-2 Fine-Tuning (Windows Safe)
# ================================
import os
import math
import argparse
from dataclasses import dataclass
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

from transformers.dynamic_module_utils import get_imports as hf_get_imports
from unittest.mock import patch

from peft import LoraConfig, get_peft_model
from src.data.dataset_vllm import VLLMDataset


# ---------------------------
# Patch flash-attn requirement
# ---------------------------
def patched_get_imports(filename):
    items = hf_get_imports(filename)
    if "flash_attn" in items:
        items.remove("flash_attn")   # Windows safe
    return items


# ---------------------------
# Config
# ---------------------------
@dataclass
class TrainConfig:
    model_name: str = "microsoft/Florence-2-base-ft"

    train_parquet: str = "metadata/train.parquet"
    val_parquet: str = "metadata/val.parquet"
    images_dir: str = "images"
    output_dir: str = "florence2_finetune"

    batch_size: int = 1
    num_epochs: int = 1
    num_workers: int = 0            # <--- MUST be 0 on Windows

    lr: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    grad_accum_steps: int = 4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------
# Collate Function (Correct)
# ---------------------------
def collate_fn(batch, processor, cfg):

    images = [x["image"] for x in batch]
    prompts = [x["metadata_text"] for x in batch]
    captions = [x["caption"] for x in batch]

    # Florence expects text + image together
    enc = processor(
        images=images,
        text=prompts,
        padding=True,
        return_tensors="pt"
    )

    # Florence uses SAME tokenizer for targets
    tgt = processor.tokenizer(
        captions,
        padding=True,
        return_tensors="pt"
    )

    # Shift labels convention
    labels = tgt["input_ids"]
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "pixel_values": enc["pixel_values"],
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


# ---------------------------
# Training Loop
# ---------------------------
def main():
    cfg = TrainConfig()

    # Windows fix
    os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"

    print("\nLoading Florence2 model:", cfg.model_name)

    processor = AutoProcessor.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
    )

    # Load model with patched import check
    with patch("transformers.dynamic_module_utils.get_imports", patched_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation="sdpa"
        ).to(cfg.device)

    # ----------------------
    # LoRA
    # ----------------------
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ----------------------
    # Dataset
    # ----------------------
    train_ds = VLLMDataset(cfg.train_parquet, cfg.images_dir)
    val_ds = VLLMDataset(cfg.val_parquet, cfg.images_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(collate_fn, processor=processor, cfg=cfg),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_fn, processor=processor, cfg=cfg),
    )

    # ----------------------
    # Optimizer + Scheduler
    # ----------------------
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = math.ceil(len(train_loader) / cfg.grad_accum_steps) * cfg.num_epochs
    warmup = int(total_steps * cfg.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=total_steps,
    )

    # ----------------------
    # Train
    # ----------------------
    best_val = float("inf")

    for epoch in range(cfg.num_epochs):
        print(f"\nEpoch {epoch+1}/{cfg.num_epochs}\n")
        model.train()
        running_loss = 0

        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader)):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}

            loss = model(**batch).loss
            (loss / cfg.grad_accum_steps).backward()

            running_loss += loss.item()

            if (step + 1) % cfg.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # -------------------
        # Validation
        # -------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(cfg.device) for k, v in batch.items()}
                val_loss += model(**batch).loss.item()

        val_loss /= len(val_loader)

        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            print("Saving best checkpoint...")
            model.save_pretrained(cfg.output_dir)
            processor.save_pretrained(cfg.output_dir)

    print("\nTraining done. Best val loss:", best_val)


if __name__ == "__main__":
    main()
