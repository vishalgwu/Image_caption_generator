# src/train/train_vllm_blip2.py
#
# Fine-tune BLIP-2 (OPT 2.7B) in FP32 on images + metadata + captions.
# Clean + Windows compatible, no bitsandbytes.

import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    get_cosine_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model

from src.data.dataset_vllm import VLLMDataset


@dataclass
class TrainConfig:
    model_name: str = "Salesforce/blip2-opt-2.7b"
    train_parquet: str = "metadata/train.parquet"
    val_parquet: str = "metadata/val.parquet"
    images_dir: str = "images"
    output_dir: str = "blip2_finetune"

    batch_size: int = 1       # OPT 2.7B + FP32 = large model → keep small
    num_workers: int = 2
    num_epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_input_length: int = 128
    max_target_length: int = 32
    grad_accum_steps: int = 4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Fine-tune BLIP-2 (OPT) on fashion captions")
    parser.add_argument("--train_parquet", default="metadata/train.parquet")
    parser.add_argument("--val_parquet", default="metadata/val.parquet")
    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--output_dir", default="blip2_finetune")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=32)
    args = parser.parse_args()

    cfg = TrainConfig(
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        lr=args.lr,
        grad_accum_steps=args.grad_accum_steps,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )
    return cfg


def collate_fn(batch: List[Dict], processor: Blip2Processor, cfg: TrainConfig):
    images = [b["image"] for b in batch]
    prompts = [b["metadata_text"] for b in batch]
    captions = [b["caption"] for b in batch]

    enc_inputs = processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_input_length,
    )

    with processor.as_target_processor():
        tgt = processor(
            text=captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_target_length,
        )

    labels = tgt["input_ids"]
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "pixel_values": enc_inputs["pixel_values"],
        "input_ids": enc_inputs["input_ids"],
        "attention_mask": enc_inputs["attention_mask"],
        "labels": labels,
    }


def main():
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Device: {cfg.device}")
    print("Loading processor and model:", cfg.model_name)
    os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    processor = Blip2Processor.from_pretrained(cfg.model_name)

    model = Blip2ForConditionalGeneration.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float32,
    )

    if torch.cuda.is_available():
        model = model.to(cfg.device)

    # LoRA for OPT-based transformer
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Datasets ----
    train_ds = VLLMDataset(cfg.train_parquet, cfg.images_dir)
    val_ds = VLLMDataset(cfg.val_parquet, cfg.images_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,      # <-- FIXED
        collate_fn=lambda b: collate_fn(b, processor, cfg),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda b: collate_fn(b, processor, cfg),
    )

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = math.ceil(len(train_loader) / cfg.grad_accum_steps) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")

    for epoch in range(cfg.num_epochs):
        # === Train ===
        model.train()
        train_loss_sum = 0.0
        step_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} - train")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}

            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss / cfg.grad_accum_steps
            loss.backward()

            train_loss_sum += loss.item()
            step_count += 1

            if (step + 1) % cfg.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": train_loss_sum / max(step_count, 1)})

        # === Val ===
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} - val"):
                batch = {k: v.to(cfg.device) for k, v in batch.items()}
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                val_loss_sum += outputs.loss.item()
                val_steps += 1

        avg_train_loss = train_loss_sum / max(step_count, 1)
        avg_val_loss = val_loss_sum / max(val_steps, 1)

        print(
            f"Epoch {epoch+1} finished. Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("Saving best model...")
            model.save_pretrained(cfg.output_dir)
            processor.save_pretrained(cfg.output_dir)

    print("Training complete. Best val loss =", best_val_loss)


if __name__ == "__main__":
    main()
