"""LoRA fine-tuning script for BLIP-2 on the fashion dataset.

This script adapts the fp16 BLIP-2 OPT-2.7B checkpoint from Hugging Face using
Low-Rank Adaptation (LoRA). It is intentionally lightweight so you can run it on
single-GPU machines (>=16 GB) by adjusting batch size / gradient accumulation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from peft import LoraConfig, get_peft_model
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    Trainer,
    TrainingArguments,
)


@dataclass
class TrainConfig:
    model_name: str
    train_parquet: Path
    images_dir: Path
    output_dir: Path
    num_epochs: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    max_samples: int | None


class FashionBLIP2Dataset(Dataset):
    """Simple dataset that pairs product images with captions."""

    def __init__(self, parquet_path: Path, images_dir: Path, processor: Blip2Processor, max_samples: int | None = None) -> None:
        self.df = pd.read_parquet(parquet_path)
        if max_samples is not None:
            self.df = self.df.head(max_samples)
        self.images_dir = images_dir
        self.processor = processor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = self.images_dir / f"{row['id']}.jpg"
        image = Image.open(image_path).convert("RGB")
        caption = str(row["caption"])

        inputs = self.processor(images=image, text=caption, return_tensors="pt")
        sample = {k: v.squeeze(0) for k, v in inputs.items()}
        sample["labels"] = sample["input_ids"].clone()
        return sample


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    collated: Dict[str, torch.Tensor] = {}
    for key in batch[0].keys():
        collated[key] = torch.stack([example[key] for example in batch])
    return collated


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Fine-tune BLIP-2 with LoRA on the fashion dataset")
    parser.add_argument("--model_name", default="ybelkada/blip2-opt-2.7b-fp16-sharded")
    parser.add_argument("--train_parquet", default="metadata/train.parquet")
    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--output_dir", default="experiments/blip2_lora")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    return TrainConfig(
        model_name=args.model_name,
        train_parquet=Path(args.train_parquet),
        images_dir=Path(args.images_dir),
        output_dir=Path(args.output_dir),
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_samples=args.max_samples,
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    processor = Blip2Processor.from_pretrained(cfg.model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    train_dataset = FashionBLIP2Dataset(
        parquet_path=cfg.train_parquet,
        images_dir=cfg.images_dir,
        processor=processor,
        max_samples=cfg.max_samples,
    )

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    processor.save_pretrained(str(cfg.output_dir))


if __name__ == "__main__":
    main()
