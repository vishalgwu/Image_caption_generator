# src/eval/eval_vllm.py

import os
import argparse
from typing import List

import pandas as pd
from tqdm.auto import tqdm

import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from transformers import Blip2ForConditionalGeneration, Blip2Processor
from PIL import Image


def load_model(model_dir: str, device: str):
    processor = Blip2Processor.from_pretrained(model_dir)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = model.to(device)
    model.eval()
    return processor, model


def generate_caption(
    image: Image.Image,
    metadata_text: str,
    processor: Blip2Processor,
    model: Blip2ForConditionalGeneration,
    device: str,
    max_new_tokens: int = 32,
) -> str:
    inputs = processor(
        images=image,
        text=metadata_text,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
        )

    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    return generated_text.strip()


def build_metadata_text(row: pd.Series) -> str:
    parts = []

    def add(label: str, key: str):
        if key in row and pd.notna(row[key]):
            parts.append(f"{label}: {row[key]}")

    add("Gender", "gender")
    add("Category", "masterCategory")
    add("Subcategory", "subCategory")
    add("Article type", "articleType")
    add("Base colour", "baseColour")
    add("Season", "season")
    add("Usage", "usage")

    meta_text = ". ".join(parts)
    if meta_text:
        meta_text += ". "
    meta_text += "Generate a concise, descriptive caption for this fashion product."
    return meta_text


def compute_bleu_and_rouge(refs: List[str], hyps: List[str]):
    # BLEU
    smooth = SmoothingFunction().method4
    refs_tokenized = [[r.split()] for r in refs]
    hyps_tokenized = [h.split() for h in hyps]
    bleu1 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]
    rougeL = sum(rouge_scores) / max(len(rouge_scores), 1)

    return bleu1, bleu4, rougeL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="blip2_finetune")
    parser.add_argument("--test_parquet", default="metadata/test.parquet")
    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--output_csv", default="metadata/vllm_finetune_eval.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model from", args.model_dir)
    processor, model = load_model(args.model_dir, device)

    df = pd.read_parquet(args.test_parquet)

    pred_caps = []
    gt_caps = []
    image_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating VLLM"):
        image_id = str(row["id"])
        img_path = os.path.join(args.images_dir, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        metadata_text = build_metadata_text(row)
        gt_caption = str(row["caption"])

        pred_caption = generate_caption(
            image, metadata_text, processor, model, device
        )

        image_ids.append(image_id)
        pred_caps.append(pred_caption)
        gt_caps.append(gt_caption)

    # metrics
    bleu1, bleu4, rougeL = compute_bleu_and_rouge(gt_caps, pred_caps)
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    print(f"ROUGE-L: {rougeL:.4f}")

    # save CSV
    out_df = pd.DataFrame(
        {
            "id": image_ids,
            "gt_caption": gt_caps,
            "pred_caption": pred_caps,
        }
    )
    out_df.to_csv(args.output_csv, index=False)
    print("Saved per-sample predictions to:", args.output_csv)


if __name__ == "__main__":
    main()
