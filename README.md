# Image_caption_generator
Generates on-brand fashion captions using images and product data. 

#  Image Caption Generator (Baseline   And Qwen2-VL)

Fashion image captioning system with:
- **Baseline CNN + Transformer** caption model (trained from scratch on Myntra dataset).
- **Qwen2-VL** vision-language model for **semantic explainability** and caption comparison.
- **Streamlit dashboard** for image upload, caption generation, token-level importance and model comparison. :contentReference[oaicite:0]{index=0}
![](https://github.com/vishalgwu/Image_caption_generator/blob/main/Project_images/Baseline-model-word-imp.png)

---
## Overview

This repository builds an image-captioning / VLM (Vision-Language) pipeline and a Streamlit-based explainability dashboard. The project includes data preprocessing, dataset building, training scripts for several models (baseline CNN+Transformer, metadata-enhanced models, and optional experiments with BLIP2 / Florence2 / vLLM), evaluation, and a Streamlit UI for running the caption generator and explainability visualizations.

## Results:

ine-tuned a 48M-parameter CNN-Transformer encoder-decoder captioner on 45K images from Myntra Fashion dataset, reaching 0.82
CIDEr and 71% attribute-level F1—outperforming zero-shot Qwen2-VL-2B (0.44 CIDEr) and BLIP-2 (0.31) on catalog-style garment
attributes at 18x fewer parameters.
• Cut p95 batch captioning latency 41% (1,240ms to 730ms) and raised throughput to 11 images/sec on a single A100, by migrating
inference from a per-image HuggingFace generate loop to vLLM with continuous batching and tuned maxnumseqsand pre f ixcaching.
• Validated attribute grounding at 63% region-agreement against 80 hand-annotated garment images (4 attribute classes) using decoder
cross-attention rollout, surfacing that the model inferred fabric type from silhouette rather than texture and informing a targeted retraining
pass.
• Collected 120 human preference judgments across 3 reviewers via a Streamlit A/B interface, producing the error taxonomy used to
prioritize captioning failure modes.


## Prerequisites

* Python 3.9+ recommended.
* `git` installed.
* GPU recommended for training (CUDA + drivers installed) but not required for running preprocessing or the Streamlit demo.



## Quick start (complete flow)

### 0. Clone the repository

```bash
git clone https://github.com/vishalgwu/Image_caption_generator.git
cd Image_caption_generator
```

### 1. Create and activate a virtual environment

Unix / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows (cmd.exe):

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

Upgrade pip and install dependencies:

```bash
pip install --upgrade pip
# The repository may include either `requirements.txt` or `req.txt`.
# Preferred file name: requirements.txt. If your repo has req.txt, use that instead.
pip install -r requirements.txt
# or (if present)
# pip install -r req.txt
```





## 1. Folder Structure

```text
Image_caption_generator/
├── Group-Proposal/
├── images/                           # Raw fashion images (JPEG/PNG)
│   └── ...                           # e.g. 15970.jpg, 39386.jpg, ...
├── metadata/
│   ├── styles.csv                    # Original Myntra metadata
│   ├── merged.parquet                # Metadata merged + cleaned
│   ├── metadata_with_captions.parquet# Final metadata + captions
│   ├── train.parquet                 # Train split
│   ├── val.parquet                   # Validation split
│   ├── test.parquet                  # Test split
│   ├── baseline_eval.csv             # Baseline metrics
│   ├── model1_eval.csv               # Stronger model metrics
│   ├── vocab.json                    # Token → id mapping
│   └── vocab.pkl                     # Pickled vocab object (for PyTorch)
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── captions.py               # Caption text templates/helpers
│   │   ├── collate.py                # Baseline dataloader collate fn
│   │   ├── collate_model1.py         # Model1 collate fn
│   │   ├── dataset.py                # Baseline dataset
│   │   ├── dataset_model1.py         # Model1 dataset (extra metadata)
│   │   ├── dataset_vlm.py            # Dataset for VLM evaluation
│   │   ├── debug.py
│   │   ├── text_preprocess.py        # Text cleaning/tokenization utils
│   │   ├── utils.py
│   │   └── vocab.py                  # FashionVocab class
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── eval_baseline.py          # Evaluate baseline model
│   │   ├── eval_model1.py            # Evaluate stronger model
│   │   └── eval_vllm.py              # Evaluate VLM-based models
│   ├── models/
│   │   ├── __init__.py
│   │   ├── caption_model.py          # Baseline CNN + Transformer
│   │   ├── encoder_resnet50.py       # ResNet-50 image encoder
│   │   ├── decoder_transformer.py    # Transformer decoder
│   │   ├── model1.py                 # Stronger multimodal model
│   │   └── blip2_model.py            # BLIP-2 / Florence2 wrappers
│   ├── train/
│   │   ├── __init__.py
│   │   ├── train_baseline.py         # Train baseline model
│   │   ├── train_model1.py           # Train stronger model
│   │   ├── train_blip2_lora.py       # LoRA finetuning for BLIP-2
│   │   ├── train_florence2.py        # Florence2 training script
│   │   └── train_vllm_blip2.py       # vLLM-based training/eval
│   └── vlm/
│       ├── __init__.py
│       ├── metadata.py               # Load & clean styles.csv
│       ├── merge.py                  # Merge metadata with image info
│       ├── image_ing.py              # Image ingestion checks
│       ├── make_splits.py            # Train/val/test splits
│       ├── build_vocab.py            # Build vocabulary from captions
│       ├── build_captions.py         # Generate training captions parquet
│       ├── qwen2_inference.py        # Run Qwen2-VL on images
│       └── eval_qwen2.py             # Compare Qwen2-VL vs baseline
├── app.py                            # Streamlit dashboard (main entrypoint)
├── baseline_best.pth                 # Trained baseline model checkpoint
├── req.txt                           # Python dependencies
├── LICENSE-CC-BY-4.0.txt
├── vocab.pkl                         # (duplicate copy if needed by app)
└── README.md                         # <-- this file

```





---

## 2. One-time data preprocessing pipeline

**Requirements before running the pipeline**

* `images/` should contain your dataset images (Myntra, ABO, or custom dataset folders).
* `metadata/styles.csv` (or your metadata CSV) should be present and contain the ground-truth metadata for images.

Run preprocessing steps in order (each step writes outputs into `metadata/`):

```bash
# 2.1 Clean metadata → metadata/merged.parquet
python src/vlm/metadata.py

# 2.2 Merge metadata with image paths → updates merged.parquet with absolute/relative image paths
python src/vlm/merge.py

# 2.3 Optional: verify images folder (checks for missing/corrupt images)
python src/vlm/image_ing.py

# 2.4 Create train/val/test splits
python src/vlm/make_splits.py

# 2.5 Build vocabulary (vocab.json, vocab.pkl)
python src/vlm/build_vocab.py

# 2.6 Build final caption dataset (metadata/metadata_with_captions.parquet plus train/val/test parquet files)
python src/vlm/build_captions.py
```

**Expected outputs (metadata/):**

* `merged.parquet`
* `train.parquet`, `val.parquet`, `test.parquet`
* `vocab.json`, `vocab.pkl`
* `metadata_with_captions.parquet`

---

## 3. Model training & evaluation (optional)

> NOTE: Training scripts assume you have an appropriate GPU and model weights/configs set up. Check `src/train/*.py` for hyperparameters and data paths.

```bash
# 3.1 Train baseline CNN + Transformer
python src/train/train_baseline.py

# 3.2 Evaluate baseline (BLEU / ROUGE etc.) → outputs metadata/baseline_eval.csv
python src/eval/eval_baseline.py

# 3.3 Train metadata-enhanced Model1
python src/train/train_model1.py

# 3.4 Evaluate Model1 → outputs metadata/model1_eval.csv
python src/eval/eval_model1.py

# 3.5 Optional experiments: BLIP2 / Florence2 / vLLM
python src/train/train_blip2_lora.py
python src/train/train_florence2.py
python src/train/train_vllm_blip2.py
python src/eval/eval_vllm.py
```
<p align="center">
  <img src="https://github.com/vishalgwu/Image_caption_generator/blob/main/Project_images/Image_caption_generator-1.png" width="600">
</p>
<p align="center">
  <img src="https://github.com/vishalgwu/Image_caption_generator/blob/main/Project_images/Qwan_model_imp_pic.png" width="600">
</p>
<p align="center">
  <img src="https://github.com/vishalgwu/Image_caption_generator/blob/main/Project_images/model_compare_keyword.png" width="600">
</p>


---

## 4. Qwen2-VL explainability pipeline

> This step runs inference with Qwen2-VL (if available) and computes token importance and similarity metrics used by the Streamlit explainability UI.

```bash
# 4.1 Generate Qwen2-VL captions + token importance
python src/vlm/qwen2_inference.py

# 4.2 Compare baseline vs Qwen2-VL captions (metrics, heatmaps, tables)
python src/vlm/eval_qwen2.py
```

Outputs used by Streamlit:

* Token importance CSVs and heatmap files
* Caption similarity metrics CSVs
* Caption tables for baseline vs Qwen2-VL

---


## 5. Launch Streamlit dashboard (explainability + caption generator)

```bash
streamlit run app.py
```

Open the UI in your browser at `http://localhost:8501`.

Streamlit tabs:

* **Caption Generator** — interactively generate captions for uploaded images.
* **Explainability (Baseline + Qwen2-VL)** — compare captions, view token importances and heatmaps.

---
<p align="center">
  <img src="https://github.com/vishalgwu/Image_caption_generator/blob/main/Project_images/Difference_through_heatmap.png" width="650">
</p>

## 6. Troubleshooting & common fixes

* **requirements file not found**: some forks use `req.txt`. If you see an error, either rename `requirement.txt` → `req.txt` or install with `pip install -r req.txt`.
* **Missing images in `images/`**: run `python src/vlm/image_ing.py` to get a report of missing or corrupted files.
* **Path problems (Windows)**: use forward or double-backslashes for paths in config files. Prefer relative paths (`images/`, `metadata/`) wherever possible.
* **GPU issues**: ensure CUDA toolkit and drivers are compatible with the installed PyTorch / TensorFlow versions.
* **Streamlit caching / stale data**: stop Streamlit and restart if you modify underlying CSV / parquet files.

---


---

---

## 9. Citations

* Qwen2-VL (paper): Peng Wang et al., *Qwen2-VL*, arXiv:2409.12191 (2024).

---

## 10. Contributing

1) VISHAL FULSUNDAR
2) Amrutha Jayachandradhara 


Instructor: 
Prof.Aamir Jafari 

---






















