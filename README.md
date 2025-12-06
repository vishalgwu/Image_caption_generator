# Image_caption_generator
Generates on-brand fashion captions using images and product data. 

# 🖼️ Image Caption Generator (Baseline + Qwen2-VL)

Fashion image captioning system with:
- **Baseline CNN + Transformer** caption model (trained from scratch on Myntra dataset).
- **Qwen2-VL** vision-language model for **semantic explainability** and caption comparison.
- **Streamlit dashboard** for image upload, caption generation, token-level importance and model comparison. :contentReference[oaicite:0]{index=0}

---

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
# --------------------------------------------------------
# 0. Clone the repository
# --------------------------------------------------------
git clone <your-repo-url>.git
cd Image_caption_generator

# --------------------------------------------------------
# 1. Create and activate virtual environment
# --------------------------------------------------------
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r req.txt


# ========================================================
# 2. (One-time) Data Preprocessing Pipeline
# ========================================================
# Requirements:
#   - images/ contains all image files (Myntra / ABO / custom dataset)
#   - metadata/styles.csv contains item metadata

# 2.1 Clean metadata → merged.parquet
python src/vlm/metadata.py

# 2.2 Merge metadata with image paths
python src/vlm/merge.py

# 2.3 Optional: verify images folder
python src/vlm/image_ing.py

# 2.4 Create train/val/test parquet splits
python src/vlm/make_splits.py

# 2.5 Build vocabulary
python src/vlm/build_vocab.py

# 2.6 Build caption training parquet
python src/vlm/build_captions.py

# Output Files:
# metadata/merged.parquet
# metadata/train.parquet
# metadata/val.parquet
# metadata/test.parquet
# metadata/vocab.json
# metadata/vocab.pkl
# metadata/metadata_with_captions.parquet


# ========================================================
# 3. Model Training & Evaluation (Optional)
# ========================================================

# 3.1 Train baseline CNN + Transformer caption model
python src/train/train_baseline.py

# 3.2 Evaluate baseline
python src/eval/eval_baseline.py     # writes metadata/baseline_eval.csv

# 3.3 Train metadata-enhanced stronger model
python src/train/train_model1.py

# 3.4 Evaluate stronger model
python src/eval/eval_model1.py       # writes metadata/model1_eval.csv

# 3.5 Optional: BLIP2 / Florence2 / vLLM experiments
python src/train/train_blip2_lora.py
python src/train/train_florence2.py
python src/train/train_vllm_blip2.py
python src/eval/eval_vllm.py


# ========================================================
# 4. Qwen2-VL Semantic Explainability Pipeline
# ========================================================

# 4.1 Generate Qwen2-VL captions + token importance
python src/vlm/qwen2_inference.py

# 4.2 Compare baseline vs Qwen2 captions
python src/vlm/eval_qwen2.py

# Produces CSV files for Streamlit explainability tab:
# - Token importance scores
# - Caption similarity metrics
# - Importance-difference heatmaps
# - Baseline vs Qwen2-VL caption comparisons


# ========================================================
# 5. Launch Streamlit Dashboard
# ========================================================
streamlit run app.py
# Opens at: http://localhost:8501
# Tabs:
#  • Caption Generator
#  • Explainability (Baseline + Qwen2-VL)


# ========================================================
# 6. Environment Reminder
# ========================================================
# Always activate your environment before running scripts:
# source .venv/bin/activate
# Windows: .venv\Scripts\activate


# ========================================================
# 7. Citations
# ========================================================
# Dataset: Myntra / Amazon ABO / or equivalent image dataset
#
# Qwen2-VL Paper:
# @article{Qwen2-VL,
#  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
#  author={Peng Wang and Shuai Bai and Sinan Tan and ... Junyang Lin},
#  journal={arXiv preprint arXiv:2409.12191},
#  year={2024}
# }
