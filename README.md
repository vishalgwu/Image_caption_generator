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
``

# 0. Clone repo and go into project folder

git clone <your-repo-url>.git
cd Image_caption_generator
```
```
# 1. Create + activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip


# 2. Install dependencies
# --------------------------------------------------------
pip install -r req.txt

```
```
# 3. (One-time) Data preprocessing pipeline
#    Assumes:
#       - images/ contains all image files
#       - metadata/styles.csv contains the Myntra metadata

# 3.1 Build cleaned metadata parquet
python src/vlm/metadata.py        # cleans styles.csv -> metadata/merged.parquet

# 3.2 Merge metadata with image availability
python src/vlm/merge.py           # adds image paths, drops missing images

# 3.3 Optional image ingestion checks / thumbnails
python src/vlm/image_ing.py       # verifies images folder

# 3.4 Train/val/test split
python src/vlm/make_splits.py     # outputs train/val/test parquet files

# 3.5 Build vocabulary from training captions
python src/vlm/build_vocab.py     # writes metadata/vocab.json + vocab.pkl

# 3.6 Build caption dataset parquet (text + meta)
python src/vlm/build_captions.py  # writes metadata/metadata_with_captions.parquet
```

```
# 4. Train models (optional if you use shipped checkpoints)

# 4.1 Train baseline CNN + Transformer caption model
python src/train/train_baseline.py     # saves baseline_best.pth

# 4.2 Evaluate baseline on validation/test
python src/eval/eval_baseline.py       # writes metadata/baseline_eval.csv

# 4.3 (Optional) Train stronger metadata-aware model
python src/train/train_model1.py       # saves model1 checkpoint
python src/eval/eval_model1.py         # writes metadata/model1_eval.csv

# 4.4 (Optional) BLIP2 / Florence2 / vLLM experiments
python src/train/train_blip2_lora.py   # LoRA finetuning for BLIP2
python src/train/train_florence2.py
python src/train/train_vllm_blip2.py
python src/eval/eval_vllm.py


```
```
# 5. Qwen2-VL explainability pipeline (semantic comparison)

# 5.1 Run Qwen2-VL to generate captions + token scores
python src/vlm/qwen2_inference.py

# 5.2 Compare Qwen2-VL captions with baseline captions
python src/vlm/eval_qwen2.py

#  -> Produces CSVs used by the Streamlit Explainability tab
#     (token importance, overlap scores, difference heatmaps etc.)
```
```

```
# 6. Launch Streamlit dashboard
streamlit run app.py

# Streamlit will open in the browser (default: http://localhost:8501)
# Use the 'Caption Generator' and 'Explainability (Baseline and  Qwen)' tabs.


```

```

# Environment Setup
Install Python 3.10+.

From project root:
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r req.txt
```
Now
# Data & Metadata

Place data as follows:

All images go into images/. ( Download from the  .@article{Qwen2-VL, title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the 
World at Any Resolution}, Or get dataset from ABO data( amazon dataset) 

The original Myntra metadata CSV is metadata/styles.csv.

Then run the preprocessing scripts (once):

python src/vlm/metadata.py
python src/vlm/merge.py
python src/vlm/image_ing.py
python src/vlm/make_splits.py
python src/vlm/build_vocab.py
python src/vlm/build_captions.py
```
```
#The out put will be like -
Outputs:

metadata/merged.parquet

metadata/train.parquet, val.parquet, test.parquet

metadata/vocab.json, vocab.pkl

metadata/metadata_with_captions.parquet

```

