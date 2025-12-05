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

