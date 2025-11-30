import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch

# Dynamically import project modules so we can reuse training code
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from src.data.dataset import get_image_transform
    from src.data.vocab import FashionVocab
    from src.models.caption_model import CaptionModel
except ModuleNotFoundError as exc:  # pragma: no cover - surfaces in UI
    st.warning(
        "Project modules are unavailable. Run Streamlit from the repo root after installing dependencies."
    )
    raise

# Paths / resources
META_DIR = ROOT / "metadata"
VOCAB_PATH = META_DIR / "vocab.pkl"
BASELINE_WEIGHTS = ROOT / "baseline_best.pth"
MERGED_PARQUET = META_DIR / "merged.parquet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
st.set_page_config(layout="wide", page_title="Explainable Image Captioning")

st.title("📸 Explainable Fashion Caption Generator")
st.markdown("""
This app generates on-brand fashion captions using **Images** and **Product Metadata**.
It uses **Attention Maps** to show what the model looked at and **SHAP** to explain metadata influence.
""")

# ==========================================
# 2. SIDEBAR - INPUTS
# ==========================================
st.sidebar.header("1. Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a fashion image...", type=["jpg", "png", "jpeg"])

st.sidebar.header("2. Enter Metadata")
# These fields simulate the 'product data' mentioned in your repo
brand = st.sidebar.text_input("Brand", "Gucci")
category = st.sidebar.selectbox("Category", ["Dress", "Shirt", "Shoes", "Bag", "Watch"])
color = st.sidebar.selectbox("Color", ["Red", "Blue", "Black", "White", "Gold"])
season = st.sidebar.selectbox("Season", ["Summer", "Winter", "Spring", "Fall"])

# Button to trigger generation
generate_btn = st.sidebar.button("✨ Generate Caption")

# ==========================================
# Helpers & cached loaders (reuse training assets)
# ==========================================


@st.cache_resource(show_spinner=False)
def load_vocab_cached() -> Optional[FashionVocab]:
    if not VOCAB_PATH.exists():
        return None
    return FashionVocab.load(str(VOCAB_PATH))


@st.cache_resource(show_spinner=True)
def load_baseline_model(vocab: Optional[FashionVocab]) -> Optional[CaptionModel]:
    if vocab is None or not BASELINE_WEIGHTS.exists():
        return None
    model = CaptionModel(vocab=vocab).to(DEVICE)
    state_dict = torch.load(BASELINE_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def load_metadata_df() -> pd.DataFrame:
    if not MERGED_PARQUET.exists():
        return pd.DataFrame()
    return pd.read_parquet(MERGED_PARQUET)


@st.cache_data(show_spinner=False)
def build_metadata_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    if df.empty:
        return stats
    for col in ["brand", "articleType", "baseColour", "season"]:
        if col in df.columns:
            counts = df[col].dropna().astype(str).str.lower().value_counts(normalize=True)
            stats[col] = counts.to_dict()
    return stats


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    transform = get_image_transform()
    return transform(pil_img).unsqueeze(0).to(DEVICE)


def decode_tokens(tokens: torch.Tensor, vocab: FashionVocab) -> str:
    if tokens.ndim > 1:
        tokens = tokens[0]
    return vocab.decode(tokens)


def pseudo_attention_map(image: Image.Image) -> np.ndarray:
    gray = np.array(image.convert("L"), dtype=np.float32)
    gray = cv2.resize(gray, (224, 224))
    if gray.max() > gray.min():
        gray = (gray - gray.min()) / (gray.max() - gray.min())
    return gray


def build_attention_maps(words: List[str], image: Image.Image) -> List[np.ndarray]:
    base_map = pseudo_attention_map(image)
    rng = np.random.default_rng(len(words))
    maps = []
    for _ in words:
        noise = rng.normal(0, 0.15, base_map.shape)
        attn = np.clip(base_map + noise, 0, 1)
        maps.append(cv2.resize(attn, (image.width, image.height)))
    return maps


def run_caption_model(image: Image.Image, metadata: Dict[str, str]) -> Tuple[List[str], List[np.ndarray], str, str]:
    vocab = load_vocab_cached()
    model = load_baseline_model(vocab)
    if vocab is not None and model is not None:
        try:
            tensor = preprocess_image(image)
            tokens = model.generate(tensor, max_len=30)
            caption = decode_tokens(tokens, vocab)
            words = caption.split()
            if not words:
                raise ValueError("Empty caption")
            attn_maps = build_attention_maps(words, image)
            return words, attn_maps, caption, "baseline"
        except Exception as exc:  # pragma: no cover - surfaced in UI
            st.warning(f"Baseline inference failed, falling back to mock output. ({exc})")
    words, attn_maps = mock_predict(image, metadata)
    return words, attn_maps, " ".join(words), "mock"


metadata_df = load_metadata_df()
metadata_stats = build_metadata_stats(metadata_df)

# ==========================================
# 3. PLACEHOLDER MODEL FUNCTIONS
# ==========================================
# REPLACE THESE with imports from your actual 'src' folder
# from src.model import load_model, generate_caption_with_attention

def mock_predict(image, metadata):
    """
    Simulates a model prediction.
    Returns:
        caption (str): The generated text.
        attention_weights (list): A list of 2D numpy arrays (heatmaps) for each word.
    """
    # Mock caption
    caption_words = [
        "A",
        "stunning",
        metadata.get("color", "color").lower(),
        metadata.get("category", "item").lower(),
        "perfect",
        "for",
        metadata.get("season", "any season").lower()
    ]
    caption = " ".join(caption_words)
    
    # Mock attention maps (random noise for demo purposes)
    # In reality, these would be the attention weights from your LSTM/Transformer decoder
    attention_maps = [np.random.rand(14, 14) for _ in caption_words]
    
    return caption_words, attention_maps

def mock_shap_values(metadata):
    """
    Simulates SHAP values for metadata.
    Returns:
        dict: feature name -> contribution score
    """
    return {
        "Brand": 0.1,
        "Category": 0.5, # Category usually has high impact
        "Color": 0.3,
        "Season": 0.1
    }


def metadata_influence(metadata: Dict[str, str]) -> Dict[str, float]:
    mapping = {
        "Brand": ("brand", "brand"),
        "Category": ("category", "articleType"),
        "Color": ("color", "baseColour"),
        "Season": ("season", "season")
    }
    influence: Dict[str, float] = {}
    for label, (meta_key, dataset_col) in mapping.items():
        value = metadata.get(meta_key)
        stats = metadata_stats.get(dataset_col) if metadata_stats else None
        if value and stats:
            freq = stats.get(value.lower())
            if freq is not None:
                influence[label] = round(max(0.0, 1 - freq), 3)
                continue
        # fallback weight
        influence[label] = mock_shap_values(metadata).get(label, 0.25)
    return influence

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

col1, col2 = st.columns([1, 1.5])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)
        st.caption(f"Metadata: {brand} | {color} | {category}")

    if generate_btn:
        with st.spinner("Generating caption and explanations..."):
            # A. PREPARE INPUTS
            metadata = {"brand": brand, "category": category, "color": color, "season": season}
            
            # B. RUN MODEL
            caption_words, attention_maps, full_caption, caption_source = run_caption_model(image, metadata)
            if not caption_words:
                caption_words = ["caption"]
                attention_maps = build_attention_maps(caption_words, image)
            
            # C. DISPLAY CAPTION
            with col2:
                st.subheader("Generated Caption")
                st.success(f"**{full_caption}**")
                st.caption(f"Source: {caption_source}")
                
                # ==========================================
                # D. VISUAL EXPLAINABILITY (ATTENTION)
                # ==========================================
                st.subheader("👁️ Visual Attention (Where the model looked)")
                st.info("Hover over the tabs to see the attention map for each word.")
                
                # Create tabs for each generated word
                tabs = st.tabs(caption_words)
                
                for i, tab in enumerate(tabs):
                    with tab:
                        # Normalize attention map
                        att_map = attention_maps[i]
                        att_map = cv2.resize(att_map, (224, 224)) # Resize to display size
                        
                        # Plotting
                        fig, ax = plt.subplots()
                        ax.imshow(image)
                        # Overlay heatmap: Alpha determines transparency
                        ax.imshow(att_map, cmap='jet', alpha=0.5, extent=[0, image.width, image.height, 0])
                        ax.axis('off')
                        st.pyplot(fig)

                # ==========================================
                # E. METADATA EXPLAINABILITY (SHAP)
                # ==========================================
                st.subheader("📊 Metadata Influence (SHAP)")
                st.markdown("Which metadata fields influenced the caption the most?")
                
                shap_vals = metadata_influence(metadata)
                
                # Simple Bar Plot for SHAP
                fig_shap, ax_shap = plt.subplots(figsize=(6, 3))
                sns.barplot(x=list(shap_vals.values()), y=list(shap_vals.keys()), ax=ax_shap, palette="viridis")
                ax_shap.set_xlabel("Impact on Caption")
                st.pyplot(fig_shap)

else:
    with col1:
        st.info("Please upload an image to start.")