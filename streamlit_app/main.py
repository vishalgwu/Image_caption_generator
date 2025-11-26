import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

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
    caption_words = ["A", "stunning", color.lower(), category.lower(), "perfect", "for", season.lower()]
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
            caption_words, attention_maps = mock_predict(image, metadata)
            full_caption = " ".join(caption_words)
            
            # C. DISPLAY CAPTION
            with col2:
                st.subheader("Generated Caption")
                st.success(f"**{full_caption}**")
                
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
                
                shap_vals = mock_shap_values(metadata)
                
                # Simple Bar Plot for SHAP
                fig_shap, ax_shap = plt.subplots(figsize=(6, 3))
                sns.barplot(x=list(shap_vals.values()), y=list(shap_vals.keys()), ax=ax_shap, palette="viridis")
                ax_shap.set_xlabel("Impact on Caption")
                st.pyplot(fig_shap)

else:
    with col1:
        st.info("Please upload an image to start.")