# app.py
# Image Caption Generator + Explainability (Baseline + Qwen2-VL)

import os
import re
import warnings

import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

from transformers import AutoProcessor, AutoModelForVision2Seq
from src.models.caption_model import CaptionModel
from src.data.vocab import FashionVocab

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Torch Compiler Patch (Fix for Qwen Image Processor)
# -------------------------------------------------------------------
if not hasattr(torch, "compiler"):
    class _FakeCompiler:
        @staticmethod
        def is_compiling():
            return False
    torch.compiler = _FakeCompiler()
elif not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

# -------------------------------------------------------------------
# Absolute Paths
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.pkl")
BASELINE_MODEL_PATH = os.path.join(BASE_DIR, "baseline_best.pth")

# -------------------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Image Caption Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    "<h1 style='text-align:center;'>Image Semantics Explorer- Emphasizes explainability, caption understanding.)</h1>",
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# Load Baseline Transformer
# -------------------------------------------------------------------
@st.cache_resource
def load_baseline():
    vocab = FashionVocab.load(VOCAB_PATH)
    model = CaptionModel(vocab=vocab, d_model=512)
    model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, vocab

baseline_model, vocab = load_baseline()

baseline_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

def generate_baseline_caption(img_pil, model, vocab, max_len=30):
    img_tensor = baseline_transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        output_ids = model.generate(img_tensor, max_len=max_len)[0].tolist()
    caption = vocab.decode(output_ids)
    return caption, img_tensor

def baseline_confidence_from_tokens(tokens):
    if not tokens:
        return 0.0
    unk_count = sum(1 for t in tokens if "<unk>" in t.lower())
    return float(max(0.0, 1.0 - unk_count / len(tokens)))

# -------------------------------------------------------------------
# Qwen2-VL Loader
# -------------------------------------------------------------------
@st.cache_resource
def load_qwen2():
    MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )
    model.eval()
    return processor, model

processor, qwen_model = load_qwen2()

# Build Qwen Inputs
def build_qwen_inputs(img_pil, prompt: str):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_pil},
            {"type": "text", "text": prompt},
        ],
    }]

    chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(text=chat_prompt, images=img_pil, return_tensors="pt")

# Clean Qwen Output
def clean_qwen_output(text: str):
    s = text.strip()
    lower = s.lower()
    marker = "assistant"
    idx = lower.rfind(marker)
    if idx != -1:
        s = s[idx + len(marker):]
    for m in ["system", "user"]:
        s = re.sub(rf"\b{m}\b[: ]*", "", s, flags=re.IGNORECASE)
    parts = [p.strip() for p in re.split(r"[.\n]", s) if p.strip()]
    return parts[-1] if parts else s.strip()

# Generate Qwen Caption
def generate_qwen_caption(img_pil, processor, model, max_new_tokens=40):
    prompt = "Describe this fashion product in one concise sentence."
    inputs = build_qwen_inputs(img_pil, prompt)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    clean = clean_qwen_output(raw)
    return clean if clean else raw

def qwen_confidence_from_scores(scores):
    if scores is None or len(scores) == 0:
        return 0.0
    return float(np.clip(np.mean(scores), 0.0, 1.0))

# -------------------------------------------------------------------
# SentenceTransformer for Semantic Explainability
# -------------------------------------------------------------------
@st.cache_resource
def load_semantic_encoder():
    if SentenceTransformer is None:
        st.error("Install: pip install sentence-transformers")
        return None
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

semantic_encoder = load_semantic_encoder()

# Tokenize
def tokenize_caption(caption):
    raw = caption.strip().split()
    return [t.strip(".,!?;:") for t in raw if t.strip(".,!?;:")]

# Qwen Semantic Explainability
def explain_qwen_semantic(caption, encoder):
    if encoder is None:
        return [], np.array([])
    caption = caption.strip()
    if not caption:
        return [], np.array([])

    tokens = tokenize_caption(caption)
    if not tokens:
        return [], np.array([])

    full_emb = encoder.encode([caption], normalize_embeddings=True)[0]

    masked_caps = []
    for i in range(len(tokens)):
        masked = tokens.copy()
        masked[i] = "[MASK]"
        masked_caps.append(" ".join(masked))

    masked_embs = encoder.encode(masked_caps, normalize_embeddings=True)
    sims = (masked_embs * full_emb).sum(axis=1)
    scores = 1.0 - sims
    scores = np.maximum(scores, 0)
    if scores.max() > 0:
        scores = scores / scores.max()
    return tokens, scores
# -------------------------------------------------------------------
# Visualization Helpers
# -------------------------------------------------------------------
def render_colored_caption(tokens, scores, label):
    if not tokens:
        st.info(f"No tokens to display for {label}")
        return
    max_s = max(scores)
    spans = []
    for tok, s in zip(tokens, scores):
        alpha = 0.2 + 0.6 * (s / max_s)
        color = f"rgba(255, 99, 71, {alpha:.2f})"
        spans.append(
            f"<span style='background:{color}; padding:3px; margin:2px; "
            f"border-radius:5px; display:inline-block;'>{tok}</span>"
        )
    st.markdown(" ".join(spans), unsafe_allow_html=True)

def plot_bar_chart(tokens, scores, title):
    if len(tokens) == 0:
        st.info("No token importance available.")
        return
    import pandas as pd
    import altair as alt
    df = pd.DataFrame({"token": tokens, "importance": scores})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("token", sort=None),
            y=alt.Y("importance", scale=alt.Scale(domain=[0, 1])),
            tooltip=["token", "importance"]
        )
        .properties(width=900, height=250, title=title)
    )
    st.altair_chart(chart, use_container_width=True)

# Jaccard Similarity (Caption Overlap)
def jaccard_similarity(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0

# -------------------------------------------------------------------
# Streamlit Tabs
# -------------------------------------------------------------------
tab1, tab2 = st.tabs(["📌 Caption Generator", "🧪 Explainability (Baseline + Qwen)"])

# ===================== TAB 1 =============================
with tab1:
    st.subheader("Upload an Image")

    uploaded = st.file_uploader("Upload", type=["jpg", "jpeg", "png"])

    model_choice = st.selectbox("Choose model", ["Baseline Transformer", "Qwen2-VL"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=350)

        if st.button("Generate Caption"):
            if model_choice == "Baseline Transformer":
                caption, _ = generate_baseline_caption(img, baseline_model, vocab)
            else:
                caption = generate_qwen_caption(img, processor, qwen_model)
            st.success(caption)
# ===================== TAB 2 =============================
with tab2:
    st.subheader("Explainability")

    exp_file = st.file_uploader(
        "Upload image (both models analyzed)", type=["jpg", "jpeg", "png"], key="exp"
    )

    if exp_file:
        exp_img = Image.open(exp_file).convert("RGB")
        st.image(exp_img, width=260)

        if st.button("Run Explainability"):

            # ----------------- BASELINE -----------------
            st.markdown("## 🔵 Baseline Transformer")
            base_caption, img_tensor = generate_baseline_caption(
                exp_img, baseline_model, vocab
            )
            st.success(base_caption)

            try:
                _, base_tokens, base_scores = baseline_model.explain_tokens(
                    img_tensor, max_len=30
                )
                base_scores = (
                    base_scores.detach().cpu().numpy()
                    if hasattr(base_scores, "detach")
                    else np.asarray(base_scores)
                )
            except Exception as e:
                st.error(f"Baseline explainability error:\n{e}")
                base_tokens, base_scores = [], np.array([])

            base_conf = baseline_confidence_from_tokens(base_tokens)
            st.markdown(f"**Confidence:** `{base_conf:.2f}`")

            if base_tokens:
                plot_bar_chart(base_tokens, base_scores, "Baseline Token Importance")
                render_colored_caption(base_tokens, base_scores, "Baseline")

            st.markdown("---")

            # ----------------- QWEN -----------------
            st.markdown("## 🟣 Qwen2-VL (Semantic Explainability)")
            q_caption = generate_qwen_caption(exp_img, processor, qwen_model)
            st.success(q_caption)

            q_tokens, q_scores = explain_qwen_semantic(q_caption, semantic_encoder)
            q_conf = qwen_confidence_from_scores(q_scores)
            st.markdown(f"**Confidence:** `{q_conf:.2f}`")

            if q_tokens:
                plot_bar_chart(q_tokens, q_scores, "Qwen Semantic Importance")
                render_colored_caption(q_tokens, q_scores, "Qwen2-VL")

            st.markdown("---")

            # ----------------- COMPARISON -----------------
            st.markdown("## 🔄 Baseline vs Qwen Comparison")

            base_set = set(t.lower() for t in base_tokens)
            q_set = set(t.lower() for t in q_tokens)

            sim = jaccard_similarity(base_set, q_set)
            st.markdown(f"**Token-overlap similarity:** `{sim:.2f}`")

            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Baseline-only Tokens:**")
                base_only = sorted(base_set - q_set)
                st.write(", ".join(base_only) if base_only else "—")

            with cols[1]:
                st.markdown("**Qwen-only Tokens:**")
                q_only = sorted(q_set - base_set)
                st.write(", ".join(q_only) if q_only else "—")

            st.markdown("---")

            # ----------------- HEATMAP (2 Columns Only) -----------------
            st.markdown("## 🔥 Token Importance Heatmap")

            if len(base_tokens) == 0 or len(q_tokens) == 0:
                st.info("Need valid importance values from both models.")
            else:
                import pandas as pd
                import altair as alt

                all_tokens = list(dict.fromkeys(base_tokens + q_tokens))
                trans_scores_aligned = []
                qwen_scores_aligned = []

                for tok in all_tokens:
                    trans_scores_aligned.append(
                        float(base_scores[base_tokens.index(tok)]) if tok in base_tokens else 0.0
                    )
                    qwen_scores_aligned.append(
                        float(q_scores[q_tokens.index(tok)]) if tok in q_tokens else 0.0
                    )

                df = pd.DataFrame({
                    "token": all_tokens,
                    "Transformer": trans_scores_aligned,
                    "Qwen": qwen_scores_aligned
                })

                df_melt = df.melt(id_vars="token", var_name="Model", value_name="Importance")

                heatmap = (
                    alt.Chart(df_melt)
                    .mark_rect()
                    .encode(
                        x=alt.X("token:N", sort=None),
                        y=alt.Y("Model:N"),
                        color=alt.Color("Importance:Q", scale=alt.Scale(scheme="reds")),
                        tooltip=["token", "Model", "Importance"],
                    )
                    .properties(width=800, height=200)
                )

                st.altair_chart(heatmap, use_container_width=True)

