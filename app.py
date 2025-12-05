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
# 