import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T

print(">>> Streamlit app started...")

# -----------------------------
# BASELINE MODEL SETUP
# -----------------------------
print(">>> Loading baseline imports...")

from src.models.caption_model import CaptionModel
from src.data.vocab import FashionVocab

print(">>> Baseline imports loaded successfully.")


@st.cache_resource
def load_baseline_model():
    print(">>> Loading baseline vocab & model...")

    # Load vocab from root folder
    vocab = FashionVocab.load("vocab.pkl")

    # Correct CaptionModel constructor
    model = CaptionModel(
        vocab=vocab,
        d_model=512
    )

    # Load saved weights
    model.load_state_dict(torch.load("baseline_best.pth", map_location="cpu"))
    model.eval()

    print(">>> Baseline model loaded.")
    return model, vocab


# Image transform for baseline
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])


def generate_baseline_caption(img_pil, model, vocab, max_len=25):
    img_tensor = transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        output_ids = model.generate(
            img_tensor,
            max_len=max_len
        )[0].tolist()

    caption = vocab.decode(output_ids)
    return caption


print(">>> Baseline code ready.")

# -----------------------------
# QWEN2-VL MODEL SETUP
# -----------------------------
print(">>> Preparing Qwen2 loader (lazy-load)...")


@st.cache_resource
def load_qwen_model():
    print(">>> Importing transformers inside Qwen2 loader...")
    from transformers import AutoProcessor, AutoModelForVision2Seq

    MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

    # FIX: use_fast=False to avoid torch.compiler error
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    ).to("cpu")

    print(">>> Qwen2-VL model loaded successfully.")
    return processor, model


def generate_qwen_caption(img_pil, processor, model):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_pil},
                {"type": "text",
                 "text": "Describe this fashion product in one concise sentence."}
            ]
        }
    ]

    chat_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        text=chat_prompt,
        images=img_pil,
        return_tensors="pt"
    )

    output = model.generate(**inputs, max_new_tokens=60)
    full = processor.batch_decode(output, skip_special_tokens=True)[0]
    # Qwen2-VL format: "system ... assistant <caption>"
    caption = full.split("assistant")[-1].strip()
    return caption

    return caption.strip()


print(">>> Qwen2 loader ready.")

# -----------------------------
# STREAMLIT UI
# -----------------------------
print(">>> Setting up Streamlit UI...")

st.set_page_config(page_title="Image Caption Generator", layout="wide")
st.title("🖼️ Image Caption Generator — Baseline vs Qwen2-VL")
st.write("Upload an image and compare captions from two models!")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

model_choice = st.selectbox(
    "Choose model",
    ["Both Models", "Baseline Only", "Qwen2-VL Only"]
)

print(">>> UI rendered.")


if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    col1, col2 = st.columns(2)

    # -----------------------------
    # BASELINE CAPTION
    # -----------------------------
    if model_choice in ["Both Models", "Baseline Only"]:
        with col1:
            st.subheader("🧠 Baseline Transformer Caption")

            with st.spinner("Loading baseline model..."):
                model, vocab = load_baseline_model()

            caption = generate_baseline_caption(img, model, vocab)
            st.success(caption)

    # -----------------------------
    # QWEN2-VL CAPTION
    # -----------------------------
    if model_choice in ["Both Models", "Qwen2-VL Only"]:
        with col2:
            st.subheader("🤖 Qwen2-VL Caption")

            with st.spinner("Loading Qwen2-VL model (heavy)..."):
                processor, qwen_model = load_qwen_model()

            caption = generate_qwen_caption(img, processor, qwen_model)
            st.info(caption)

st.write("---")
st.write("💡 More features coming soon: attention maps, Grad-CAM, BLEU score evaluation.")
