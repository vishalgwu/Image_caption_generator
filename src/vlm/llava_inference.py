import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# Load processor and model
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
).to(device)


def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")

    # Step 1: chat message format (Qwen2-VL requires this)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this fashion product in one concise sentence."}
            ]
        }
    ]

    # Step 2: create input text (string)
    chat_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    # Step 3: Tokenize + process image into tensors
    inputs = processor(
        text=chat_prompt,
        images=image,
        return_tensors="pt"
    ).to(device)

    # Step 4: Generate caption
    output = model.generate(
        **inputs,
        max_new_tokens=60
    )

    # Step 5: Decode to string
    caption = processor.batch_decode(output, skip_special_tokens=True)[0]
    return caption.strip()


if __name__ == "__main__":
    print(generate_caption("../../images/10010.jpg"))
