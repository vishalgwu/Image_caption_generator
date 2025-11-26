import torch
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from sklearn.preprocessing import LabelEncoder

from src.data.vocab import FashionVocab
from src.models.model1 import Model1
from src.data.dataset_model1 import FashionDatasetV2, simple_tokenize


META_COLS = [
    "gender", "masterCategory", "subCategory",
    "articleType", "baseColour", "season", "usage", "year"
]


def build_meta_encoders(df):
    LE = {}
    meta_sizes = {}
    for col in META_COLS:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col].values)
        LE[col] = le
        meta_sizes[col] = len(le.classes_)
    return LE, meta_sizes


def decode_caption(token_ids, vocab):
    if torch.is_tensor(token_ids):
        token_ids = token_ids.tolist()

    words = []
    for idx in token_ids:
        token = vocab.idx2word[idx]
        if token == "<eos>":
            break
        if token in ("<pad>", "<sos>"):
            continue
        words.append(token)
    return " ".join(words)


def evaluate_model1():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Evaluating Model1 on:", device)

    # 1. vocab
    vocab = FashionVocab.load("metadata/vocab.pkl")

    # 2. metadata encoders
    merged_df = pd.read_parquet("metadata/merged.parquet")
    LE, meta_sizes = build_meta_encoders(merged_df)

    # 3. dataset
    test_df = pd.read_parquet("metadata/test.parquet")
    test_ds = FashionDatasetV2(
        parquet_path="metadata/test.parquet",
        vocab=vocab,
        meta_label_encoders=LE,
        images_dir="images",
        max_len=30
    )

    # 4. load model1
    model = Model1(vocab, meta_sizes)
    model.load_state_dict(torch.load("model1_best.pth", map_location=device))
    model.to(device)
    model.eval()

    references = []
    candidates = []

    print("Generating predictions...")
    with torch.no_grad():
        for idx in range(len(test_ds)):
            image, _, meta_dict = test_ds[idx]

            # reference
            ref_caption = test_df.iloc[idx]["caption"]
            ref_tokens = simple_tokenize(ref_caption)
            references.append([ref_tokens])

            # prepare batch
            img_batch = image.unsqueeze(0).to(device)
            meta_batch = {k: v.unsqueeze(0).to(device)
                          for k, v in meta_dict.items()}

            out = model.generate(img_batch, meta_batch, max_len=30)
            out = out[0]

            pred_caption = decode_caption(out, vocab)
            pred_tokens = simple_tokenize(pred_caption)
            candidates.append(pred_tokens)

            if (idx + 1) % 200 == 0:
                print(f"Processed {idx+1}/{len(test_ds)}")

    # BLEU
    bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, candidates, weights=(1/3, 1/3, 1/3, 0))
    bleu4 = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))

    print("\n=== Model1 BLEU Scores ===")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")

    pd.DataFrame({
        "id": test_df["id"],
        "reference": [" ".join(r[0]) for r in references],
        "prediction": [" ".join(c) for c in candidates]
    }).to_csv("metadata/model1_eval.csv", index=False)

    print("Saved results → metadata/model1_eval.csv")


if __name__ == "__main__":
    evaluate_model1()
