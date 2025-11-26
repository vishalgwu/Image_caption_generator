import torch
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu

from src.data.vocab import FashionVocab
from src.models.caption_model import CaptionModel
from src.data.dataset import FashionDataset
from src.data.dataset_model1 import simple_tokenize


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


def evaluate_baseline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Evaluating baseline on:", device)

    # 1. Load vocab
    vocab = FashionVocab.load("metadata/vocab.pkl")
    print("Vocab size:", len(vocab))

    # 2. Load test dataset
    test_ds = FashionDataset(
        parquet_path="metadata/test.parquet",
        vocab=vocab,
        images_dir="images",
        max_len=30,
        caption_column="caption"
    )

    test_df = pd.read_parquet("metadata/test.parquet")

    # 3. Load baseline model
    model = CaptionModel(vocab, d_model=512, train_cnn=False)
    model.load_state_dict(torch.load("baseline_best.pth", map_location=device))
    model.to(device)
    model.eval()

    references = []
    candidates = []

    print("Generating predictions...")
    with torch.no_grad():
        for idx in range(len(test_ds)):
            image, caption_ids = test_ds[idx]

            # ground truth
            ref_caption = test_df.iloc[idx]["caption"]
            ref_tokens = simple_tokenize(ref_caption)
            references.append([ref_tokens])

            # prediction
            img_batch = image.unsqueeze(0).to(device)
            out = model.generate(img_batch, max_len=30)
            out = out[0] if torch.is_tensor(out) else out

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

    print("\n=== Baseline BLEU Scores ===")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")

    pd.DataFrame({
        "id": test_df["id"],
        "reference": [" ".join(r[0]) for r in references],
        "prediction": [" ".join(c) for c in candidates]
    }).to_csv("metadata/baseline_eval.csv", index=False)

    print("Saved results → metadata/baseline_eval.csv")


if __name__ == "__main__":
    evaluate_baseline()
