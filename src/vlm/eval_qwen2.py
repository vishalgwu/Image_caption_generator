import pandas as pd
import evaluate
from qwen2_inference import generate_caption

VAL_PATH = "../../metadata/val.parquet"

df = pd.read_parquet(VAL_PATH)

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

preds = []
refs = []

print("Evaluating... this will take some time on CPU.")

for idx, row in df.iterrows():
    img_path = f"../../images/{row['id']}.jpg"
    gt_caption = row["caption"]

    try:
        pred_caption = generate_caption(img_path)
    except Exception as e:
        print(f"Error with image {row['id']}: {e}")
        pred_caption = ""

    preds.append(pred_caption)
    refs.append(gt_caption)

# BLEU expects list-of-lists for references
bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])

rouge_score = rouge.compute(predictions=preds, references=refs)

print("\n===== LLaVA Evaluation Results =====")
print("BLEU:", bleu_score)
print("ROUGE:", rouge_score)

df["llava_pred"] = preds
df.to_csv("llava_predictions.csv", index=False)
print("\nSaved predictions to llava_predictions.csv")
