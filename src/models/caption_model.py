# src/models/caption_model.py
import torch
import torch.nn as nn

from src.models.encoder_resnet50 import ResNet50Encoder
from src.models.decoder_transformer import TransformerDecoderModule


class CaptionModel(nn.Module):
    """
    Full captioning model:
    - ResNet50 encoder (pretrained on ImageNet)
    - Linear projection (2048 -> d_model)
    - Transformer decoder
    """

    def __init__(self, vocab, d_model: int = 512, train_cnn: bool = False):
        super().__init__()

        # vocab
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.pad_idx = vocab.pad_idx
        self.bos_idx = vocab.sos_idx
        self.eos_idx = vocab.eos_idx

        # 1) Encoder
        self.encoder = ResNet50Encoder(train_cnn=train_cnn)

        # 2) Project encoder output to d_model
        self.img_proj = nn.Linear(self.encoder.out_dim, d_model)

        # 3) Transformer decoder
        self.decoder = TransformerDecoderModule(
            vocab_size=self.vocab_size,
            d_model=d_model,
            num_heads=8,
            num_layers=4,
            dim_feedforward=2048,
            dropout=0.1,
            pad_idx=self.pad_idx,
        )

    def forward(self, images, captions_inp):
        """
        images: (B, 3, 224, 224)
        captions_inp: (B, T)  (shifted right input)
        returns: logits (B, T, vocab_size)
        """
        img_feats = self.encoder(images)        # (B, 2048)
        img_emb = self.img_proj(img_feats)      # (B, d_model)
        memory = img_emb.unsqueeze(1)           # (B, 1, d_model)

        logits = self.decoder(captions_inp, memory)
        return logits

    @torch.no_grad()
    def generate(self, images, max_len: int = 30):
        """
        Greedy decoding.
        images: (B, 3, 224, 224)
        returns: (B, T) generated token IDs
        """
        self.eval()
        device = images.device
        B = images.size(0)

        img_feats = self.encoder(images)
        img_emb = self.img_proj(img_feats)
        memory = img_emb.unsqueeze(1)  # (B, 1, d_model)

        captions = torch.full(
            (B, 1),
            self.bos_idx,
            dtype=torch.long,
            device=device,
        )  # start with <sos>

        for _ in range(max_len - 1):
            logits = self.decoder(captions, memory)   # (B, t, vocab)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            captions = torch.cat([captions, next_token], dim=1)

            # if all sequences ended, break
            if torch.all(next_token.squeeze(1) == self.eos_idx):
                break

        return captions

    def explain_tokens(self, images, max_len: int = 30):
        """
        Compute token-level importance for a single image.

        images: (1, 3, 224, 224)
        returns:
          caption_text: str
          tokens: List[str] (w/o special tokens)
          importance: List[float] (0–1, aligned with tokens)
        """
        self.eval()
        if images.size(0) != 1:
            raise ValueError("explain_tokens currently supports batch_size=1 only.")

        device = images.device

        # 1) Generate caption ids (no gradient)
        with torch.no_grad():
            gen_ids = self.generate(images, max_len=max_len)  # (1, T)

        # 2) Compute encoder memory
        img_feats = self.encoder(images)
        img_emb = self.img_proj(img_feats)
        memory = img_emb.unsqueeze(1)  # (1, 1, d_model)

        # 3) Compute token importance via decoder
        self.zero_grad(set_to_none=True)
        importance_tensor = self.decoder.compute_token_importance(gen_ids, memory)  # (T,)

        # 4) Map ids -> tokens, filter special tokens
        token_ids = gen_ids[0].tolist()
        raw_tokens = [self.vocab.idx2word[idx] for idx in token_ids]

        tokens = []
        importance = []
        for tok, score in zip(raw_tokens, importance_tensor.tolist()):
            if tok in ("<pad>", "<sos>", "<eos>"):
                continue
            tokens.append(tok)
            importance.append(score)

        caption_text = " ".join(tokens)
        return caption_text, tokens, importance
