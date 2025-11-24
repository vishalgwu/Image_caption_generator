import torch
import torch.nn as nn

from src.models.encoder_resnet50 import ResNet50Encoder
from src.models.decoder_transformer import TransformerDecoderModule
from src.utils import load_vocab


class CaptionModel(nn.Module):
    """
    Full baseline captioning model:
    - ResNet50 encoder (image -> 2048 features)
    - Linear projection (2048 -> d_model)
    - Transformer decoder (text generation)
    """

    def __init__(self, d_model=512, train_cnn=False):
        super().__init__()

        vocab = load_vocab()
        self.vocab_size = vocab["vocab_size"]
        self.pad_idx = vocab["pad_idx"]
        self.bos_idx = vocab["bos_idx"]   # <sos>
        self.eos_idx = vocab["eos_idx"]   # <eos>

        # 1) Encoder
        self.encoder = ResNet50Encoder(train_cnn=train_cnn)

        # 2) Project image feature (2048) -> decoder dimension (d_model)
        self.img_proj = nn.Linear(2048, d_model)

        # 3) Decoder
        self.decoder = TransformerDecoderModule(
            vocab_size=self.vocab_size,
            d_model=d_model,
            num_heads=8,
            num_layers=4,
            dim_feedforward=2048,
            dropout=0.1,
            pad_idx=self.pad_idx
        )

    def forward(self, images, captions_inp):
        """
        Training forward pass (teacher forcing).

        images: (B, 3, 224, 224)
        captions_inp: (B, T)   -> shifted right caption tokens

        Returns:
        logits: (B, T, vocab_size)
        """
        # Encode image
        img_feats = self.encoder(images)            # (B, 2048)

        # Project to (B, d_model)
        img_emb = self.img_proj(img_feats)          # (B, 512)

        # Make memory shape (B, 1, d_model) for decoder cross-attention
        memory = img_emb.unsqueeze(1)               # (B, 1, 512)

        # Decode
        logits = self.decoder(captions_inp, memory) # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, images, max_len=30):
        """
        Greedy caption generation.
        images: (B, 3, 224, 224)

        Returns:
        captions_out: (B, max_len) generated token IDs
        """
        self.eval()

        B = images.size(0)
        device = images.device

        # Encode image
        img_feats = self.encoder(images)  # (B, 2048)
        img_emb = self.img_proj(img_feats)  # (B, 512)
        memory = img_emb.unsqueeze(1)  # (B, 1, 512)

        # Start with <sos>
        captions = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = self.decoder(captions, memory)  # (B, t, vocab)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)

            captions = torch.cat([captions, next_token], dim=1)

            # Stop only for those batches that hit <eos>
            # But continue generation for others
            if torch.all(next_token.squeeze(1) == self.eos_idx):
                break

        # Always return captions (never None)
        return captions
