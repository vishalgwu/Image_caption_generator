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
