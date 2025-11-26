import torch
import torch.nn as nn

from src.models.encoder_resnet50 import ResNet50Encoder
from src.models.decoder_transformer import TransformerDecoderModule


class CaptionModel(nn.Module):
    """
    Baseline image captioning model:
    - ResNet50 encoder (frozen)
    - Linear projection to d_model
    - Transformer decoder
    """

    def __init__(self, vocab, d_model=512, train_cnn=False):
        super().__init__()

        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.pad_idx = vocab.word2idx["<pad>"]
        self.bos_idx = vocab.word2idx["<sos>"]
        self.eos_idx = vocab.word2idx["<eos>"]

        # 1) Encoder: ResNet50
        self.encoder = ResNet50Encoder(train_cnn=train_cnn)

        # 2) Project 2048 -> d_model
        self.img_proj = nn.Linear(2048, d_model)

        # 3) Transformer decoder
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
        images: [B, 3, 224, 224]
        captions_inp: [B, T] (input tokens, starting with <sos>)
        """
        img_feats = self.encoder(images)              # [B, 2048]
        img_emb = self.img_proj(img_feats)           # [B, d_model]
        memory = img_emb.unsqueeze(1)                # [B, 1, d_model]

        logits = self.decoder(captions_inp, memory)  # [B, T, vocab]
        return logits

    @torch.no_grad()
    def generate(self, images, max_len=30):
        """
        Greedy decoding for caption generation.
        images: [B, 3, 224, 224]
        returns: [B, T] token ids
        """
        self.eval()
        device = images.device
        B = images.size(0)

        # Encode image
        img_feats = self.encoder(images)
        img_emb = self.img_proj(img_feats)
        memory = img_emb.unsqueeze(1)  # [B, 1, d_model]

        # Start with <sos>
        captions = torch.full(
            (B, 1),
            self.bos_idx,
            dtype=torch.long,
            device=device
        )

        for _ in range(max_len - 1):
            logits = self.decoder(captions, memory)       # [B, T, vocab]
            next_token = logits[:, -1, :].argmax(dim=-1)  # [B]
            next_token = next_token.unsqueeze(1)          # [B, 1]

            captions = torch.cat([captions, next_token], dim=1)

            # Early stop if all reached <eos>
            if torch.all(next_token.squeeze(1) == self.eos_idx):
                break

        return captions
