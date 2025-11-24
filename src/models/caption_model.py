import torch
import torch.nn as nn

from src.models.encoder_resnet50 import ResNet50Encoder
from src.models.decoder_transformer import TransformerDecoderModule
from src.data.vocab import PAD_IDX, SOS_IDX, EOS_IDX


class CaptionModel(nn.Module):
    """
    Full baseline captioning model:
    - ResNet50 image encoder
    - Linear projection (2048 -> d_model)
    - Transformer decoder
    """

    def __init__(self, vocab, d_model=512, train_cnn=False):
        super().__init__()

        # Vocab info from FashionVocab
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.pad_idx = PAD_IDX
        self.bos_idx = SOS_IDX
        self.eos_idx = EOS_IDX

        # 1) Encoder
        self.encoder = ResNet50Encoder(train_cnn=train_cnn)

        # 2) Projection layer
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
        img_feats = self.encoder(images)
        img_emb = self.img_proj(img_feats)
        memory = img_emb.unsqueeze(1)
        logits = self.decoder(captions_inp, memory)
        return logits

    @torch.no_grad()
    def generate(self, images, max_len=30):
        self.eval()
        B = images.size(0)
        device = images.device

        img_feats = self.encoder(images)
        img_emb = self.img_proj(img_feats)
        memory = img_emb.unsqueeze(1)

        captions = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = self.decoder(captions, memory)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            captions = torch.cat([captions, next_token], dim=1)

            if torch.all(next_token.squeeze(1) == self.eos_idx):
                break

        return captions
