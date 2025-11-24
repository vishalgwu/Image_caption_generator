import torch
import torch.nn as nn

from src.models.encoder_resnet50 import ResNet50Encoder
from src.models.decoder_transformer import TransformerDecoderModule


class CaptionModel(nn.Module):
    """
    Full baseline captioning model:
    - ResNet50 encoder
    - Linear projection
    - Transformer decoder
    """

    def __init__(self, vocab, d_model=512, train_cnn=False):
        super().__init__()

        # USE PASSED VOCAB OBJECT
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.pad_idx = vocab.word2idx["<pad>"]
        self.bos_idx = vocab.word2idx["<sos>"]
        self.eos_idx = vocab.word2idx["<eos>"]

        # 1) Encoder
        self.encoder = ResNet50Encoder(train_cnn=train_cnn)

        # 2) Project 2048 → d_model
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

        # Start with <sos>
        captions = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = self.decoder(captions, memory)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            captions = torch.cat([captions, next_token], dim=1)

            # break if all batches reached EOS
            if torch.all(next_token.squeeze(1) == self.eos_idx):
                break

        return captions
