# src/models/decoder_transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.cos(position * div_term)
        pe[:, 1::2] = torch.sin(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        T = x.size(1)
        x = x + self.pe[:, :T]
        return x


class TransformerDecoderModule(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _generate_square_subsequent_mask(self, sz: int, device):
        # standard autoregressive mask (T, T)
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(self, tgt_tokens, memory):
        """
        tgt_tokens: (B, T) token ids
        memory: (B, S, d_model) image features

        returns:
          logits: (B, T, vocab_size)
        """
        device = tgt_tokens.device
        B, T = tgt_tokens.size()

        tgt_emb = self.embedding(tgt_tokens) * math.sqrt(self.d_model)  # (B, T, d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        tgt_mask = self._generate_square_subsequent_mask(T, device=device)  # (T, T)

        tgt_key_padding_mask = (tgt_tokens == self.pad_idx)  # (B, T)

        decoded = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B, T, d_model)

        logits = self.output_proj(decoded)
        return logits
