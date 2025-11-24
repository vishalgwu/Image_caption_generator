import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Adds information about word order for the Transformer.
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        T = x.size(1)
        return x + self.pe[:, :T]


class TransformerDecoderModule(nn.Module):
    """
    Full Transformer Decoder:
    - Token embedding
    - Positional encoding
    - TransformerDecoder layers
    - Output linear layer projecting to vocab size
    """
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        pad_idx=0
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, size, device):
        """
        Autoregressive mask: block future tokens.
        returns (size, size)
        """
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, tgt_tokens, memory):
        """
        tgt_tokens: (B, T)
        memory: (B, 1, d_model)

        returns logits: (B, T, vocab_size)
        """
        # Embed tokens and add positions
        x = self.tok_embed(tgt_tokens) * math.sqrt(self.d_model)  # (B, T, d_model)
        x = self.pos_encoding(x)

        T = tgt_tokens.size(1)
        device = tgt_tokens.device

        tgt_mask = self.generate_square_subsequent_mask(T, device=device)  # (T, T)
        tgt_padding_mask = (tgt_tokens == self.pad_idx)  # (B, T)

        decoded = self.transformer_decoder(
            x,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )  # (B, T, d_model)

        logits = self.output_layer(decoded)  # (B, T, vocab_size)
        return logits
