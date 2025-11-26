import torch
import torch.nn as nn
import torchvision.models as models


class Model1(nn.Module):
    """
    Stronger caption model:
    - ResNet101 encoder (frozen)
    - Metadata embeddings for 8 metadata fields
    - Projection → Transformer decoder
    """

    def __init__(
        self,
        vocab,
        meta_sizes,          # dict: {"gender": n1, "masterCategory": n2, ...}
        d_model=512,
        meta_emb_dim=16,
    ):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.pad_idx = vocab.word2idx["<pad>"]
        self.bos_idx = vocab.word2idx["<sos>"]
        self.eos_idx = vocab.word2idx["<eos>"]

        # -------------------------------
        # 1. Stronger Vision Encoder
        # -------------------------------
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.cnn = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_dim = 1280

        # -------------------------------
        # 2. Metadata Embeddings
        # -------------------------------
        self.meta_embeddings = nn.ModuleDict()
        total_meta_dim = 0

        for key, size in meta_sizes.items():
            emb = nn.Embedding(size, meta_emb_dim)
            self.meta_embeddings[key] = emb
            total_meta_dim += meta_emb_dim

        # -------------------------------
        # 3. Fuse image + metadata
        # -------------------------------
        self.fuse = nn.Linear(self.cnn_dim + total_meta_dim, d_model)

        # -------------------------------
        # 4. Decoder
        # -------------------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.token_emb = nn.Embedding(self.vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, self.vocab_size)

    def forward(self, images, captions, meta_dict):
        # vision
        with torch.no_grad():
            v = self.cnn(images)
            v = self.avgpool(v)
            v = torch.flatten(v, 1)

        # metadata embeddings
        meta_vecs = []
        for key, emb in self.meta_embeddings.items():
            meta_vecs.append(emb(meta_dict[key]))
        meta_vecs = torch.cat(meta_vecs, dim=1)

        fused = torch.cat([v, meta_vecs], dim=1)
        memory = self.fuse(fused).unsqueeze(1)

        tgt = self.token_emb(captions)
        out = self.decoder(tgt, memory)
        return self.fc_out(out)

    @torch.no_grad()
    def generate(self, images, meta_dict, max_len=30):
        self.eval()
        device = images.device
        B = images.size(0)

        # -------------------------------------------------
        # 1. CNN → pooled vector (same as forward)
        # -------------------------------------------------
        v = self.cnn(images)  # [B, 1280, H, W]
        v = self.avgpool(v)  # [B, 1280, 1, 1]
        v = torch.flatten(v, 1)  # [B, 1280]

        # -------------------------------------------------
        # 2. Metadata vectors (same as forward)
        # -------------------------------------------------
        meta_vecs = []
        for key, emb in self.meta_embeddings.items():
            meta_vecs.append(emb(meta_dict[key]))  # each: [B, 16]
        meta_vecs = torch.cat(meta_vecs, dim=1)  # [B, total_meta_dim]

        # -------------------------------------------------
        # 3. Fuse and create decoder memory
        # -------------------------------------------------
        fused = torch.cat([v, meta_vecs], dim=1)  # [B, 1280 + meta]
        memory = self.fuse(fused).unsqueeze(1)  # [B, 1, d_model]

        # -------------------------------------------------
        # 4. Start sequence with <sos>
        # -------------------------------------------------
        sequences = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=device)

        # -------------------------------------------------
        # 5. Greedy decoding
        # -------------------------------------------------
        for _ in range(max_len - 1):
            tgt = self.token_emb(sequences)  # [B, T, d_model]
            out = self.decoder(tgt, memory)  # [B, T, d_model]
            logits = self.fc_out(out)  # [B, T, vocab]

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            sequences = torch.cat([sequences, next_token], dim=1)

            if torch.all(next_token.squeeze(1) == self.eos_idx):
                break

        return sequences
