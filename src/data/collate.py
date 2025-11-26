# src/data/collate.py
import torch


def collate_fn(batch):
    """
    batch: list of (image, caption_tensor)
      - image: (3, H, W)
      - caption_tensor: (T,)

    Returns:
      images: (B, 3, H, W)
      captions: (B, T)
    """
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    captions = torch.stack(captions, dim=0)
    return images, captions
