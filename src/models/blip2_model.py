"""BLIP-2 based captioning helper.

This wrapper keeps the integration minimal so training/inference scripts can
optionally leverage a pretrained multimodal model alongside the in-house
architectures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

# Transformers is an optional dependency for this project. Import inside a try/except
# so the rest of the codebase keeps working even if it is missing.
try:  # pragma: no cover - exercised only when transformers is installed
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "`transformers` is required for BLIP-2 support. Install it via `pip install transformers`."
    ) from exc


@dataclass
class BLIP2Config:
    """Configuration for BLIP-2 captioning.

    Attributes:
        model_name: Hugging Face checkpoint identifier.
        device: torch device string ("cuda"/"cpu"/"mps").
        max_new_tokens: maximum length for generated captions.
    """

    model_name: str = "Salesforce/blip2-opt-2.7b"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 50


class BLIP2Captioner:
    """Thin wrapper around Hugging Face BLIP-2 for quick inference.

    Usage
    -----
    >>> captioner = BLIP2Captioner()
    >>> text = captioner.generate(pil_image)
    """

    def __init__(self, config: Optional[BLIP2Config] = None) -> None:
        self.config = config or BLIP2Config()
        self.processor = Blip2Processor.from_pretrained(self.config.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if "cuda" in self.config.device else torch.float32,
        ).to(self.config.device)

    @torch.no_grad()
    def generate(self, image, prompt: str = "Describe this product in one sentence.") -> str:
        """Generate a caption for the provided PIL image."""

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.config.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens
        )

        caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return caption.strip()
