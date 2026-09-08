"""Multimodal Data Collator for Qwen2.5-VL SFT Training.

Processes multimodal conversation dictionaries, loads/synthesizes images (optical & SAR),
formats inputs via Qwen's chat template, aligns pixel values with visual tokens,
and constructs loss masks so supervision is strictly applied to assistant tokens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import torch

from core.logging import get_logger
from specialists.single_image.adaptation.qwen25vl.sar import SARPreprocessor

logger = get_logger("qwen25vl_collator")


class Qwen25VLDataCollator:
    """Multimodal data collator supporting optical and SAR rasters with label masking."""

    def __init__(
        self,
        processor: Any,
        label_pad_token_id: int = -100,
    ) -> None:
        self.processor = processor
        self.label_pad_token_id = label_pad_token_id

    def _resolve_image(self, example: Dict[str, Any]) -> Image.Image:
        """Resolve example image field to a 3-channel RGB PIL Image."""
        img_val = example.get("image")
        modality = example.get("modality", "optical").lower()

        if isinstance(img_val, Image.Image):
            pil_img = img_val
        elif isinstance(img_val, (str, Path)):
            p = Path(img_val)
            if p.exists():
                if modality == "sar" or p.suffix.lower() in {".tif", ".tiff"}:
                    pil_img, _ = SARPreprocessor.process_file(p)
                else:
                    with Image.open(p) as img:
                        pil_img = img.convert("RGB")
            else:
                # Defensive fallback for unmaterialized remote sensing shards
                logger.warning(
                    f"Image not found at path '{p}'. Using modality-appropriate raster fallback."
                )
                demo_opt = Path("demo_assets/demo_optical_single.png")
                demo_sar = Path("demo_assets/demo_sar_cross.tif")
                if modality == "sar" and demo_sar.exists():
                    pil_img, _ = SARPreprocessor.process_file(demo_sar)
                elif demo_opt.exists():
                    with Image.open(demo_opt) as img:
                        pil_img = img.convert("RGB")
                else:
                    import numpy as np
                    synth = np.full((120, 120, 3), 128, dtype=np.uint8)
                    pil_img = Image.fromarray(synth, mode="RGB")
        else:
            raise ValueError(f"Unsupported image type in example: {type(img_val)}")

        return pil_img

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of multimodal instruction examples into model inputs and labels."""
        from qwen_vl_utils import process_vision_info

        formatted_messages_list = []
        for example in batch:
            pil_img = self._resolve_image(example)
            raw_messages = example["messages"]

            # Replace the image placeholder dict with actual PIL Image object for process_vision_info
            conv = []
            for msg in raw_messages:
                role = msg["role"]
                content = msg["content"]
                if isinstance(content, list):
                    new_content = []
                    for item in content:
                        if item.get("type") == "image":
                            new_content.append({"type": "image", "image": pil_img})
                        else:
                            new_content.append(item)
                    conv.append({"role": role, "content": new_content})
                else:
                    conv.append({"role": role, "content": content})
            formatted_messages_list.append(conv)

        # 1. Generate text prompts via chat template
        texts = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in formatted_messages_list
        ]

        # 2. Extract visual inputs (images/videos)
        image_inputs, video_inputs = process_vision_info(formatted_messages_list)

        # 3. Process inputs through AutoProcessor
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.label_pad_token_id

        # 4. Mask user and system tokens so loss is computed exclusively on assistant tokens
        # In Qwen chat template:
        # <|im_start|>assistant\n...<|im_end|>
        im_start_id = self.processor.tokenizer.encode("<|im_start|>")[-1]
        im_end_id = self.processor.tokenizer.encode("<|im_end|>")[-1]
        assistant_token_id = self.processor.tokenizer.encode("assistant")[-1]

        for i, seq in enumerate(inputs["input_ids"]):
            seq_len = len(seq)
            is_assistant_block = False
            for j in range(seq_len):
                if seq[j] == im_start_id and j + 1 < seq_len and seq[j + 1] == assistant_token_id:
                    is_assistant_block = True
                elif seq[j] == im_end_id and is_assistant_block:
                    # Include the im_end_id in loss then end assistant block
                    is_assistant_block = False
                    continue

                if not is_assistant_block:
                    labels[i, j] = self.label_pad_token_id

        inputs["labels"] = labels
        return inputs
