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
from specialists.single_image.adaptation.qwen25vl.sensor_converters import Sentinel2MultispectralConverter

logger = get_logger("qwen25vl_collator")


class DatasetIntegrityError(RuntimeError):
    """Raised when dataset integrity checks fail in strict real-data training/evaluation."""
    pass


class Qwen25VLDataCollator:
    """Multimodal data collator supporting optical and SAR rasters with label masking and strict real-data enforcement."""

    def __init__(
        self,
        processor: Any,
        label_pad_token_id: int = -100,
        strict_real_data: bool = True,
        demo_mode: bool = False,
    ) -> None:
        self.processor = processor
        self.label_pad_token_id = label_pad_token_id
        self.strict_real_data = strict_real_data
        self.demo_mode = demo_mode

    def _resolve_image(self, example: Dict[str, Any]) -> Image.Image:
        """Resolve example image field to a 3-channel RGB PIL Image with strict real-data enforcement."""
        img_val = example.get("image")
        rec_id = example.get("id", "unknown")
        pair_id = example.get("pair_id")

        # Determine authoritative modality with robust multi-field inference
        raw_mod = example.get("modality")
        sensor_str = str(example.get("sensor", "")).lower()
        if raw_mod:
            modality = str(raw_mod).strip().lower()
        elif "sentinel-1" in sensor_str or "s1" in sensor_str or "sar" in sensor_str:
            modality = "sar"
        elif "sentinel-2" in sensor_str or "s2" in sensor_str or "optical" in sensor_str or "msi" in sensor_str:
            modality = "optical"
        elif isinstance(img_val, (str, Path)):
            fname = Path(img_val).name.lower()
            if any(k in fname for k in ["s1", "sar"]):
                modality = "sar"
            elif any(k in fname for k in ["s2", "optical", "msi"]):
                modality = "optical"
            else:
                modality = "optical"
        else:
            modality = "optical"

        # If sensor metadata explicitly designates SAR but modality was defaulted or mismatched, trust sensor
        if ("sentinel-1" in sensor_str or "sar" in sensor_str) and modality != "sar":
            modality = "sar"
        elif ("sentinel-2" in sensor_str or "msi" in sensor_str) and modality != "optical":
            modality = "optical"

        if isinstance(img_val, Image.Image):
            pil_img = img_val
        elif isinstance(img_val, (str, Path)):
            p = Path(img_val)

            # 1. STRICT REAL DATA: Check for demo or fallback assets
            p_str_lower = str(p).lower()
            if "demo" in p_str_lower or "fallback" in p_str_lower:
                if self.strict_real_data or not self.demo_mode:
                    raise DatasetIntegrityError(
                        f"Dataset integrity error: demo or fallback image path detected at '{p}' for record '{rec_id}'. "
                        "Strict real BigEarthNet data enforcement is active; demo substitutions are strictly forbidden in training/eval."
                    )

            # 2. STRICT REAL DATA: Check file existence
            if not p.exists():
                if self.strict_real_data or not self.demo_mode:
                    raise DatasetIntegrityError(
                        f"Dataset integrity error: source image not found at path '{p}' for record '{rec_id}' (pair: '{pair_id}'). "
                        "Strict real BigEarthNet data enforcement is active; missing imagery is strictly forbidden."
                    )
                # Isolated DEMO_MODE fallback (only when strict_real_data=False AND demo_mode=True)
                logger.warning(
                    f"DEMO MODE ONLY: Image not found at path '{p}'. Using modality-appropriate demo fallback."
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
                return pil_img

            # 3. STRICT REAL DATA: Pair ID validation
            if self.strict_real_data and pair_id is not None:
                if pair_id not in str(p) and p.parent.name != pair_id:
                    raise DatasetIntegrityError(
                        f"Dataset integrity error: pair ID mismatch. Record specifies pair '{pair_id}', "
                        f"but image path is '{p}'."
                    )

            # 4. STRICT REAL DATA: Modality validation
            if self.strict_real_data:
                fname_lower = p.name.lower()
                if modality == "sar":
                    if not any(k in fname_lower for k in ["s1", "sar", "sentinel1"]):
                        raise DatasetIntegrityError(
                            f"Dataset integrity error: modality mismatch. Record specifies modality 'sar', "
                            f"but image filename '{p.name}' does not indicate Sentinel-1 SAR imagery."
                        )
                elif modality == "optical":
                    if not any(k in fname_lower for k in ["s2", "optical", "sentinel2", "msi"]):
                        raise DatasetIntegrityError(
                            f"Dataset integrity error: modality mismatch. Record specifies modality 'optical', "
                            f"but image filename '{p.name}' does not indicate Sentinel-2 Optical imagery."
                        )

            # Load and convert image
            try:
                if modality == "sar":
                    pil_img, _ = SARPreprocessor.process_file(p)
                elif modality == "optical" and p.suffix.lower() in {".tif", ".tiff"}:
                    pil_img, _ = Sentinel2MultispectralConverter.convert_s2_to_rgb(file_path=p, record_id=rec_id, patch_id=str(pair_id))
                elif p.suffix.lower() in {".tif", ".tiff"}:
                    # Infer modality from filename
                    if any(k in p.name.lower() for k in ["s1", "sar"]):
                        pil_img, _ = SARPreprocessor.process_file(p)
                    else:
                        pil_img, _ = Sentinel2MultispectralConverter.convert_s2_to_rgb(file_path=p, record_id=rec_id, patch_id=str(pair_id))
                else:
                    with Image.open(p) as img:
                        pil_img = img.convert("RGB")
            except Exception as e:
                raise DatasetIntegrityError(
                    f"Dataset integrity error: failed to decode/convert image at '{p}' for record '{rec_id}': {e}"
                ) from e
        else:
            raise DatasetIntegrityError(f"Unsupported image type in example '{rec_id}': {type(img_val)}")

        # 5. STRICT REAL DATA: Validate decoded PIL image
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        w, h = pil_img.size
        if w <= 0 or h <= 0:
            raise DatasetIntegrityError(
                f"Dataset integrity error: invalid decoded image dimensions ({w}x{h}) at path '{img_val}'"
            )

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
