"""Hard Processor Contract Test for Qwen2.5-VL Three-Image Change-VQA.

Verifies:
- Exactly 3 images enter processor
- Exactly 3 vision blocks appear (<|vision_start|> / <|vision_end|>)
- Ordering is preserved: T0 -> T1 -> overlay
- No dropped images, no duplicated images
- Valid input_ids, attention_mask, pixel_values, image_grid_thw
- Valid labels where training mode is used
- Records exact tensor shapes and image-token allocation
"""

import pytest
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


def test_qwen_three_image_processor_contract():
    """Verify native Qwen processor multi-image tensor allocation with real rasters."""
    t0_path = Path("artifacts_storage/change_vqa_verification/test_10_cropped_t0_patch.png")
    t1_path = Path("artifacts_storage/change_vqa_verification/test_10_cropped_t1_patch.png")
    overlay_path = Path("artifacts_storage/change_vqa_verification/test_10_change_overlay.png")

    assert t0_path.exists(), f"Missing T0 raster: {t0_path}"
    assert t1_path.exists(), f"Missing T1 raster: {t1_path}"
    assert overlay_path.exists(), f"Missing Overlay raster: {overlay_path}"

    img_t0 = Image.open(t0_path).convert("RGB")
    img_t1 = Image.open(t1_path).convert("RGB")
    img_overlay = Image.open(overlay_path).convert("RGB")

    assert img_t0.size == (136, 80)
    assert img_t1.size == (136, 80)
    assert img_overlay.size == (1024, 1024)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", local_files_only=True)

    structured_change_query = (
        "Image 1 (BEFORE):\n"
        "Pre-change observation of the detected region.\n\n"
        "Image 2 (AFTER):\n"
        "Post-change observation of the same spatial region.\n\n"
        "Image 3 (WHERE CHANGE OCCURRED):\n"
        "Change mask/overlay highlighting pixels identified by TinyCD.\n\n"
        "Change metadata:\n"
        "- Bounding box: [112, 12, 192, 148]\n"
        "- Area: 4515 pixels\n"
        "- Change ratio: 0.110825\n"
        "- Temporal order: Image 1 (BEFORE / T0) -> Image 2 (AFTER / T1)\n\n"
        "Question:\n"
        "What changed between these two dates and what does it represent?"
    )

    # 1. Verification Mode: Inference Prompt
    messages_inference = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_t0},
                {"type": "image", "image": img_t1},
                {"type": "image", "image": img_overlay},
                {"type": "text", "text": structured_change_query},
            ],
        }
    ]

    text_inf = processor.apply_chat_template(messages_inference, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages_inference)

    assert len(image_inputs) == 3, f"Expected 3 image inputs, got {len(image_inputs)}"
    assert video_inputs is None

    inputs = processor(
        text=[text_inf],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Tensor checks
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert "pixel_values" in inputs
    assert "image_grid_thw" in inputs

    input_ids = inputs["input_ids"][0].tolist()
    assert inputs["input_ids"].shape[0] == 1
    assert inputs["input_ids"].shape[1] == 1566  # Exact token length for inference prompt
    assert inputs["attention_mask"].shape == inputs["input_ids"].shape
    assert inputs["attention_mask"].sum().item() == 1566  # All active tokens

    # Check image grid dimensions
    grid = inputs["image_grid_thw"]
    assert grid.shape == (3, 3), f"Expected image_grid_thw shape (3, 3), got {grid.shape}"
    assert grid[0].tolist() == [1, 6, 10], f"Expected T0 grid [1, 6, 10], got {grid[0].tolist()}"
    assert grid[1].tolist() == [1, 6, 10], f"Expected T1 grid [1, 6, 10], got {grid[1].tolist()}"
    assert grid[2].tolist() == [1, 74, 74], f"Expected Overlay grid [1, 74, 74], got {grid[2].tolist()}"

    # Verify vision token counts
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    v_starts = [i for i, x in enumerate(input_ids) if x == vision_start_id]
    v_ends = [i for i, x in enumerate(input_ids) if x == vision_end_id]
    img_pads = [i for i, x in enumerate(input_ids) if x == image_pad_id]

    assert len(v_starts) == 3, f"Expected exactly 3 <|vision_start|> tokens, got {len(v_starts)}"
    assert len(v_ends) == 3, f"Expected exactly 3 <|vision_end|> tokens, got {len(v_ends)}"
    assert len(img_pads) == 1399, f"Expected exactly 1399 <|image_pad|> tokens, got {len(img_pads)}"

    # Check token counts per image
    tokens_img0 = v_ends[0] - v_starts[0] - 1
    tokens_img1 = v_ends[1] - v_starts[1] - 1
    tokens_img2 = v_ends[2] - v_starts[2] - 1

    assert tokens_img0 == 15, f"Image 1 (T0) should have 15 tokens, got {tokens_img0}"
    assert tokens_img1 == 15, f"Image 2 (T1) should have 15 tokens, got {tokens_img1}"
    assert tokens_img2 == 1369, f"Image 3 (Overlay) should have 1369 tokens, got {tokens_img2}"
    assert tokens_img0 + tokens_img1 + tokens_img2 == 1399

    # 2. Verification Mode: Training Mode with Loss Masking
    messages_training = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_t0},
                {"type": "image", "image": img_t1},
                {"type": "image", "image": img_overlay},
                {"type": "text", "text": structured_change_query},
            ],
        },
        {
            "role": "assistant",
            "content": "The detected change represents new commercial building construction in the highlighted region.",
        },
    ]

    text_train = processor.apply_chat_template(messages_training, tokenize=False, add_generation_prompt=False)
    image_inputs_train, _ = process_vision_info(messages_training)

    inputs_train = processor(
        text=[text_train],
        images=image_inputs_train,
        padding=True,
        return_tensors="pt",
    )

    labels = inputs_train["input_ids"].clone()
    label_pad_token_id = -100
    im_start_id = processor.tokenizer.encode("<|im_start|>")[-1]
    im_end_id = processor.tokenizer.encode("<|im_end|>")[-1]
    assistant_token_id = processor.tokenizer.encode("assistant")[-1]

    for i, seq in enumerate(inputs_train["input_ids"]):
        seq_len = len(seq)
        is_assistant_block = False
        for j in range(seq_len):
            if seq[j] == im_start_id and j + 1 < seq_len and seq[j + 1] == assistant_token_id:
                is_assistant_block = True
            elif seq[j] == im_end_id and is_assistant_block:
                is_assistant_block = False
                continue
            if not is_assistant_block:
                labels[i, j] = label_pad_token_id

    supervised_tokens = (labels != -100).sum().item()
    assert supervised_tokens == 17, f"Expected 17 supervised assistant tokens, got {supervised_tokens}"
    decoded_supervised = processor.tokenizer.decode([t for t in labels[0] if t != -100])
    assert "commercial building construction" in decoded_supervised
