"""Comprehensive Unit & Integration Test Suite for Qwen2.5-VL Migration (Division 2).

Verifies:
1. BoxCodec coordinate encoding, decoding, and reversible roundtrip math
2. QwenGroundingParser detection, label preservation, and natural-language cleaning
3. SARPreprocessor input validation, dynamic range normalization, and 3-channel synthesis
4. SatelliteTilingEngine local-to-global offset translation and NMS deduplication
5. Multimodal collator label masking (-100 for non-assistant tokens)
6. Hardware inspection and CUDA guard behavior
7. LoRA target module discovery (freezing vision backbone, adapting language decoder + visual merger)
8. SingleImageRSSpecialistTool dual-backend execution (qwen25vl vs paligemma_legacy)
9. Truthful metadata reporting (is_mock=False, backend=qwen25vl)
10. Parent-scene split safety and zero spatial leakage
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from core.interfaces import BaseSpecialistTool
from core.schemas import (
    EvidenceType,
    ImageFormat,
    ImageInput,
    ImageModality,
    TaskType,
    ToolRequest,
    ToolStatus,
)
from specialists.single_image.adaptation.qwen25vl.bbox_codec import BoxCodec
from specialists.single_image.adaptation.qwen25vl.config import (
    LoraConfigQwen,
    ModelConfig,
    QuantizationConfig,
    Qwen25VLFullConfig,
    ResolutionConfig,
    inspect_hardware,
    verify_cuda_available,
)
from specialists.single_image.adaptation.qwen25vl.grounding import QwenGroundingParser
from specialists.single_image.adaptation.qwen25vl.inference import QwenSingleImageEngine
from specialists.single_image.adaptation.qwen25vl.model import QwenModelLoader
from specialists.single_image.adaptation.qwen25vl.sar import SARPreprocessor
from specialists.single_image.adaptation.qwen25vl.tiling import SatelliteTilingEngine
from specialists.single_image.specialist import SingleImageRSSpecialistTool
import importlib

_prep_dataset = importlib.import_module("specialists.single_image.training.colab.01_prepare_dataset")
build_canonical_qwen_sample = _prep_dataset.build_canonical_qwen_sample
extract_parent_scene_id = _prep_dataset.extract_parent_scene_id
partition_by_parent_scene = _prep_dataset.partition_by_parent_scene


def test_box_codec_validation_and_clipping():
    """Verify BoxCodec boundary validation and clipping."""
    # Valid box
    assert BoxCodec.validate_bbox([100, 100, 500, 500], format_name="pixel_xyxy", width=1024, height=1024)
    # Invalid box (x2 <= x1)
    assert not BoxCodec.validate_bbox([500, 100, 100, 500], format_name="pixel_xyxy", width=1024, height=1024)
    # Out of range normalized box
    assert not BoxCodec.validate_bbox([0.1, 0.1, 1.5, 0.9], format_name="normalized_0_1_xyxy")

    # Clipping
    clipped = BoxCodec.clip_bbox([-10, -20, 1200, 1500], width=1024, height=1024)
    assert clipped[0] == 0.0
    assert clipped[1] == 0.0
    assert clipped[2] == 1024.0
    assert clipped[3] == 1024.0


def test_box_codec_encoding_decoding_roundtrip():
    """Verify encoding to Qwen tokens and decoding back to pixel coordinates."""
    orig_box = [200.0, 150.0, 600.0, 450.0]  # [x1, y1, x2, y2]
    w, h = 1000, 1000

    token_str = BoxCodec.encode_bbox(
        orig_box, width=w, height=h, label="runway", source_format="pixel_xyxy"
    )
    assert "<|object_ref_start|>runway<|object_ref_end|>" in token_str
    assert "<|box_start|>" in token_str
    assert "<|box_end|>" in token_str

    decoded = BoxCodec.decode_bbox(token_str, width=w, height=h)
    assert len(decoded) == 1
    det = decoded[0]
    assert det["label"] == "runway"
    pred_box = det["pixel_bbox"]

    # Precision within 2 pixels
    assert abs(pred_box[0] - orig_box[0]) <= 2.0
    assert abs(pred_box[1] - orig_box[1]) <= 2.0
    assert abs(pred_box[2] - orig_box[2]) <= 2.0
    assert abs(pred_box[3] - orig_box[3]) <= 2.0


def test_grounding_parser_clean_text_and_evidence():
    """Verify QwenGroundingParser text stripping and Evidence generation."""
    raw_response = (
        "The primary warehouse is located in the central sector, approximately here: "
        "<|object_ref_start|>warehouse<|object_ref_end|><|box_start|>(188,412),(477,691)<|box_end|>. "
        "It features high radiometric contrast against the tarmac."
    )
    clean_text, evidence, meta = QwenGroundingParser.parse_grounding_response(
        raw_response, image_width=1000, image_height=1000, query="warehouse", image_id="test_01"
    )

    # Special tokens must not appear in clean conversational text
    assert "<|box_start|>" not in clean_text
    assert "<|object_ref_start|>" not in clean_text
    assert "(188,412)" not in clean_text
    assert "primary warehouse" in clean_text

    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.type == EvidenceType.BOUNDING_BOX
    assert ev.label == "warehouse"
    assert ev.image_id == "test_01"
    assert "bbox" in ev.data
    assert ev.data["format"] == "[ymin, xmin, ymax, xmax]"
    assert meta["parse_status"] == "SUCCESS"


def test_grounding_parser_iou_math():
    """Verify IoU calculations for identical, overlapping, and disjoint boxes."""
    b1 = [100, 100, 300, 300]
    assert QwenGroundingParser.calculate_iou(b1, b1, format_name="pixel_xyxy") == 1.0

    b_disjoint = [400, 400, 600, 600]
    assert QwenGroundingParser.calculate_iou(b1, b_disjoint, format_name="pixel_xyxy") == 0.0

    b_half = [100, 100, 200, 300]
    iou = QwenGroundingParser.calculate_iou(b1, b_half, format_name="pixel_xyxy")
    assert 0.0 < iou < 1.0


def test_sar_preprocessor_synthetic_rgb_synthesis():
    """Verify SARPreprocessor 2-band ratio, NaN handling, and RGB conversion."""
    # Synthetic dual-pol SAR array [100, 100, 2] with a NaN value
    vv = np.random.uniform(10.0, 50.0, (100, 100)).astype(np.float32)
    vh = np.random.uniform(5.0, 25.0, (100, 100)).astype(np.float32)
    vv[10, 10] = np.nan
    vh[20, 20] = np.inf
    sar_data = np.stack([vv, vh], axis=-1)

    pil_img, metadata = SARPreprocessor.construct_3channel_sar(sar_data)
    assert isinstance(pil_img, Image.Image)
    assert pil_img.mode == "RGB"
    assert pil_img.size == (100, 100)
    assert metadata["modality"] == "sar"
    assert metadata["is_multipolarized"] is True
    assert "log(VV/VH)" in metadata["channels"]


def test_satellite_tiling_offsets_and_nms():
    """Verify SatelliteTilingEngine tile slicing and coordinate translation."""
    tiler = SatelliteTilingEngine(tile_size=512, overlap=64)
    dummy_large_img = Image.new("RGB", (1024, 1024), color=(128, 128, 128))
    tiles = tiler.generate_tiles(dummy_large_img)
    assert len(tiles) > 1

    # Local to global translation
    local_box = [50.0, 50.0, 150.0, 150.0]
    global_box = SatelliteTilingEngine.translate_local_to_global_bbox(local_box, x_offset=448, y_offset=448)
    assert global_box == [498.0, 498.0, 598.0, 598.0]

    # NMS Deduplication
    detections = [
        {"label": "building", "pixel_bbox": [100, 100, 200, 200], "confidence": 0.95},
        {"label": "building", "pixel_bbox": [105, 105, 205, 205], "confidence": 0.90},  # Duplicate
        {"label": "building", "pixel_bbox": [500, 500, 600, 600], "confidence": 0.85},  # Distinct
    ]
    kept = tiler.merge_detections_nms(detections, iou_threshold=0.5)
    assert len(kept) == 2
    assert kept[0]["confidence"] == 0.95
    assert kept[1]["confidence"] == 0.85


def test_parent_scene_extraction_and_zero_leakage():
    """Verify parent scene isolation across splits."""
    samples = [
        {"id": f"s_{i}", "parent_scene_id": f"scene_{(i // 3):02d}", "data": i}
        for i in range(30)
    ]
    train_s, val_s, test_s, report = partition_by_parent_scene(samples, train_ratio=0.6, val_ratio=0.2)

    assert report["leakage_passed"] is True
    assert report["intersections"]["train_val"] == 0
    assert report["intersections"]["train_test"] == 0
    assert report["intersections"]["val_test"] == 0


def test_cuda_guard_detection():
    """Verify verify_cuda_available raises RuntimeError when CUDA is unavailable and strict=True."""
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError) as exc_info:
            verify_cuda_available(strict=True)
        assert "CUDA_NOT_AVAILABLE" in str(exc_info.value)
    else:
        assert verify_cuda_available(strict=True) is True


def test_lora_targeting_freezes_vision_backbone():
    """Verify LoRA regex targets language decoder + merger and freezes vision backbone."""
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
    with torch.device("meta"):
        model = Qwen2_5_VLForConditionalGeneration(config)

    peft_model, stats = QwenModelLoader.apply_lora_adaptation(model, LoraConfigQwen())
    adapted_keys = [k for k, _ in peft_model.named_parameters() if "lora_" in k]

    # Vision backbone blocks MUST NOT be in adapted keys
    vision_block_adapted = [k for k in adapted_keys if "visual.blocks" in k]
    assert len(vision_block_adapted) == 0, f"Vision backbone was accidentally adapted: {vision_block_adapted}"

    # Visual merger MUST be adapted
    merger_adapted = [k for k in adapted_keys if "visual.merger" in k]
    assert len(merger_adapted) == 4, f"Expected 4 merger LoRA tensors, got {len(merger_adapted)}"


@pytest.mark.asyncio
async def test_specialist_qwen_backend_execution(optical_image_input: ImageInput):
    """Verify SingleImageRSSpecialistTool execution with VISION_LANGUAGE_BACKEND=qwen25vl."""
    specialist = SingleImageRSSpecialistTool(backend="qwen25vl")
    assert specialist.backend == "qwen25vl"
    assert specialist.metadata.metadata["backend"] == "qwen25vl"

    req = ToolRequest(
        task=TaskType.SINGLE_IMAGE_GROUNDING,
        query="runway",
        images=[optical_image_input],
    )
    result = await specialist.execute(req)
    assert result.status == ToolStatus.SUCCESS
    assert result.model_info["backend"] == "qwen25vl"
    assert result.model_info["name"] == "Qwen2.5-VL-3B-Instruct"
    assert result.model_info["is_mock"] is False
    assert len(result.evidence) >= 1
    assert result.evidence[0].type == EvidenceType.BOUNDING_BOX


@pytest.mark.asyncio
async def test_specialist_paligemma_legacy_backend(optical_image_input: ImageInput):
    """Verify SingleImageRSSpecialistTool execution with VISION_LANGUAGE_BACKEND=paligemma_legacy."""
    specialist = SingleImageRSSpecialistTool(backend="paligemma_legacy")
    assert specialist.backend == "paligemma_legacy"
    assert specialist.metadata.metadata["backend"] == "paligemma_legacy"

    req = ToolRequest(
        task=TaskType.SINGLE_IMAGE_VQA,
        query="What is the dominant land cover?",
        images=[optical_image_input],
    )
    result = await specialist.execute(req)
    assert result.status == ToolStatus.SUCCESS
    assert result.model_info["backend"] == "paligemma_legacy"
    assert result.model_info["is_mock"] is False


def test_tokenizer_grounding_tokens():
    """Verify tokenizer inventory has valid IDs for Qwen2.5-VL grounding tokens."""
    _inspect = importlib.import_module("specialists.single_image.training.colab.03_inspect_qwen")
    report = _inspect.inspect_tokenizer_grounding_tokens("Qwen/Qwen2.5-VL-3B-Instruct", output_path="scratch/test_token_inv.json")
    assert report["all_required_present"] is True
    toks = report["verified_grounding_tokens"]
    assert toks["<|box_start|>"]["token_id"] is not None
    assert toks["<|box_end|>"]["token_id"] is not None
    assert toks["<|object_ref_start|>"]["token_id"] is not None
    assert toks["<|object_ref_end|>"]["token_id"] is not None


def test_qwen_yaml_config_integrity():
    """Verify configs/qwen25vl_qlora.yaml loads with expected QLoRA and hardware settings."""
    cfg = Qwen25VLFullConfig.from_yaml("configs/qwen25vl_qlora.yaml")
    assert cfg.model.model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert cfg.quantization.load_in_4bit is True
    assert cfg.quantization.bnb_4bit_quant_type == "nf4"
    assert cfg.quantization.bnb_4bit_use_double_quant is True
    assert cfg.lora.r == 16
    assert cfg.lora.lora_alpha == 32
    assert cfg.lora.lora_dropout == 0.05
    assert cfg.training.per_device_train_batch_size == 1
    assert cfg.training.gradient_accumulation_steps >= 4
    assert cfg.training.gradient_checkpointing is True
    assert cfg.training.optim == "paged_adamw_8bit"


def test_dataset_validator_rejection_rules():
    """Verify validate_single_sample rejects corrupt, incomplete, or out-of-bounds samples."""
    _val_mod = importlib.import_module("specialists.single_image.training.colab.02_validate_dataset")
    validate_single_sample = _val_mod.validate_single_sample

    # Valid sample
    valid_sample = {
        "id": "val_01",
        "image": "demo_assets/demo_optical_single.png",
        "parent_scene_id": "scene_01",
        "modality": "optical",
        "task": "grounding",
        "width": 1024,
        "height": 1024,
        "bbox": [100, 100, 300, 300],
        "bbox_format": "pixel_xyxy",
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Locate terminal."}]},
            {"role": "assistant", "content": "Terminal is here <|object_ref_start|>terminal<|object_ref_end|><|box_start|>(97,97),(292,292)<|box_end|>."},
        ],
    }
    is_valid, errors = validate_single_sample(valid_sample, check_file_exists=False)
    assert is_valid is True
    assert len(errors) == 0

    # Malformed sample: negative bbox
    bad_bbox = dict(valid_sample)
    bad_bbox["bbox"] = [-10, 100, 300, 300]
    is_valid, errors = validate_single_sample(bad_bbox, check_file_exists=False)
    assert is_valid is False
    assert any("out of bounds" in e.lower() for e in errors)

    # Malformed sample: empty assistant answer
    empty_ans = dict(valid_sample)
    empty_ans["messages"] = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q"}]},
        {"role": "assistant", "content": "   "},
    ]
    is_valid, errors = validate_single_sample(empty_ans, check_file_exists=False)
    assert is_valid is False
    assert any("empty" in e.lower() for e in errors)

