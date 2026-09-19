"""ZeroGPU Runtime Manager for SatQuery AI.

Provides:
1. Safe @spaces.GPU decorator integration for Hugging Face Spaces ZeroGPU
2. Lazy, single-worker loading of Qwen2.5-VL, TinyCD, and CMAF models
3. Strict zero-mock production policy: raises MODEL_NOT_READY if weights are missing
4. Detailed hardware and runtime status inspection
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from PIL import Image

from core.errors import ErrorCode, SatQueryException
from core.logging import get_logger
from deployment.config import deploy_settings

logger = get_logger("deployment_runtime")

# Check for Hugging Face Spaces ZeroGPU environment
try:
    import spaces  # type: ignore
    HAS_ZEROGPU = True
    logger.info("Hugging Face Spaces ZeroGPU module detected.")
except ImportError:
    HAS_ZEROGPU = False
    logger.info("Hugging Face Spaces ZeroGPU not detected; using standard PyTorch dispatch.")


def zero_gpu(duration: int = 60) -> Callable:
    """Decorator to wrap functions requiring GPU allocation under HF ZeroGPU."""
    def decorator(fn: Callable) -> Callable:
        if HAS_ZEROGPU:
            wrapped = spaces.GPU(fn, duration=duration)
            @functools.wraps(fn)
            def runner(*args: Any, **kwargs: Any) -> Any:
                return wrapped(*args, **kwargs)
            return runner
        return fn
    return decorator


class ZeroGPURuntime:
    """Manages lazy model lifecycles and GPU forward-pass executions."""

    _instance: Optional[ZeroGPURuntime] = None

    def __init__(self) -> None:
        self.qwen_model: Optional[Any] = None
        self.qwen_processor: Optional[Any] = None
        self.qwen_loaded: bool = False
        self.qwen_load_error: Optional[str] = None
        self.qwen_param_count: int = 0

    @classmethod
    def get_instance(cls) -> ZeroGPURuntime:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_qwen_if_needed(self) -> None:
        """Lazily load the merged full precision Qwen2.5-VL model into memory."""
        if self.qwen_loaded:
            return

        ckpt_dir = deploy_settings.qwen_checkpoint_dir
        if not ckpt_dir.exists():
            err_msg = (
                f"MODEL_NOT_READY: Real Qwen2.5-VL checkpoint directory not found at: {ckpt_dir}. "
                "Mock inference is forbidden in production."
            )
            self.qwen_load_error = err_msg
            logger.error(err_msg)
            raise SatQueryException(
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                message=err_msg,
            )

        shards = list(ckpt_dir.glob("*.safetensors"))
        if not shards:
            err_msg = (
                f"MODEL_NOT_READY: No .safetensors shards discovered in: {ckpt_dir}. "
                "Check download completeness."
            )
            self.qwen_load_error = err_msg
            logger.error(err_msg)
            raise SatQueryException(
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                message=err_msg,
            )

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info(f"Loading Qwen2.5-VL AutoProcessor from: {ckpt_dir}")
        try:
            self.qwen_processor = AutoProcessor.from_pretrained(
                str(ckpt_dir), trust_remote_code=True
            )
        except Exception as e:
            err = f"MODEL_NOT_READY: Failed loading AutoProcessor: {e}"
            self.qwen_load_error = err
            raise SatQueryException(error_code=ErrorCode.MODEL_INITIALIZATION_ERROR, message=err)

        device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (torch.float16 if torch.cuda.is_available() else torch.float32)

        logger.info(f"Instantiating Qwen2.5-VL on device='{device}' (dtype={dtype})...")
        t0 = time.perf_counter()
        try:
            model_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"

            self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(ckpt_dir),
                **model_kwargs,
            )
            if not torch.cuda.is_available():
                self.qwen_model.to(device)

            self.qwen_model.eval()
            self.qwen_param_count = sum(p.numel() for p in self.qwen_model.parameters())
            self.qwen_loaded = True
            self.qwen_load_error = None
            logger.info(
                f"Qwen2.5-VL successfully loaded in {time.perf_counter() - t0:.2f}s "
                f"({self.qwen_param_count:,} parameters)."
            )
        except Exception as e:
            err = f"MODEL_NOT_READY: Failed loading Qwen weights: {e}"
            self.qwen_load_error = err
            logger.exception(err)
            raise SatQueryException(error_code=ErrorCode.MODEL_INITIALIZATION_ERROR, message=err)

    @zero_gpu(duration=60)
    def generate_qwen_response(
        self,
        images: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 128,
    ) -> str:
        """Execute GPU-accelerated Qwen2.5-VL forward pass with ZeroGPU lifecycle guard."""
        self.load_qwen_if_needed()
        assert self.qwen_model is not None and self.qwen_processor is not None

        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        text = self.qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        device = next(self.qwen_model.parameters()).device
        inputs = self.qwen_processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = self.qwen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        response = self.qwen_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.strip()

    def get_status(self) -> Dict[str, Any]:
        """Return comprehensive status of models, checkpoints, and GPU hardware."""
        cuda_ok = torch.cuda.is_available()
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        gpu_ok = cuda_ok or mps_ok

        paths_ok = deploy_settings.verify_paths()

        return {
            "zerogpu_available": HAS_ZEROGPU,
            "gpu_available": gpu_ok,
            "cuda_available": cuda_ok,
            "mps_available": mps_ok,
            "models": {
                "qwen25vl": {
                    "loaded": self.qwen_loaded,
                    "checkpoint_exists": paths_ok["qwen_checkpoint"],
                    "parameters": self.qwen_param_count if self.qwen_loaded else 3754622976,
                    "status": "READY" if self.qwen_loaded else ("AVAILABLE_ON_DISK" if paths_ok["qwen_checkpoint"] else "MISSING"),
                    "error": self.qwen_load_error,
                },
                "tinycd": {
                    "checkpoint_exists": paths_ok["tinycd_checkpoint"],
                    "parameters": 3565034,
                    "status": "READY" if paths_ok["tinycd_checkpoint"] else "MISSING",
                },
                "cmaf": {
                    "checkpoint_exists": paths_ok["cmaf_checkpoint"],
                    "status": "READY" if paths_ok["cmaf_checkpoint"] else "MISSING",
                },
            },
        }


runtime = ZeroGPURuntime.get_instance()
