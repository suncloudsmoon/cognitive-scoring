# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Monkey-patch neuralset extractors to support Apple Silicon MPS device.

The ``neuralset`` package only knows about ``"cuda"`` and ``"cpu"`` for its
auto-device detection.  This module patches the relevant ``model_post_init``
methods so that both ``device="auto"`` and ``device="cuda"`` resolve to
``"mps"`` on Apple Silicon when CUDA is unavailable.

The pretrained config from ``facebook/tribev2`` hardcodes ``device: "cuda"``
for all extractors, so we must intercept *both* ``"auto"`` and ``"cuda"``.

**Exception — HuggingFaceText stays on CPU:**
Llama-3.2-3B uses Grouped Query Attention (24 heads, 8 KV heads) whose
attention matmul shapes cannot be broadcast by the MPS ``mps_matmul`` kernel
(``error: incompatible dimensions``).  Both ``model_post_init`` and
``_load_model`` are patched to ensure the text extractor never runs on MPS.

**Memory optimizations (Apple Silicon / 16 GB):**
- Model weights loaded in float16 (~6.4 GB instead of ~12.8 GB float32).
- INT8 weight-only quantization via ``torchao`` when available (~3.2 GB).
  Falls back to float16 if ``torchao`` is not installed.

Import this module **before** any neuralset extractor is instantiated (it is
imported automatically by ``tribev2/__init__.py``).
"""

import logging

import torch

logger = logging.getLogger(__name__)


def _get_best_device() -> str:
    """Return the best available PyTorch compute device string."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _try_int8_quantize(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """Apply INT8 weight-only quantization if torchao is available.

    INT8 weight-only quantization halves the float16 weight memory from
    ~6.4 GB to ~3.2 GB for Llama-3.2-3B with negligible quality loss.
    Activations remain in the original precision during the forward pass.

    Returns the (possibly quantized) model.
    """
    try:
        from torchao.quantization import int8_weight_only, quantize_

        quantize_(model, int8_weight_only())
        logger.info(
            "Applied INT8 weight-only quantization to %s (weights ~3.2 GB)",
            model_name,
        )
    except ImportError:
        logger.info(
            "torchao not installed — using float16 weights for %s (~6.4 GB). "
            "Install torchao for INT8 quantization: pip install torchao",
            model_name,
        )
    except Exception as e:
        logger.warning(
            "INT8 quantization failed for %s, falling back to float16: %s",
            model_name,
            e,
        )
    return model


def _patch_mps_support() -> None:
    """Patch neuralset extractors for MPS support on Apple Silicon."""
    from neuralset.extractors import base as ns_base
    from neuralset.extractors import text as ns_text

    best = _get_best_device()
    if best != "mps":
        return

    logger.debug("Patching neuralset extractors for MPS device support")

    # ── HuggingFaceMixin.model_post_init ─────────────────────────────────
    # Intercepts device resolution for most extractors (image, audio, video).
    # Handles both device="auto" (resolves to "cpu" upstream) and
    # device="cuda" (hardcoded in the HF config) → redirect to "mps".
    #
    # IMPORTANT: HuggingFaceText is kept on "cpu" because Llama-3.2-3B's
    # Grouped Query Attention (24 attn heads, 8 KV heads) triggers an MPS
    # mps_matmul kernel error ("incompatible dimensions").  The device must
    # be set correctly HERE (during model_post_init, before pydantic freeze)
    # rather than in _load_model, because _get_data reads self.device
    # before _load_model is called.  On Apple Silicon unified memory,
    # CPU and MPS share the same physical memory pool.
    _orig_hf_post_init = ns_base.HuggingFaceMixin.model_post_init

    def _hf_model_post_init(self, log__):
        orig_device = getattr(self, "device", None)
        # If device was explicitly "cuda", override *before* the upstream
        # model_post_init runs, since validation may reject unknown values.
        if orig_device == "cuda":
            self.device = "auto"
        _orig_hf_post_init(self, log__)
        # After upstream resolution: "auto" → "cpu" (no CUDA).
        if orig_device in ("auto", "cuda") and self.device == "cpu":
            # Text models stay on CPU due to GQA incompatibility with MPS.
            # All other extractors (image, audio, video) go to MPS.
            if isinstance(self, ns_text.HuggingFaceText):
                logger.debug(
                    "Keeping %s on CPU (Llama GQA incompatible with MPS)",
                    getattr(self, "model_name", "text"),
                )
            else:
                self.device = "mps"

    ns_base.HuggingFaceMixin.model_post_init = _hf_model_post_init

    # ── HuggingFaceText._load_model — FLOAT16 ───────────────────────────
    # Device is already set to "cpu" in model_post_init above.
    # This patch loads weights in float16 (~6.4 GB vs ~12.8 GB float32).
    #
    # NOTE: INT8 weight-only quantization (torchao) was tested but produces
    # NaN values in Llama's later layers when combined with float16
    # activations — hidden states reach ±300 which overflows during
    # INT8 dequantization.  Float16-only is the safe choice here.
    _orig_text_load_model = ns_text.HuggingFaceText._load_model

    def _text_load_model_cpu(self, **kwargs):
        # Safety check: ensure device is CPU even if model_post_init
        # was bypassed or the model was created with an explicit device.
        if self.device in ("mps", "cuda"):
            logger.info(
                "Forcing text model %s to CPU (Llama GQA incompatible with MPS)",
                self.model_name,
            )
            object.__setattr__(self, "device", "cpu")

        # Load in float16 to halve peak memory during model loading.
        # Llama-3.2-3B native dtype is bfloat16; float16 is safe and
        # avoids the CPU performance penalty of bfloat16 emulation.
        kwargs.setdefault("torch_dtype", torch.float16)

        model = _orig_text_load_model(self, **kwargs)

        return model

    ns_text.HuggingFaceText._load_model = _text_load_model_cpu

    # ── OpticalFlow (video) ──────────────────────────────────────────────
    # Guard with try/except so the package works without torchvision.
    # Video support is optional: install with `pip install tribev2[video]`.
    try:
        from neuralset.extractors import video as ns_video

        _orig_of_post_init = ns_video.OpticalFlow.model_post_init

        def _of_model_post_init(self, log__):
            orig_device = getattr(self, "device", None)
            if orig_device == "cuda":
                self.device = "auto"
            _orig_of_post_init(self, log__)
            if orig_device in ("auto", "cuda") and self.device == "cpu":
                self.device = "mps"

        ns_video.OpticalFlow.model_post_init = _of_model_post_init
    except (ImportError, ModuleNotFoundError):
        logger.debug(
            "Skipping OpticalFlow MPS patch (torchvision not installed). "
            "Install with: pip install tribev2[video]"
        )

    logger.debug("neuralset MPS patches applied")


# Apply patches on import.
_patch_mps_support()
