# Apple Silicon (MPS) Compatibility

## The Problem

TRIBE v2 was built for NVIDIA GPUs (CUDA). The original codebase assumes CUDA is available everywhere. Running on a Mac Mini M4 (Apple Silicon with Metal Performance Shaders / MPS) required solving several problems:

1. **Hard-coded `device="cuda"`** in the HuggingFace model config
2. **LLaMA 3.2 incompatible with MPS** due to Grouped Query Attention
3. **Memory constraints** — 16 GB unified memory shared between CPU/GPU
4. **neuralset library** doesn't know about MPS devices

## The Solution: Monkey-Patching

We created `tribev2/_mps_compat.py` — a module that patches the `neuralset` library at import time to support MPS. It is imported automatically by `tribev2/__init__.py` before any model code runs.

### How `__init__.py` Triggers It

```python
# tribev2/__init__.py
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Step 1
import tribev2._mps_compat  # Step 2 — patches applied on import
from tribev2.demo_utils import TribeModel
```

**Step 1**: `PYTORCH_ENABLE_MPS_FALLBACK=1` tells PyTorch to silently fall back to CPU for any Metal operations that aren't implemented yet (instead of crashing).

**Step 2**: Importing `_mps_compat` runs `_patch_mps_support()` which modifies the `neuralset` library's classes.

### What Gets Patched

#### Patch 1: HuggingFaceMixin.model_post_init

The pretrained config from `facebook/tribev2` sets `device: "cuda"` for all feature extractors. When there's no CUDA, the upstream code resolves `"auto"` to `"cpu"`. We intercept this:

```python
# Before the original model_post_init runs:
if self.device == "cuda":
    self.device = "auto"    # Pretend it said "auto" so pydantic doesn't reject "cuda"

# After the original model_post_init runs (which resolves "auto" → "cpu"):
if isinstance(self, HuggingFaceText):
    pass  # TEXT STAYS ON CPU — Llama GQA doesn't work on MPS
else:
    self.device = "mps"  # Everything else goes to MPS
```

**Why text stays on CPU**: LLaMA 3.2-3B uses Grouped Query Attention (GQA) — 24 attention heads sharing 8 key-value heads. The MPS `mps_matmul` kernel can't handle the resulting matrix shapes and throws `"incompatible dimensions"`. On Apple Silicon, CPU and MPS share the same physical memory (unified memory), so keeping the text model on CPU doesn't waste RAM — it just uses a different compute API.

#### Patch 2: HuggingFaceText._load_model (Float16)

To fit in 16 GB of RAM, the text model loads in float16 precision:

```python
# Llama-3.2-3B at float32: ~12.8 GB
# Llama-3.2-3B at float16: ~6.4 GB  ← we use this
kwargs.setdefault("torch_dtype", torch.float16)
```

INT8 quantization was tested but produced NaN values in later layers due to overflow in attention computations.

#### Patch 3: OpticalFlow.model_post_init

Same as Patch 1 but for the optical flow extractor (a video analysis component that also hardcodes CUDA).

### Memory Management

The project also frees feature extractor models after their features are cached to disk:

```python
# tribev2/main.py, _free_extractor_model()
# After each extractor finishes, delete its GPU model to free memory
del target._model
del target._tokenizer
gc.collect()
torch.mps.empty_cache()
```

This is critical because the three feature extractors (V-JEPA, Wav2Vec, LLaMA) collectively use more memory than is available. They run sequentially, and each one is freed before the next loads.

### Other Modifications

#### DataLoader Workers

The pretrained config uses 20 DataLoader workers (for cluster use). On the Mac Mini with 10 cores, this causes excessive subprocess spawning overhead:

```python
# tribev2/demo_utils.py, from_pretrained()
max_workers = min(multiprocessing.cpu_count(), 4)
if config.get("data.num_workers", 99) > max_workers:
    config["data.num_workers"] = max_workers
```

#### Float16 Inference

The TRIBE model itself (the Transformer, not the feature extractors) is converted to float16 for inference:

```python
# tribev2/demo_utils.py, from_pretrained()
model.half()  # float16 inference saves ~338 MB
```

### The Notebook Update Script

`scripts/update_demo_notebook.py` patches the original `tribe_demo.ipynb` to work on Mac M4:
- Replaces Colab-specific instructions with local Mac instructions
- Updates model loading to use HuggingFace Hub
- Clears stale cell outputs
- Adds MPS compatibility notes

## Summary: What Runs Where

| Component | Device | Precision | Memory |
|-----------|--------|-----------|--------|
| V-JEPA2 (video extractor) | MPS | float32 | ~2 GB |
| Wav2Vec-BERT (audio extractor) | MPS | float32 | ~1 GB |
| LLaMA 3.2-3B (text extractor) | **CPU** | **float16** | ~6.4 GB |
| TRIBE v2 Transformer | MPS | float16 | ~0.3 GB |

Each feature extractor runs sequentially and is freed before the next one loads, so peak memory stays within the 16 GB limit.

## Files Involved

| File | What It Does |
|------|-------------|
| `tribev2/__init__.py` | Sets `PYTORCH_ENABLE_MPS_FALLBACK=1`, imports `_mps_compat` |
| `tribev2/_mps_compat.py` | All monkey-patches for MPS support |
| `tribev2/demo_utils.py` | `from_pretrained()` caps workers, loads model in float16 |
| `tribev2/main.py` | `_free_extractor_model()` frees memory between stages |
| `scripts/update_demo_notebook.py` | Patches the demo notebook for Mac compatibility |
