# Run TRIBE v2 on Mac Mini M4 (16 GB) via Metal / MPS

## Background

TRIBE v2 is a multimodal brain-encoding model that predicts fMRI responses from video, audio, and text. The codebase is written for **CUDA GPUs** (Slurm clusters with NVIDIA hardware). Your Mac Mini M4 has **no CUDA** but has **MPS (Metal Performance Shaders)** available through PyTorch. The goal is to make both **inference** (the demo / `TribeModel.predict` flow) and **training** (`TribeExperiment.run`) work on MPS.

### Current state (verified in your venv)

| Feature | Status |
|---|---|
| `torch.backends.mps.is_available()` | ✅ `True` |
| `torch.backends.mps.is_built()` | ✅ `True` |
| `torch.cuda.is_available()` | ❌ `False` |
| PyTorch version | 2.6.0 |

## User Review Required

> [!IMPORTANT]
> **Memory budget is tight.** 16 GB unified memory must be shared between macOS (~3–4 GB), the OS kernel, and the model pipeline. The default configuration uses `facebook/vjepa2-vitg-fpc64-256` (ViT-G, ~1.2B params, ~5 GB fp32) plus `meta-llama/Llama-3.2-3B` (~12 GB fp32 / ~6 GB fp16). **These will not fit simultaneously in 16 GB.** The plan uses the `mini_config` models (`vjepa2-vitl`, `Qwen3-0.6B`) and sequential memory management. See the Memory Budget section below.

> [!WARNING]
> **The `neuralset` package (installed via pip, version 0.0.2) needs a small source patch** to its `HuggingFaceMixin.device` type literal to accept `"mps"`. This is a 1-line change in the installed package. An alternative is to monkey-patch at import time. Both options are described below — please choose your preference.

> [!CAUTION]
> **Feature extraction is the bottleneck.** Even with smaller models, each extractor (text, audio, video) loads a multi-hundred-MB to multi-GB neural network onto MPS, runs inference over potentially long media, then frees the memory. The code already does sequential extraction with `_free_extractor_model()`, but we will reinforce this with explicit `torch.mps.empty_cache()` calls. Expect the first-time feature extraction to be **slow** (minutes to tens of minutes for a few-minute video), but cached results will be reused on subsequent runs.

---

## Memory Budget Analysis

| Component | Model | Precision | Memory (approx.) |
|---|---|---|---|
| **Text extractor** | `Qwen/Qwen3-0.6B` | fp32 | ~2.5 GB |
| **Video extractor** | `facebook/vjepa2-vitl-fpc64-256` (ViT-L, ~300M params) | fp32 | ~1.5 GB |
| **Audio extractor** | `facebook/w2v-bert-2.0` (~580M params) | fp32 | ~2.5 GB |
| **TRIBE brain model** | FmriEncoder (Transformer, ~15M params) | fp32 | ~0.1 GB |
| **WhisperX** (subprocess) | `large-v3` via `uvx whisperx` | fp16 | ~3 GB (separate process) |
| **macOS + runtime** | — | — | ~4 GB |

**Strategy:** Extractors run **sequentially** and each is freed before the next loads (the code already does this). WhisperX runs as a separate subprocess.  At any one time, the peak usage should be around **~7–8 GB** (macOS + one extractor + working memory), which fits in 16 GB. The brain model itself is tiny (<100 MB).

> [!IMPORTANT]
> **If you want to use the full-sized models** (`vjepa2-vitg` + `Llama-3.2-3B`), you will need to switch to `float16` or quantized inference. This plan does NOT attempt that — it uses the `mini_config` model sizes which are designed for local testing.

---

## Proposed Changes

The changes are grouped by scope. **All modifications are in the `tribev2/` package** except for one patch to the `neuralset` dependency.

---

### Component 1: Device Detection — make `"auto"` resolve to `"mps"`

Currently, every place that resolves `device = "auto"` does:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
We need to add MPS as a middle option:
```python
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
```

#### [MODIFY] [demo_utils.py](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/demo_utils.py)

- **Line 193** — `from_pretrained`: Change auto-device resolution to include MPS
  ```diff
  -            device = "cuda" if torch.cuda.is_available() else "cpu"
  +            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
  ```

#### [MODIFY] [eventstransforms.py](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/eventstransforms.py)

- **Line 107–108** — `_get_transcript_from_audio`: Add MPS option for whisperx device; also handle `compute_type` (MPS does not support `float16` the same way as CUDA for whisperx — use `int8` or `float32` on non-CUDA):
  ```diff
  -        device = "cuda" if torch.cuda.is_available() else "cpu"
  -        compute_type = "float16"
  +        if torch.cuda.is_available():
  +            device = "cuda"
  +            compute_type = "float16"
  +        else:
  +            device = "cpu"
  +            compute_type = "int8"
  ```
  > Note: WhisperX calls out via `subprocess` to `uvx whisperx`, so it manages its own device. The `--device` flag is passed as a CLI argument. WhisperX through faster-whisper does not support MPS natively; it falls back to CPU. We use CPU with int8 compute type for best CPU performance.

#### [MODIFY] [main.py](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/main.py)

- **Lines 78–79** — `_free_extractor_model`: Add MPS cache clearing alongside CUDA:
  ```diff
  -    if torch.cuda.is_available():
  -        torch.cuda.empty_cache()
  +    if torch.cuda.is_available():
  +        torch.cuda.empty_cache()
  +    if hasattr(torch, "mps") and torch.backends.mps.is_available():
  +        torch.mps.empty_cache()
  ```

---

### Component 2: PyTorch Lightning Accelerator Configuration

The `TribeExperiment` class hardcodes `accelerator: str = "gpu"` and uses `self.infra.gpus_per_node` for device count. On MPS, Lightning uses `accelerator="mps"` with `devices=1`.

#### [MODIFY] [main.py](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/main.py)

- **Line 298** — Change default accelerator to `"auto"` so Lightning can detect MPS:
  ```diff
  -    accelerator: str = "gpu"
  +    accelerator: str = "auto"
  ```

- **Lines 557–559** — Guard the `devices` argument. MPS only supports 1 device and `"fsdp"` strategy is CUDA-only. Use `"auto"` strategy unconditionally on non-multi-GPU:
  ```diff
  -        trainer = pl.Trainer(
  -            strategy="auto" if self.infra.gpus_per_node == 1 else "fsdp",
  -            devices=override_n_devices or self.infra.gpus_per_node,
  -            accelerator=self.accelerator,
  +        n_devices = override_n_devices or self.infra.gpus_per_node
  +        # MPS only supports 1 device and does not support FSDP
  +        if self.accelerator == "mps" or (self.accelerator == "auto" and not torch.cuda.is_available()):
  +            n_devices = 1
  +            strategy = "auto"
  +        else:
  +            strategy = "auto" if n_devices == 1 else "fsdp"
  +        trainer = pl.Trainer(
  +            strategy=strategy,
  +            devices=n_devices,
  +            accelerator=self.accelerator,
  ```

---

### Component 3: `neuralset` Dependency — Allow `"mps"` device

The `neuralset.extractors.base.HuggingFaceMixin` class declares:
```python
device: tp.Literal["auto", "cpu", "cuda", "accelerate"] = "auto"
```
and its `model_post_init` resolves `"auto"` → `"cuda"` or `"cpu"`. This needs `"mps"` support.

**Option A (recommended): Patch the installed package directly** — edit 2 lines in the installed `neuralset` package:

#### [MODIFY] `.venv/.../neuralset/extractors/base.py`
  ```diff
  -    device: tp.Literal["auto", "cpu", "cuda", "accelerate"] = "auto"
  +    device: tp.Literal["auto", "cpu", "cuda", "mps", "accelerate"] = "auto"
  ```
  ```diff
  -            self.device = "cuda" if torch.cuda.is_available() else "cpu"
  +            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
  ```

Similarly patch the `OpticalFlow` class in `video.py` and `test_text.py` device literals if needed (but these are not used by the inference path).

**Option B: Monkey-patch at import time** — add an `_mps_compat.py` module that patches `neuralset` at tribev2 import:

#### [NEW] [_mps_compat.py](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/_mps_compat.py)

A small module imported early in `tribev2/__init__.py` that patches the `HuggingFaceMixin.model_post_init` method to resolve `"auto"` to `"mps"` and updates the Pydantic model field to accept `"mps"`.

> [!IMPORTANT]
> **Please decide:** Option A (direct patch, simpler and more robust) or Option B (monkey-patch, no change to venv files but more fragile)?

---

### Component 4: Grid Configuration for Local Mac Runs

#### [NEW] [mac_config.py](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/grids/mac_config.py)

A new convenience config for Mac-local runs that:
- Uses mini-config models (Qwen3-0.6B, vjepa2-vitl, w2v-bert-2.0)
- Sets `accelerator="auto"`, `infra.cluster=None` (no Slurm)
- Sets `num_workers=0` (MPS doesn't play well with multiprocessing dataloaders)
- Sets `batch_size=2` (conservative for 16 GB)
- Sets all extractor `infra.cluster=None` and `infra.gpus_per_node=0` (local CPU/MPS)
- Removes wandb integration

#### [MODIFY] [test_run.py](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/grids/test_run.py)

- Add a note/import for the mac_config variant

---

### Component 5: Environment Variable Safety Net

#### [MODIFY] [__init__.py](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/__init__.py)

Add an environment variable check for MPS fallback at package import time:
```python
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
```
This ensures any unsupported MPS operations automatically fall back to CPU instead of crashing.

---

## Summary of All Changes

| File | Type | Description |
|---|---|---|
| `tribev2/__init__.py` | MODIFY | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` env var |
| `tribev2/demo_utils.py` | MODIFY | Auto-device detection: add MPS |
| `tribev2/eventstransforms.py` | MODIFY | WhisperX: use CPU + int8 when no CUDA |
| `tribev2/main.py` | MODIFY | MPS cache clearing; accelerator `auto`; Lightning trainer MPS guard |
| `tribev2/grids/mac_config.py` | NEW | Mac-local configuration using small models |
| `.venv/.../neuralset/extractors/base.py` | MODIFY | Add `"mps"` to device literal and auto-resolution |
| `tribev2/_mps_compat.py` *(Option B only)* | NEW | Monkey-patch neuralset for MPS |

---

## Open Questions

> [!IMPORTANT]
> 1. **neuralset patching approach:** Do you prefer **Option A** (directly edit the installed `.venv` package — simpler, but changes are lost on reinstall) or **Option B** (monkey-patch from tribev2 — survives reinstalls but more fragile)?

> [!IMPORTANT]
> 2. **Use case focus:** Are you primarily interested in **inference only** (loading pretrained weights from HuggingFace and running `model.predict`)? Or do you also need **training** to work on MPS? Training on MPS is functional but slower than CUDA and some operations may fall back to CPU.

> [!IMPORTANT]
> 3. **Model sizes:** The plan uses `mini_config` models (Qwen3-0.6B, vjepa2-vitl). Are you OK with these smaller models, or do you specifically need the full-sized models (Llama-3.2-3B, vjepa2-vitg)? The latter would require additional float16/quantization work.

---

## Verification Plan

### Automated Tests

1. **Device detection sanity check:**
   ```bash
   python -c "from tribev2 import TribeModel; import torch; print('MPS:', torch.backends.mps.is_available())"
   ```

2. **Inference smoke test** (loads pretrained model, processes a short test clip):
   ```bash
   python -c "
   from tribev2 import TribeModel
   model = TribeModel.from_pretrained('facebook/tribev2', cache_folder='./cache')
   print('Model device:', model._model.device)
   # Confirm it's on MPS
   "
   ```

3. **Training smoke test** (using mac_config, 1 epoch on dummy data):
   ```bash
   python -m tribev2.grids.mac_config
   ```

### Manual Verification

- Monitor memory usage via macOS Activity Monitor during inference
- Confirm no CUDA-related errors appear in logs
- Verify that feature extraction cache files are created in `./cache/`
