# Text-Only TribeV2 Optimization — Mac Mini M4 16GB

## Current Architecture Overview

Your text-only pipeline has **3 sequential stages**, each with its own model:

```mermaid
graph LR
    A["📝 Input Text"] --> B["🔊 gTTS\n(text→speech)"]
    B --> C["🎙️ WhisperX\nlarge-v3\n~3 GB RAM"]
    C --> D["🧠 Llama-3.2-3B\n~6–13 GB RAM"]
    D --> E["🧩 FmriEncoder\n~676 MB RAM"]
    E --> F["📊 Brain\nPredictions"]
```

### Current Memory Footprint (Worst Case)

| Component | Precision | Weight Memory | Peak w/ Overhead |
|:---|:---|:---|:---|
| WhisperX large-v3 (faster-whisper) | int8 | ~1.6 GB | ~2.5 GB |
| Llama-3.2-3B | float32 (default!) | **~12.8 GB** | **~14–16 GB** |
| FmriEncoder (brain model) | float32 | ~676 MB | ~800 MB |
| **Theoretical peak** | | | **~19 GB** ❌ |

> [!CAUTION]
> **The models load sequentially** (WhisperX → Llama → FmriEncoder), but neither WhisperX nor Llama are explicitly freed before the brain model runs inference. On 16GB unified memory (with ~2–3GB reserved by macOS), you're at serious risk of swap thrashing.

### Key Code Path for Text Input

1. [demo_utils.py:303-310](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/demo_utils.py#L303-L310) — `TextToEvents` converts text → TTS audio → events
2. [eventstransforms.py:86-218](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/eventstransforms.py#L86-L218) — `ExtractWordsFromAudio` runs WhisperX for transcription
3. [text.py:384-462](file:///Volumes/dev/ai_work/meta/tribe_v2/code/.venv/lib/python3.11/site-packages/neuralset/extractors/text.py#L384-L462) — `HuggingFaceText._get_data()` loads Llama and extracts hidden states
4. [demo_utils.py:356-378](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/demo_utils.py#L356-L378) — `TribeModel.predict()` runs FmriEncoder inference

### What the Config Actually Requests from Llama

From [config.yaml](file:///Users/ganesha/.cache/huggingface/hub/models--facebook--tribev2/snapshots/f894e783020944dcd96e5568550afe2aa9743f9f/config.yaml):
- **Model**: `meta-llama/Llama-3.2-3B` (28 hidden layers, 3072 hidden dim, GQA 24/8 heads)
- **Layers used**: `[0.5, 0.75, 1.0]` → layers **13, 20, 27** out of 28
- **cache_n_layers**: 20 (caches 20 equidistant layers to disk for reuse)
- **layer_aggregation**: `group_mean` → averages into 2 groups: [13–20] and [20–27]
- **token_aggregation**: `mean` → averages all token embeddings per word
- **Output shape**: `(2, 3072)` per word — just 2 layer-groups × 3072 dim

> [!IMPORTANT]
> The pipeline extracts **all 28 layers** of hidden states via `output_hidden_states=True`, even though it only **uses 3 layers** (indices 13, 20, 27). This wastes massive memory storing 25 unnecessary layer tensors.

---

## Proposed Optimizations (Tiered)

### Tier 1: Quick Wins — No Code Changes Required

#### 1A. Load Llama in float16

The Llama-3.2-3B model's native dtype is `bfloat16`, but the current `_load_model` in neuralset loads it in default precision (`float32`) when running on CPU. Since your MPS compat patch forces the text model to CPU (due to GQA incompatibility), it loads as **12.8 GB float32**.

**Fix**: Pass `torch_dtype=torch.float16` to `from_pretrained()`. This halves memory to **~6.4 GB**.

> [!WARNING]
> `float16` on CPU is slower than `float32` for compute (no native FP16 ALU on CPU). However, for this use case the bottleneck is memory, not compute — and the model runs through a single forward pass per word batch, cached to disk. The slight compute slowdown is negligible vs. avoiding swap death.

**Implementation** — patch [_mps_compat.py](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/_mps_compat.py) `_text_load_model_cpu`:
```python
def _text_load_model_cpu(self, **kwargs):
    if self.device in ("mps", "cuda"):
        self.device = "cpu"
    # Load in half precision to halve memory from ~12.8GB to ~6.4GB
    if "torch_dtype" not in kwargs:
        kwargs["torch_dtype"] = torch.float16
    return _orig_text_load_model(self, **kwargs)
```

**Memory savings**: **~6.4 GB** ✅

---

#### 1B. Use WhisperX `large-v3-turbo` Instead of `large-v3`

The transcription step uses `large-v3` (1.55B params). The `large-v3-turbo` model provides nearly identical accuracy with **significantly** faster inference and lower memory.

**Implementation** — change [eventstransforms.py line 124](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/eventstransforms.py#L124):
```python
# Before
"--model", "large-v3",
# After
"--model", "large-v3-turbo",
```

Or, for aggressive memory savings, use `medium` (~769M params, ~1.5 GB vs ~3 GB).

**Memory savings**: ~0.5–1.5 GB depending on model choice

---

#### 1C. Reduce WhisperX Batch Size

Currently hardcoded to `--batch_size 16`. On 16GB RAM, reduce to `4` or `8`:

```python
"--batch_size", "4",  # was "16"
```

**Memory savings**: Reduces peak WhisperX memory by ~30–50%

---

### Tier 2: Medium Effort — Targeted Code Changes

#### 2A. Forward Hooks for Selective Layer Extraction

**The biggest single optimization.** Currently, `output_hidden_states=True` forces Llama to store all 28 layers of hidden states in memory simultaneously. For a batch of 4 contexts × 1024 tokens × 3072 dims × 28 layers × 4 bytes = **~1.3 GB** of activation memory per batch — on top of the model weights.

**Fix**: Use PyTorch forward hooks to capture only the 3 needed layers (13, 20, 27), and **don't pass `output_hidden_states=True`**.

**Implementation** — monkey-patch `HuggingFaceText._get_data` in `_mps_compat.py`:

```python
def _patched_get_data(self, events):
    """Memory-efficient layer extraction using forward hooks."""
    # Determine which layer indices we actually need
    n_model_layers = 28  # Llama-3.2-3B
    layers = self.layers if isinstance(self.layers, list) else [self.layers]
    needed_indices = set(int(l * (n_model_layers - 1)) for l in layers)
    
    # Also include cache_n_layers if set
    if self.cache_n_layers:
        cached = [int(round(x)) for x in np.linspace(0, n_model_layers - 1, self.cache_n_layers)]
        needed_indices.update(cached)
    
    captured = {}
    hooks = []
    for idx in needed_indices:
        layer = self.model.model.layers[idx]
        def hook_fn(module, input, output, layer_idx=idx):
            captured[layer_idx] = output[0].detach().cpu()
        hooks.append(layer.register_forward_hook(hook_fn))
    
    try:
        # Run forward pass WITHOUT output_hidden_states
        # ... (process batches, use captured dict instead of outputs.hidden_states)
        pass
    finally:
        for h in hooks:
            h.remove()
```

**Memory savings**: Eliminates ~1+ GB of activation storage per batch

> [!NOTE]
> This is more complex to implement because it requires replacing the inner loop of `_get_data`. We'd need to either subclass `HuggingFaceText` or do a more aggressive monkey-patch. Worth it if Tier 1 isn't enough.

---

#### 2B. Aggressive Memory Cleanup Between Pipeline Stages

The pipeline loads WhisperX → frees it (via subprocess) → loads Llama → **doesn't free it** → loads FmriEncoder.

**Fix**: Ensure Llama is explicitly deleted after `extractor.prepare()`. The existing [_free_extractor_model](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/main.py#L59-L82) already does this! But verify it works correctly for the `HuggingFaceText` class:

```python
# In _free_extractor_model, ensure _tokenizer is also freed:
for attr in ("_model", "_tokenizer"):
    obj = getattr(target, attr, None)
    if obj is not None:
        try:
            delattr(target, attr)
        except Exception:
            pass
```

**Memory savings**: Frees ~6.4 GB (after Tier 1A) before brain model loads

---

#### 2C. Brain Model in float16

The FmriEncoder loads as float32 (676 MB). Convert to float16 for inference:

**Implementation** — in [demo_utils.py line 238](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/demo_utils.py#L238):
```python
# Before
model.to(device)
# After
model.to(device=device, dtype=torch.float16)
```

**Memory savings**: ~338 MB

---

### Tier 3: Text-Only Mode — Skip Unused Modalities Entirely

Since you only care about text, the pipeline currently still **configures** audio and video extractors (they just produce zeros due to `allow_missing=True`). But the brain model still allocates parameters for all 3 modality projectors.

#### 3A. Text-Only Config Override

Pass `config_update` to `from_pretrained` to restrict to text only:

```python
model = TribeModel.from_pretrained(
    "facebook/tribev2",
    config_update={
        "data.features_to_use": ["text"],
    }
)
```

This won't save model weight memory (projectors are tiny), but it avoids instantiating unused feature extractors.

---

### Tier 4: Quantization (Advanced)

#### 4A. INT8 Quantization via `torchao` or `quanto`

If Tier 1–3 aren't enough, quantize Llama to INT8 or INT4:

```python
# Using torchao (works on CPU)
import torchao
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
torchao.quantize_(model, torchao.int8_weight_only())
```

**Memory savings**: ~3.2 GB (INT8) or ~1.6 GB (INT4) for model weights

> [!WARNING]
> INT4/INT8 quantization may slightly alter the hidden state values, which could affect downstream brain prediction accuracy. Needs validation.

---

## Optimized Memory Profile (After Tier 1 + 2)

| Component | Before | After | Savings |
|:---|:---|:---|:---|
| WhisperX (transcription) | ~2.5 GB peak | ~1.5 GB (turbo) | 1.0 GB |
| Llama-3.2-3B (weights) | 12.8 GB | **6.4 GB** (fp16) | **6.4 GB** |
| Llama activations/batch | ~1.3 GB | ~0.15 GB (hooks) | 1.15 GB |
| FmriEncoder (inference) | ~0.8 GB | ~0.5 GB (fp16) | 0.3 GB |
| **Sequential peak** | **~14–16 GB** ❌ | **~7–8 GB** ✅ | **~8 GB** |

> [!TIP]
> Because the stages run **sequentially** and models are freed between stages, the peak memory at any given time would be the **maximum** of any single stage — not the sum. With proper cleanup (Tier 2B), your peak would be **~7 GB** (Llama stage), well within 16 GB.

---

## Recommended Implementation Order

1. **Tier 1A** — `torch_dtype=float16` for Llama ← **biggest bang for buck**
2. **Tier 2B** — Verify aggressive memory cleanup between stages
3. **Tier 1B** — Switch to `large-v3-turbo` for WhisperX
4. **Tier 1C** — Reduce WhisperX batch size to 4
5. **Tier 2C** — Brain model in float16
6. **Tier 3A** — Text-only config override
7. **Tier 2A** — Forward hooks (only if still memory-constrained)
8. **Tier 4A** — Quantization (nuclear option)

## Speed Optimizations

Beyond memory, several changes improve inference **speed**:

| Optimization | Impact | Notes |
|:---|:---|:---|
| WhisperX turbo model | ~2× faster transcription | Near-identical accuracy |
| Feature caching (`exca`) | Instant on 2nd run | Already built into pipeline |
| Smaller Llama batch size | Prevents swap thrashing | Slower per-word but faster overall |
| `torch.inference_mode()` | ~5–10% faster | Already used in `predict()`, needs adding to feature extraction |
| Reduce `max_context_len` | Fewer tokens to process | Currently 1024, try 512 for shorter texts |

## Verification Plan

### Automated Tests
1. Run text-only inference before and after changes
2. Compare brain prediction outputs (Pearson correlation should be >0.99 for fp16 changes)
3. Monitor peak memory with `torch.mps.current_allocated_memory()` + Activity Monitor

### Manual Verification
- Process `sample_text.txt` end-to-end
- Verify no swap usage via `vm_stat` or Activity Monitor
- Time the full pipeline (target: <60s for short text)

## Open Questions

> [!IMPORTANT]
> 1. **Accuracy tolerance**: How much deviation in brain predictions is acceptable from fp16 vs fp32? (Typical answer: negligible for inference, but worth confirming with a correlation check.)
> 2. **WhisperX model**: Is `large-v3-turbo` acceptable, or does your use case require the original `large-v3` for transcription accuracy?
> 3. **Do you want me to implement Tiers 1+2 now?** These are the highest-impact, lowest-risk changes.
