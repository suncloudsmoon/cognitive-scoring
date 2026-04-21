# AGENTS.md

Instructions and context for AI coding agents working on this repository.

## Project Overview

TRIBE v2 is Meta's multimodal brain encoding model that predicts fMRI brain responses from text, audio, and video. This repository wraps the upstream model with:

- **`api.py`** — `BrainAPI`: Two-line Python interface (text → brain scores)
- **`server.py`** — FastAPI HTTP server with lazy model loading + auto-unload
- **`brain_states.py`** — Destrieux atlas mapping, ROI scoring, emotional classification
- **`_mps_compat.py`** — Monkey-patches for Apple Silicon MPS device support

The upstream model code (`model.py`, `main.py`, `demo_utils.py`, `pl_module.py`) was written by Meta and should be modified conservatively.

## Setup

```bash
# Python 3.11+ required
python -m venv .venv && source .venv/bin/activate

# Core install (text + audio inference)
pip install -e .

# With all optional extras
pip install -e ".[all,test]"

# Video support (optional, adds ~2GB of dependencies)
pip install -e ".[video]"
```

**HuggingFace access:** The model downloads weights from `facebook/tribev2` and `meta-llama/Llama-3.2-3B`. The Llama model requires accepting Meta's license on HuggingFace. If gated, the config uses `alpindale/Llama-3.2-3B` as a fallback — check `grids/mac_config.py`.

## Architecture

```
User Code
    │
    ▼
BrainAPI (api.py)           ← Public API entry point
    │
    ├─► TribeModel (demo_utils.py)     ← Model loading + inference
    │       │
    │       ├─► TextToEvents            ← gTTS + WhisperX transcription
    │       ├─► Feature Extractors      ← LLaMA 3.2, Wav2Vec-BERT, V-JEPA
    │       │     (via neuralset)         (video extractors optional)
    │       └─► FmriEncoderModel        ← Fusion Transformer → 20,484 vertices
    │             (model.py)
    │
    ├─► BrainAtlas (brain_states.py)    ← Destrieux region mapping
    ├─► compute_normalized_scores()     ← Z-score + sigmoid scoring
    └─► BrainStateClassifier            ← Pearson correlation vs. reference profiles

Server (server.py)          ← FastAPI HTTP wrapper over BrainAPI
    └─► Lazy load + auto-unload after idle timeout
```

### Key Data Flow

1. **Text input** → gTTS generates speech → WhisperX transcribes back to get word-level timings
2. **Events DataFrame** — aligned text/audio/video events with timestamps
3. **Feature extraction** — each extractor produces embeddings, cached to disk by content hash
4. **Fusion Transformer** — combines all modalities → `(n_timesteps, 20484)` cortical predictions
5. **Atlas mapping** — vertices → 10 named region groups → composite scores

## Critical Design Constraints

### MPS / GPU Safety

The `TribeModel` and feature extractors are **NOT thread-safe**. All inference must be serialized:
- In `server.py`: Uses `asyncio.Lock()` to serialize requests
- In `api.py`: Single-threaded by design
- **Never** run two `analyze()` calls concurrently on the same `BrainAPI` instance

### Import Order Matters

`tribev2/__init__.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` and imports `_mps_compat` **before any other module**. This is critical because:
- The MPS fallback env var must be set before PyTorch is imported
- The monkey-patches must apply before any `neuralset` extractor is instantiated
- Do NOT move or reorder these imports

### Video Is Optional

`torchvision` and `moviepy` are optional dependencies (`pip install tribev2[video]`):
- `_mps_compat.py` guards the video extractor import with `try/except`
- `demo_utils.py` has `_check_video_deps()` that raises `ImportError` with install instructions
- Note: `torchvision` is currently still transitively installed via `neuraltrain`, but `moviepy` is truly optional

### Feature Caching

Feature extraction is expensive (~3-5 min per text). Results are cached:
- Cache key = deterministic hash of text content → temp file path
- Location: `TRIBE_CACHE_DIR` (default `./cache/`)
- Safe to delete the entire cache directory; it will rebuild on next inference
- The `exca` library manages cache persistence

### Memory on Apple Silicon (16 GB)

- LLaMA 3.2 3B runs on **CPU** (not MPS) due to GQA kernel incompatibility
- Model weights loaded in **float16** to halve memory (~6.4 GB vs 12.8 GB)
- `_free_extractor_model()` in `main.py` releases each extractor after use
- Server auto-unloads after 2 min idle (`TRIBE_IDLE_TIMEOUT_SECONDS`)

## File Reference

### Our custom modules (safe to modify freely)

| File | Purpose |
|------|---------|
| `api.py` | `BrainAPI` + `BrainResult` — the public Python API |
| `server.py` | FastAPI HTTP server with lazy load, idle timeout, CORS |
| `brain_states.py` | Atlas, scoring, profiling, classification, visualization |
| `_mps_compat.py` | Monkey-patches for MPS device support |
| `config.py` | Persistent config manager (`~/Library/Application Support/TribeV2/`) |
| `menubar.py` | macOS menu bar app (`rumps`) — server lifecycle control |
| `settings_gui.py` | Native PyObjC settings window for the menu bar app |
| `launchd.py` | LaunchAgent manager for "Start at Login" |
| `grids/mac_config.py` | Mac-specific configuration overrides |

### Upstream Meta modules (modify conservatively)

| File | Purpose |
|------|---------|
| `model.py` | `FmriEncoder` / `FmriEncoderModel` — the Fusion Transformer |
| `main.py` | `Data` + `TribeExperiment` — training/inference pipeline |
| `demo_utils.py` | `TribeModel` — model loading, events, prediction |
| `pl_module.py` | PyTorch Lightning training module |
| `eventstransforms.py` | Text/audio/video → events DataFrame transforms |
| `utils.py` | Multi-study loading, splitting |
| `utils_fmri.py` | Surface projection (MNI ↔ fsaverage5) |
| `plotting/` | Brain visualization (nilearn, pyvista backends) |
| `studies/` | Dataset definitions for training datasets |

### Configuration and metadata

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package config, dependencies, optional extras |
| `stimuli/` | Reference text files for emotional state profiling |
| `wiki/` | Comprehensive documentation (11 pages) |

## Common Tasks

### Running the menu bar app

```bash
# Install with menu bar support
pip install -e ".[menubar]"

# Launch the menu bar app
tribe-menubar

# Or run directly
python -m tribev2.menubar
```

### Running the HTTP server

```bash
# Default (lazy load, 2min idle timeout)
python -c "from tribev2.server import main; main()"

# Custom config
TRIBE_IDLE_TIMEOUT_SECONDS=300 TRIBE_PORT=9000 python -c "from tribev2.server import main; main()"
```

### Running text inference (Python)

```python
from tribev2 import BrainAPI

api = BrainAPI.load()
result = api.analyze("Your text here")
print(result.summary())
```

### Building the package

```bash
pip install build twine
python -m build
twine check dist/*
```

### Adding a new brain state

1. Create `stimuli/<state_name>.txt` with 3+ representative passages
2. The profiler will auto-detect it on next `BrainAPI.load()`
3. The reference `.npy` profile is regenerated from text → full inference → median activation

### Adding an API endpoint

1. Add Pydantic request/response models in `server.py`
2. Add the route handler using `_ensure_loaded()` for lazy model access
3. Call `_reset_idle_timer()` after inference completes
4. Update `wiki/10_api_reference.md`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRIBE_MODEL_ID` | `facebook/tribev2` | HuggingFace model repo |
| `TRIBE_CACHE_DIR` | `./cache` | Feature cache directory |
| `TRIBE_STIMULI_DIR` | `./stimuli` | Reference state text files |
| `TRIBE_DEVICE` | `auto` | Torch device (auto/mps/cuda/cpu) |
| `TRIBE_HOST` | `0.0.0.0` | Server bind host |
| `TRIBE_PORT` | `8000` | Server bind port |
| `TRIBE_IDLE_TIMEOUT_SECONDS` | `120` | Auto-unload timeout (0 = disabled) |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | Set automatically by `__init__.py` |

## Gotchas and Pitfalls

1. **Never remove the `_mps_compat` import from `__init__.py`** — it must run before any extractor is created or the model will crash on Apple Silicon with "Torch not compiled with CUDA enabled"

2. **`neuralset` and `neuraltrain` are pinned to `0.0.2`** — they are Meta's internal libraries. Do not bump these versions without testing thoroughly. They have hardcoded `device="cuda"` in their HuggingFace configs which our patches override.

3. **Text inference downloads ~10 GB of model weights on first run** — TRIBE v2 checkpoint (~1GB), LLaMA 3.2 3B (~6GB), Wav2Vec2-BERT 2.0 (~2.5GB). All cached in `~/.cache/huggingface/`.

4. **The Destrieux atlas has 148 labels but only ~70 map to our 10 region groups** — unmapped regions (white matter, unknown, etc.) are intentionally excluded. See `brain_states.py` `REGION_GROUPS` for the full mapping.

5. **Scores of exactly 0.5 mean baseline, not zero** — the scoring uses z-score normalization + sigmoid transform. A score of 0.5 means the region's activation is exactly at the population mean.

6. **The server's idle timer uses `asyncio.Task` scheduling** — do not mix threading and asyncio constructs in `server.py`. All async state management must stay within the event loop.

7. **`grids/defaults.py` crashes on import without `DATAPATH` and `SAVEPATH` env vars** — this module is for training only. Never import it at the top level of inference code. The `grids/mac_config.py` is the local development equivalent and does not have this issue.

## Code Style

- **Formatter:** Black (line length 88)
- **Import ordering:** isort with Black-compatible profile
- **Type hints:** Use `from __future__ import annotations` for modern syntax
- **Docstrings:** Google-style or NumPy-style (both are used; be consistent within a file)
- **Logging:** Use `logging.getLogger(__name__)`, not print statements
- **Copyright header:** Every `.py` file starts with the Meta copyright block:
  ```python
  # Copyright (c) Meta Platforms, Inc. and affiliates.
  # All rights reserved.
  #
  # This source code is licensed under the license found in the
  # LICENSE file in the root directory of this source tree.
  ```

## Documentation

The `wiki/` directory contains 11 pages of in-depth documentation:

| Page | Content |
|------|---------|
| `00_index.md` | Wiki index and reading order |
| `01_what_is_tribev2.md` | Project overview, what TRIBE v2 is |
| `02_architecture.md` | Model architecture and data flow |
| `03_apple_silicon.md` | MPS patches, memory management |
| `04_inference_pipeline.md` | Full inference walkthrough |
| `05_brain_atlas.md` | Destrieux atlas, region groups |
| `06_brain_states.md` | Scoring formulas, classification |
| `07_ad_project.md` | Application-specific analysis project |
| `08_file_reference.md` | File-by-file descriptions |
| `09_how_to_extend.md` | Adding states, regions, models |
| `10_api_reference.md` | Python API + HTTP API docs |

**Read the wiki before making significant changes.** Start with `00_index.md`.

## License

CC-BY-NC-4.0 (non-commercial use only). All new code files must include the Meta copyright header. See README.md for full attribution requirements.
