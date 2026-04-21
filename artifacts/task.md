# Text-Only TribeV2 Optimization Tasks

## Tier 1: Quick Wins
- [x] 1A. Load Llama in float16 (`_mps_compat.py`) — saves ~6.4 GB
- [x] 1B. Switch WhisperX to `large-v3-turbo` (`eventstransforms.py`)
- [x] 1C. Reduce WhisperX batch size to 4 (`eventstransforms.py`)

## Tier 2: Targeted Code Changes
- [x] 2B. Aggressive memory cleanup — free tokenizer too (`main.py`)
- [x] 2C. Brain model in float16 (`demo_utils.py` + `model.py`) — saves ~338 MB

## Tier 3: Text-Only Mode
- [x] 3A. Text-only config via `config_update` (documented, works)

## Tier 4: Quantization
- [x] 4A. INT8 tested — produces NaN with fp16 activations (Llama hidden states ±300 overflow). Disabled. Infrastructure left in place for future torchao improvements.

## Infrastructure
- [x] Add torchao to optional dependencies (`pyproject.toml`)

## Bug Fixes Found During Implementation
- [x] Fix device timing: `_get_data` reads `self.device` before `_load_model` runs
- [x] Fix pydantic frozen model: use `object.__setattr__` for late device override
- [x] Fix HuggingFaceText kept on "cpu" in model_post_init (not just _load_model)

## Verification
- [x] E2E test: text-only inference completes with valid predictions ✅
