# BrainAPI — A Simple, Comprehensive API for TRIBE v2

## Goal

Collapse the current 6-step workflow down to **two lines**:

```python
from tribev2 import BrainAPI

api = BrainAPI.load()
result = api.analyze("She opened the letter and tears of joy streamed down her face.")

print(result.valence)         # +0.031
print(result.attention)       # 0.558
print(result.learning)        # 0.612
print(result.scores)          # {"prefrontal": 0.62, "reward_vmPFC": 0.55, ...}
print(result.classification)  # [("happy", 0.85), ("calm", 0.32), ...]
```

The current workflow requires users to manually: load model, create BrainAtlas, write text to a temp file, call `get_events_dataframe()`, call `predict()`, then call `compute_normalized_scores()` and optionally build profiles/classify. This is 15+ lines of code for the most common use case.

## Proposed Changes

### Core Module

---

#### [NEW] [api.py](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/api.py)

This is the entire API in a single, well-documented file. It contains two classes:

##### `BrainResult` (dataclass)

A structured result object returned by every `analyze()` call. Clean `__repr__` for nice printing.

| Field | Type | Always Present | Description |
|-------|------|:-:|---|
| `text` | `str` | ✓ | The input text that was analyzed |
| `scores` | `dict[str, float]` | ✓ | Per-region-group activation scores (0–1 scale, 0.5 = baseline) |
| `valence` | `float` | ✓ | Emotional valence (positive = happy, negative = sad) |
| `learning` | `float` | ✓ | Learning readiness composite score |
| `attention` | `float` | ✓ | Attention engagement composite score |
| `classification` | `list[tuple[str, float]]` or `None` | Only if profiles loaded | Ranked state matches with Pearson correlations |
| `time_series` | `dict[str, list[float]]` or `None` | Only if `include_time_series=True` | Per-timestep activation per region group |
| `raw_predictions` | `np.ndarray` or `None` | Only if `include_raw=True` | Raw `(n_timesteps, 20484)` vertex predictions |
| `n_timesteps` | `int` | ✓ | Number of 1-second time windows in the prediction |

Helper methods on `BrainResult`:
- `to_dict()` → JSON-serializable dictionary
- `top_regions(n=3)` → list of top-n most activated region groups
- `summary()` → human-readable multi-line string summary
- `__repr__()` → clean one-line representation

##### `BrainAPI` (main class)

| Method | Signature | Description |
|--------|-----------|-------------|
| `load()` | `@classmethod load(model_id="facebook/tribev2", ...) → BrainAPI` | Load model, atlas, and optionally pre-build reference profiles |
| `analyze()` | `analyze(text, include_raw=False, include_time_series=True) → BrainResult` | **The main method.** Takes a string, returns everything |
| `analyze_file()` | `analyze_file(path, include_raw=False, ...) → BrainResult` | Analyze text from a file path |
| `compare()` | `compare(text_a, text_b) → dict` | Compare two texts and return differential scores |

Key design choices:

1. **`analyze()` accepts raw text strings directly** — no need to write temp files. The API handles the temp file lifecycle internally.

2. **Reference profiles are lazy-loaded** — If a `stimuli/` directory exists annotated with state `.txt` files, profiles are automatically discovered and built on first `analyze()` call that would need them. The user can also explicitly call `load(build_profiles=True)` to pre-build them during initialization.

3. **All raw data is opt-in** — By default, only the high-level scores are returned. Set `include_raw=True` to get the full `(n_timesteps, 20484)` numpy array. This keeps the default experience clean.

4. **Thread-safe temp file handling** — Uses `tempfile.NamedTemporaryFile` with proper cleanup.

---

#### [MODIFY] [\_\_init\_\_.py](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/__init__.py)

Add `BrainAPI` and `BrainResult` to the public exports:

```diff
 from tribev2.demo_utils import TribeModel
+from tribev2.api import BrainAPI, BrainResult

-__all__ = ["TribeModel"]
+__all__ = ["TribeModel", "BrainAPI", "BrainResult"]
```

### Documentation

---

#### [NEW] [10_api_reference.md](file:///Volumes/dev/ai_work/meta/tribe_v2/code/wiki/10_api_reference.md)

A new wiki page documenting:
- Quick-start (2-line example)
- `BrainAPI.load()` parameters and behavior
- `BrainResult` fields and methods
- `analyze()`, `analyze_file()`, and `compare()` with full examples
- Performance notes (first run vs cached)
- Relationship to the lower-level classes (`TribeModel`, `BrainAtlas`, etc.)

## Open Questions

> [!IMPORTANT]
> **Classification by default?** If `stimuli/` exists with state files, should `analyze()` automatically include classification results? My current plan is **yes** — auto-discover and lazy-build profiles on first call. This adds ~10-15 minutes the first time but all subsequent calls are instant (cached `.npy` files). If you'd prefer classification to be fully opt-in, I can add a `classify=False` parameter instead.

> [!NOTE]
> **`compare()` method scope** — The `compare(text_a, text_b)` method would run both texts through the pipeline and return a differential analysis (which brain regions differ the most). This requires two full inference passes. Should I include this in v1, or defer it?

## Verification Plan

### Automated Tests
- Import test: `from tribev2 import BrainAPI, BrainResult` succeeds
- Unit test: `BrainResult` construction, `to_dict()`, `summary()`, `top_regions()`
- Smoke test: `BrainAPI.load()` initializes without error (needs model weights)

### Manual Verification
- Run `api.analyze("test text")` in a Python REPL and verify output structure
- Confirm cached runs return identical results
- Verify `include_raw=True` returns the numpy array
