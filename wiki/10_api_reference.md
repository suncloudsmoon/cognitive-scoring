# API Reference

This page documents both the **Python API** (`BrainAPI`) and the **HTTP API** (FastAPI server) for TRIBE v2 brain-state analysis.

---

## HTTP API (Server)

### Starting the Server

```bash
# Install server dependencies
pip install tribev2[server]

# Start the server
python -c "from tribev2.server import main; main()"

# Or with uvicorn directly (more control)
uvicorn tribev2.server:app --host 0.0.0.0 --port 8000
```

The server loads the TRIBE v2 model at startup (~15s), then accepts requests.

### Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `TRIBE_MODEL_ID` | `facebook/tribev2` | HuggingFace repo or local path |
| `TRIBE_CACHE_DIR` | `./cache` | Feature cache directory |
| `TRIBE_STIMULI_DIR` | `./stimuli` | Stimuli directory for classification |
| `TRIBE_DEVICE` | `auto` | PyTorch device (`auto`, `mps`, `cuda`, `cpu`) |
| `TRIBE_HOST` | `0.0.0.0` | Server bind host |
| `TRIBE_PORT` | `8000` | Server port |

### Endpoints

#### `GET /health` — Health Check

Returns server status and readiness.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "available_states": ["happy", "sad", "calm", "curious", "fearful"],
  "region_groups": ["prefrontal", "reward_vmPFC", ...]
}
```

---

#### `GET /regions` — List Brain Region Groups

Returns metadata about all 10 functional region groups, including Destrieux atlas labels.

```bash
curl http://localhost:8000/regions
```

```json
[
  {
    "name": "prefrontal",
    "display_name": "Prefrontal (Executive)",
    "description": "Executive function, planning, decision-making, complex cognition.",
    "destrieux_regions": ["G_front_sup", "G_front_middle", ...]
  },
  ...
]
```

---

#### `POST /analyze` — Analyze Text

The core endpoint. Takes text and returns brain-state scores.

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "She opened the letter and tears of joy streamed down her face."}'
```

**Request body:**

| Field | Type | Required | Default | Description |
|-------|------|:--------:|---------|-------------|
| `text` | string | ✓ | — | Text to analyze |
| `include_time_series` | bool | — | `false` | Include per-timestep scores |
| `include_raw` | bool | — | `false` | Include raw vertex predictions (large!) |

**Response:**

```json
{
  "text": "She opened the letter and tears of joy streamed down her face.",
  "n_timesteps": 8,
  "elapsed_seconds": 4.21,
  "scores": {
    "prefrontal": 0.6234,
    "reward_vmPFC": 0.5521,
    "anterior_cingulate": 0.4812,
    "default_mode": 0.5103,
    "insula": 0.4398,
    "temporal": 0.6312,
    "visual": 0.4201,
    "attention_parietal": 0.5834,
    "motor": 0.3901,
    "fusiform_parahip": 0.5298
  },
  "composites": {
    "valence": 0.0316,
    "learning": 0.5786,
    "attention": 0.6034
  },
  "classification": [
    {"state": "happy", "correlation": 0.8512},
    {"state": "calm", "correlation": 0.3201},
    {"state": "sad", "correlation": -0.1203}
  ],
  "time_series": null,
  "raw_predictions": null
}
```

---

#### `POST /compare` — Compare Two Texts

Runs both texts through the pipeline and returns differential scores.

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text_a": "The child laughed and ran through the sunlit park.",
    "text_b": "The empty house echoed with silence and regret."
  }'
```

**Response:**

```json
{
  "result_a": { "text": "...", "scores": {...}, "composites": {...}, ... },
  "result_b": { "text": "...", "scores": {...}, "composites": {...}, ... },
  "score_diff": {
    "prefrontal": 0.03,
    "reward_vmPFC": 0.08,
    "insula": -0.05,
    ...
  },
  "valence_diff": 0.12,
  "learning_diff": 0.02,
  "attention_diff": -0.01
}
```

### Interactive Documentation

The server provides auto-generated interactive docs:

- **Swagger UI**: `http://localhost:8000/docs` — try endpoints directly in the browser
- **ReDoc**: `http://localhost:8000/redoc` — clean API reference
- **OpenAPI JSON**: `http://localhost:8000/openapi.json` — for code generators

### Concurrency

The server serializes all inference requests using an `asyncio.Lock`. This is because the TRIBE v2 model uses GPU/MPS resources that cannot safely handle concurrent access. Requests that arrive while another is processing will queue and wait.

### Performance

| Step | First Run | Cached |
|------|-----------|--------|
| Server startup | ~15s | ~15s |
| `POST /analyze` | ~3–5 min | ~5–10s |
| `POST /compare` | ~6–10 min | ~10–20s |

---

## Python API

### Quick Start

```python
from tribev2 import BrainAPI

api = BrainAPI.load()
result = api.analyze("She opened the letter and tears of joy streamed down her face.")

print(result.valence)         # +0.031
print(result.learning)        # 0.612
print(result.attention)       # 0.558
print(result.scores)          # {"prefrontal": 0.62, ...}
print(result.classification)  # [("happy", 0.85), ...]
print(result.summary())       # Full formatted output
```

### `BrainAPI.load()`

```python
api = BrainAPI.load(
    model_id="facebook/tribev2",     # HuggingFace repo or local path
    cache_dir="./cache",              # Feature cache directory
    stimuli_dir="./stimuli",          # Reference stimuli for classification
    build_profiles=False,             # True = build profiles immediately
    max_profile_passages=3,           # Passages per state file
    config_update=None,               # Config overrides (dict)
    device="auto",                    # "auto", "mps", "cuda", or "cpu"
)
```

### `api.analyze(text)`

```python
result = api.analyze(
    text,                          # The text to analyze (required)
    include_raw=False,             # Include raw (n_timesteps, 20484) array?
    include_time_series=True,      # Include per-timestep scores?
)
```

### `api.analyze_file(path)`

```python
result = api.analyze_file("./my_text.txt")
```

### `api.compare(text_a, text_b)`

```python
diff = api.compare("Happy text...", "Sad text...")
print(diff["valence_diff"])     # 0.12
print(diff["score_diff"])       # {"prefrontal": 0.03, ...}
```

### `BrainResult` Fields

| Field | Type | Always Present | Description |
|-------|------|:-:|---|
| `text` | `str` | ✓ | The input text |
| `scores` | `dict[str, float]` | ✓ | Per-region scores (0–1, 0.5 = baseline) |
| `valence` | `float` | ✓ | Emotional valence (positive = happy) |
| `learning` | `float` | ✓ | Learning readiness |
| `attention` | `float` | ✓ | Attention engagement |
| `classification` | `list[tuple]` | ⚪ | Ranked state matches. `None` if no profiles. |
| `time_series` | `dict` | ⚪ | Per-timestep scores |
| `raw_predictions` | `np.ndarray` | ⚪ | Full `(n_timesteps, 20484)` array |
| `n_timesteps` | `int` | ✓ | Number of 1-second time windows |

### `BrainResult` Methods

```python
result.top_regions(3)    # [('temporal', 0.63), ('prefrontal', 0.58), ...]
result.to_dict()         # JSON-serializable dict
result.summary()         # Human-readable formatted string
```

### Understanding Scores

| Group | Brain Regions | What It Means |
|-------|---------------|---------------|
| `prefrontal` | Superior/middle/inferior frontal | Executive function, planning |
| `reward_vmPFC` | Orbital gyri, gyrus rectus | Reward processing, positive affect |
| `anterior_cingulate` | Anterior + mid-anterior cingulate | Conflict monitoring, curiosity |
| `default_mode` | Posterior cingulate, precuneus | Self-referential thought |
| `insula` | Insular gyri | Emotional awareness, negative affect |
| `temporal` | Superior/middle/inferior temporal | Language, social cognition |
| `visual` | Cuneus, occipital gyri | Visual processing |
| `attention_parietal` | Superior parietal, angular | Spatial attention |
| `motor` | Pre/postcentral | Sensorimotor processing |
| `fusiform_parahip` | Fusiform, parahippocampal | Memory, face/object recognition |

### Composite Score Formulas

| Score | Formula | Interpretation |
|-------|---------|----------------|
| **Valence** | `reward_vmPFC − (insula + ACC) / 2` | Positive = happy; negative = sad |
| **Learning** | `(prefrontal + ACC + temporal) / 3` | Higher = deeper processing |
| **Attention** | `(attention_parietal + prefrontal) / 2` | Higher = more focused |

---

## Architecture

```
BrainAPI / HTTP Server
    ├── TribeModel           ← Model loading, events, prediction
    ├── BrainAtlas           ← Destrieux region mapping
    ├── compute_normalized_scores()  ← Z-score + sigmoid scoring
    └── BrainStateClassifier ← Pearson correlation classification
```
