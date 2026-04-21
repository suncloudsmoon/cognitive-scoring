# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI HTTP server for TRIBE v2 brain-state analysis.

Run directly::

    python -m tribev2.server

Or with uvicorn for more control::

    uvicorn tribev2.server:app --host 0.0.0.0 --port 8000

Environment variables for configuration::

    TRIBE_MODEL_ID       Model repo (default: "facebook/tribev2")
    TRIBE_CACHE_DIR      Cache directory (default: "./cache")
    TRIBE_STIMULI_DIR    Stimuli directory (default: "./stimuli")
    TRIBE_DEVICE         Torch device (default: "auto")
    TRIBE_HOST           Server host (default: "0.0.0.0")
    TRIBE_PORT           Server port (default: 8000)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("tribev2.server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("TRIBE_MODEL_ID", "facebook/tribev2")
CACHE_DIR = os.environ.get("TRIBE_CACHE_DIR", "./cache")
STIMULI_DIR = os.environ.get("TRIBE_STIMULI_DIR", "./stimuli")
DEVICE = os.environ.get("TRIBE_DEVICE", "auto")

# ---------------------------------------------------------------------------
# Pydantic models for request / response validation
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Request body for ``POST /analyze``."""

    text: str = Field(
        ...,
        min_length=1,
        description="The text to analyze. Can be any length, but very short texts (<5 words) may produce unreliable predictions.",
        examples=["She opened the letter and tears of joy streamed down her face."],
    )
    include_time_series: bool = Field(
        default=False,
        description="If true, include per-timestep activation scores for each region group.",
    )
    include_raw: bool = Field(
        default=False,
        description="If true, include the raw vertex-level predictions as a nested array of shape (n_timesteps, 20484). Warning: this produces very large responses.",
    )


class CompareRequest(BaseModel):
    """Request body for ``POST /compare``."""

    text_a: str = Field(
        ...,
        min_length=1,
        description='First text (treated as the "positive" direction in the differential).',
        examples=["The child laughed and ran through the sunlit park."],
    )
    text_b: str = Field(
        ...,
        min_length=1,
        description='Second text (treated as the "negative" direction in the differential).',
        examples=["The empty house echoed with silence and regret."],
    )


class ClassificationMatch(BaseModel):
    """A single classification match against a reference profile."""

    state: str = Field(description="The emotional/cognitive state name (e.g., 'happy', 'sad').")
    correlation: float = Field(description="Pearson correlation coefficient (-1 to +1). Higher = stronger match.")


class CompositeScores(BaseModel):
    """Composite cognitive/emotional scores derived from region activations."""

    valence: float = Field(description="Emotional valence. Positive = happy/positive affect, negative = sad/negative affect. Formula: reward_vmPFC - (insula + anterior_cingulate) / 2.")
    learning: float = Field(description="Learning readiness. Higher = deeper cognitive/semantic processing. Formula: (prefrontal + anterior_cingulate + temporal) / 3.")
    attention: float = Field(description="Attention engagement. Higher = more focused attention. Formula: (attention_parietal + prefrontal) / 2.")


class AnalyzeResponse(BaseModel):
    """Response body for ``POST /analyze``."""

    text: str = Field(description="The input text that was analyzed.")
    n_timesteps: int = Field(description="Number of 1-second time windows in the prediction.")
    elapsed_seconds: float = Field(description="Wall-clock time for the analysis in seconds.")

    scores: dict[str, float] = Field(
        description="Normalized activation score (0-1) per functional region group. 0.5 = baseline; >0.5 = above-average engagement; <0.5 = below-average.",
    )
    composites: CompositeScores = Field(description="Composite cognitive/emotional scores.")

    classification: list[ClassificationMatch] | None = Field(
        default=None,
        description="Ranked state matches with Pearson correlations. None if no reference profiles are loaded.",
    )
    time_series: dict[str, list[float]] | None = Field(
        default=None,
        description="Per-timestep activation score for each region group. Only present if include_time_series=true.",
    )
    raw_predictions: list[list[float]] | None = Field(
        default=None,
        description="Raw vertex-level predictions, shape (n_timesteps, 20484). Only present if include_raw=true. Warning: very large.",
    )


class CompareResponse(BaseModel):
    """Response body for ``POST /compare``."""

    result_a: AnalyzeResponse = Field(description="Full analysis result for text_a.")
    result_b: AnalyzeResponse = Field(description="Full analysis result for text_b.")
    score_diff: dict[str, float] = Field(description="Per-region score difference (a - b). Positive = text_a scored higher.")
    valence_diff: float = Field(description="Valence difference (a - b). Positive = text_a is 'happier'.")
    learning_diff: float = Field(description="Learning readiness difference (a - b).")
    attention_diff: float = Field(description="Attention engagement difference (a - b).")


class RegionInfo(BaseModel):
    """Information about a functional brain region group."""

    name: str = Field(description="Internal region group name used in scores.")
    display_name: str = Field(description="Human-readable display name.")
    description: str = Field(description="Brief functional description.")
    destrieux_regions: list[str] = Field(description="Constituent Destrieux atlas region labels.")


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str = Field(description="Server status: 'ok' (model loaded), 'idle' (model unloaded, will load on next request), or 'loading' (model currently loading).")
    model_loaded: bool = Field(description="Whether the TRIBE v2 model is loaded and ready.")
    available_states: list[str] = Field(description="List of available classification state names.")
    region_groups: list[str] = Field(description="List of functional region group names.")


# ---------------------------------------------------------------------------
# Region metadata (for /regions endpoint)
# ---------------------------------------------------------------------------

_REGION_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "prefrontal": ("Prefrontal (Executive)", "Executive function, planning, decision-making, complex cognition."),
    "reward_vmPFC": ("Reward / vmPFC", "Reward processing, positive valence, valuation, positive affect."),
    "anterior_cingulate": ("Anterior Cingulate", "Conflict monitoring, curiosity, uncertainty, emotion regulation."),
    "default_mode": ("Default Mode Network", "Self-referential thought, mind-wandering, introspection."),
    "insula": ("Insula (Emotion)", "Interoception, emotional awareness, negative affect, empathy."),
    "temporal": ("Temporal (Language)", "Language comprehension, auditory processing, social cognition."),
    "visual": ("Visual Cortex", "Visual processing, scene recognition, object perception."),
    "attention_parietal": ("Attention (Parietal)", "Spatial attention, dorsal attention network, focused attention."),
    "motor": ("Motor Cortex", "Sensorimotor processing, action planning, body movement."),
    "fusiform_parahip": ("Memory (Fusiform)", "Memory encoding, face/object recognition, scene memory."),
}


# ---------------------------------------------------------------------------
# Lifecycle manager — lazy load + auto-unload
# ---------------------------------------------------------------------------

# Idle timeout before the model is unloaded to free memory.
# Configurable via TRIBE_IDLE_TIMEOUT_SECONDS (default: 120 = 2 minutes).
IDLE_TIMEOUT = int(os.environ.get("TRIBE_IDLE_TIMEOUT_SECONDS", "120"))

# The BrainAPI instance — starts as None, loaded on first request.
_brain_api = None

# Serialize all inference requests — the model uses GPU/MPS resources
# that cannot safely handle concurrent access.
_inference_lock = asyncio.Lock()

# Tracks the scheduled unload task so we can cancel/reset it.
_unload_task: asyncio.Task | None = None

# Tracks whether a load is currently in progress (to avoid duplicate loads).
_loading = False


async def _unload_after_idle():
    """Wait for the idle timeout, then unload the model to free memory."""
    global _brain_api
    await asyncio.sleep(IDLE_TIMEOUT)

    # Don't unload if a request is currently being processed
    if _inference_lock.locked():
        logger.info("Unload deferred — inference in progress")
        return

    if _brain_api is not None:
        logger.info("Idle for %ds — unloading model to free memory", IDLE_TIMEOUT)
        _brain_api.unload()
        _brain_api = None
        logger.info("Model unloaded — will reload on next request")


def _reset_idle_timer():
    """Reset (or start) the idle unload timer. Call after every request."""
    global _unload_task
    if _unload_task is not None:
        _unload_task.cancel()
    _unload_task = asyncio.get_event_loop().create_task(_unload_after_idle())


async def _ensure_loaded():
    """Ensure the model is loaded. Loads lazily on first call.

    Returns the BrainAPI instance. If the model was unloaded due to
    inactivity, it will be reloaded here (~15s).
    """
    global _brain_api, _loading

    if _brain_api is not None:
        return _brain_api

    if _loading:
        # Another request is already loading the model — wait for it
        while _loading:
            await asyncio.sleep(0.5)
        if _brain_api is not None:
            return _brain_api

    _loading = True
    try:
        from tribev2.api import BrainAPI

        logger.info("Loading TRIBE v2 model (lazy)...")
        t0 = time.time()

        loop = asyncio.get_event_loop()
        _brain_api = await loop.run_in_executor(
            None,
            lambda: BrainAPI.load(
                model_id=MODEL_ID,
                cache_dir=CACHE_DIR,
                stimuli_dir=STIMULI_DIR,
                device=DEVICE,
            ),
        )

        elapsed = time.time() - t0
        logger.info("Model loaded in %.1fs", elapsed)
        logger.info(
            "Available states: %s",
            _brain_api.available_states or "(none)",
        )
        return _brain_api
    finally:
        _loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Server lifecycle — model loads lazily, not here."""
    logger.info("=" * 60)
    logger.info("  TRIBE v2 Brain State API — Starting up")
    logger.info("=" * 60)
    logger.info("  Model:        %s", MODEL_ID)
    logger.info("  Cache:        %s", CACHE_DIR)
    logger.info("  Stimuli:      %s", STIMULI_DIR)
    logger.info("  Device:       %s", DEVICE)
    logger.info("  Idle timeout: %ds", IDLE_TIMEOUT)
    logger.info("  Loading:      lazy (on first request)")
    logger.info("=" * 60)
    logger.info("Server ready — model will load on first API call")

    yield

    global _brain_api, _unload_task
    if _unload_task is not None:
        _unload_task.cancel()
    logger.info("Shutting down TRIBE v2 server...")
    if _brain_api is not None:
        _brain_api.unload()
        _brain_api = None


def _build_analyze_response(result: Any, elapsed: float) -> AnalyzeResponse:
    """Convert a BrainResult to an AnalyzeResponse."""
    classification = None
    if result.classification is not None:
        classification = [
            ClassificationMatch(state=s, correlation=round(c, 4))
            for s, c in result.classification
        ]

    time_series = None
    if result.time_series is not None:
        time_series = {
            k: [round(v, 4) for v in vs]
            for k, vs in result.time_series.items()
        }

    raw_preds = None
    if result.raw_predictions is not None:
        raw_preds = result.raw_predictions.tolist()

    return AnalyzeResponse(
        text=result.text,
        n_timesteps=result.n_timesteps,
        elapsed_seconds=round(elapsed, 2),
        scores={k: round(v, 4) for k, v in result.scores.items()},
        composites=CompositeScores(
            valence=round(result.valence, 4),
            learning=round(result.learning, 4),
            attention=round(result.attention, 4),
        ),
        classification=classification,
        time_series=time_series,
        raw_predictions=raw_preds,
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


app = FastAPI(
    title="TRIBE v2 Brain State API",
    version="1.0.0",
    description="""
Predict brain activity from text using Meta's TRIBE v2 model.

TRIBE v2 takes text as input and predicts what a human brain's fMRI activity
would look like. This API maps those predictions to named brain regions,
computes cognitive/emotional scores, and classifies the input against
reference emotional states.

## Quick Start

```bash
curl -X POST http://localhost:8000/analyze \\
  -H "Content-Type: application/json" \\
  -d '{"text": "She opened the letter and tears of joy streamed down her face."}'
```

## Key Concepts

- **Scores** (0–1): Per-region activation. 0.5 = baseline, >0.5 = above average.
- **Valence**: Positive = happy, negative = sad.
- **Classification**: Pearson correlation against reference brain profiles.

## Caveats

- First inference per text takes ~3–5 min (feature extraction). Cached runs: ~5–10s.
- Cortical surface only — subcortical structures (amygdala, hippocampus) are NOT represented.
- NOT suitable for clinical use.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow any origin for development / local use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def root():
    """Landing page with links to docs."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TRIBE v2 Brain State API</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
                color: #e0e0e0;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .container {
                max-width: 640px;
                padding: 3rem;
                text-align: center;
            }
            .logo {
                font-size: 3rem;
                margin-bottom: 0.5rem;
            }
            h1 {
                font-size: 2rem;
                font-weight: 700;
                background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
            }
            .subtitle {
                color: #9ca3af;
                font-size: 1.05rem;
                margin-bottom: 2.5rem;
                line-height: 1.6;
            }
            .links {
                display: flex;
                gap: 1rem;
                justify-content: center;
                flex-wrap: wrap;
            }
            a.btn {
                display: inline-block;
                padding: 0.75rem 1.75rem;
                border-radius: 0.5rem;
                text-decoration: none;
                font-weight: 600;
                font-size: 0.95rem;
                transition: all 0.2s;
            }
            a.btn.primary {
                background: linear-gradient(135deg, #7c3aed, #3b82f6);
                color: #fff;
            }
            a.btn.primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 24px rgba(124, 58, 237, 0.35);
            }
            a.btn.secondary {
                background: rgba(255,255,255,0.08);
                color: #c4b5fd;
                border: 1px solid rgba(255,255,255,0.12);
            }
            a.btn.secondary:hover {
                background: rgba(255,255,255,0.14);
                transform: translateY(-2px);
            }
            .example {
                margin-top: 2.5rem;
                text-align: left;
                background: rgba(0,0,0,0.3);
                border-radius: 0.75rem;
                padding: 1.25rem;
                border: 1px solid rgba(255,255,255,0.06);
            }
            .example h3 {
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #9ca3af;
                margin-bottom: 0.75rem;
            }
            pre {
                font-family: 'JetBrains Mono', 'Fira Code', monospace;
                font-size: 0.82rem;
                line-height: 1.6;
                color: #a5b4fc;
                overflow-x: auto;
                white-space: pre-wrap;
                word-break: break-all;
            }
            .footer {
                margin-top: 2.5rem;
                font-size: 0.8rem;
                color: #6b7280;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">🧠</div>
            <h1>TRIBE v2 Brain State API</h1>
            <p class="subtitle">
                Predict brain activity from text using Meta's TRIBE v2 model.<br>
                Map predictions to cognitive regions, emotional states, and more.
            </p>
            <div class="links">
                <a href="/docs" class="btn primary">Interactive Docs</a>
                <a href="/redoc" class="btn secondary">API Reference</a>
                <a href="/health" class="btn secondary">Health Check</a>
            </div>
            <div class="example">
                <h3>Quick Start</h3>
                <pre>curl -X POST http://localhost:8000/analyze \\
  -H "Content-Type: application/json" \\
  -d '{"text": "She opened the letter and tears of joy streamed down her face."}'</pre>
            </div>
            <p class="footer">Meta TRIBE v2 &middot; Cortical surface predictions only &middot; Not for clinical use</p>
        </div>
    </body>
    </html>
    """


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the server status, whether the model is loaded, and available classification states.",
    tags=["Status"],
)
async def health():
    """Return server health and readiness status."""
    if _loading:
        return HealthResponse(
            status="loading",
            model_loaded=False,
            available_states=[],
            region_groups=[],
        )
    if _brain_api is None:
        return HealthResponse(
            status="idle",
            model_loaded=False,
            available_states=[],
            region_groups=[],
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        available_states=_brain_api.available_states,
        region_groups=_brain_api.region_groups,
    )


@app.get(
    "/regions",
    response_model=list[RegionInfo],
    summary="List brain region groups",
    description="Returns all functional brain region groups with their descriptions and constituent Destrieux atlas labels.",
    tags=["Reference"],
)
async def list_regions():
    """Return metadata about all functional region groups."""
    from tribev2.brain_states import REGION_GROUPS

    result = []
    for name, regions in REGION_GROUPS.items():
        display, desc = _REGION_DESCRIPTIONS.get(name, (name, ""))
        result.append(RegionInfo(
            name=name,
            display_name=display,
            description=desc,
            destrieux_regions=regions,
        ))
    return result


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze text",
    description=(
        "Run the full TRIBE v2 pipeline on a text string: "
        "text → speech → transcription → feature extraction → "
        "Transformer inference → atlas mapping → scoring → classification.\n\n"
        "**First run per text takes ~3–5 minutes** (feature extraction). "
        "Subsequent runs with the same text return in ~5–10 seconds (cached)."
    ),
    tags=["Analysis"],
)
async def analyze(request: AnalyzeRequest):
    """Analyze a text and return brain-state scores."""
    api = await _ensure_loaded()

    async with _inference_lock:
        loop = asyncio.get_event_loop()
        t0 = time.time()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: api.analyze(
                    request.text,
                    include_raw=request.include_raw,
                    include_time_series=request.include_time_series,
                ),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Analysis failed")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
        elapsed = time.time() - t0

    _reset_idle_timer()
    return _build_analyze_response(result, elapsed)


@app.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare two texts",
    description=(
        "Run TRIBE v2 on two texts and return a differential analysis "
        "showing which brain regions respond differently.\n\n"
        "**Takes ~2× the time of a single analysis** since both texts "
        "must be processed through the full pipeline."
    ),
    tags=["Analysis"],
)
async def compare(request: CompareRequest):
    """Compare brain-state responses between two texts."""
    api = await _ensure_loaded()

    async with _inference_lock:
        loop = asyncio.get_event_loop()
        t0 = time.time()
        try:
            result_a = await loop.run_in_executor(
                None,
                lambda: api.analyze(request.text_a),
            )
            mid = time.time()
            result_b = await loop.run_in_executor(
                None,
                lambda: api.analyze(request.text_b),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Comparison failed")
            raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")
        elapsed_total = time.time() - t0

    _reset_idle_timer()

    resp_a = _build_analyze_response(result_a, mid - t0)
    resp_b = _build_analyze_response(result_b, time.time() - mid)

    score_diff = {
        group: round(result_a.scores[group] - result_b.scores[group], 4)
        for group in result_a.scores
    }

    return CompareResponse(
        result_a=resp_a,
        result_b=resp_b,
        score_diff=score_diff,
        valence_diff=round(result_a.valence - result_b.valence, 4),
        learning_diff=round(result_a.learning - result_b.learning, 4),
        attention_diff=round(result_a.attention - result_b.attention, 4),
    )


# ---------------------------------------------------------------------------
# CLI entry point: python -m tribev2.server
# ---------------------------------------------------------------------------


def main():
    """Run the TRIBE v2 API server."""
    import uvicorn

    host = os.environ.get("TRIBE_HOST", "0.0.0.0")
    port = int(os.environ.get("TRIBE_PORT", "8000"))

    uvicorn.run(
        "tribev2.server:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
