# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Simple, comprehensive API for TRIBE v2 brain-state analysis.

This module provides a two-line interface to the full TRIBE v2 pipeline::

    from tribev2 import BrainAPI

    api = BrainAPI.load()
    result = api.analyze("She opened the letter and tears of joy streamed down her face.")

    print(result.valence)         # +0.031  (positive = happy)
    print(result.scores)          # {"prefrontal": 0.62, "reward_vmPFC": 0.55, ...}
    print(result.classification)  # [("happy", 0.85), ("calm", 0.32), ...]

Under the hood, ``BrainAPI`` orchestrates:

1. Text → speech (gTTS) → transcription (WhisperX) → events DataFrame
2. Feature extraction (V-JEPA, Wav2Vec, LLaMA 3.2)
3. Fusion Transformer → 20,484-vertex brain predictions
4. Destrieux atlas mapping → normalized region-group scores
5. (Optional) Reference-profile classification via Pearson correlation

All intermediate artifacts (temp files, features) are cached automatically.
Subsequent calls with the same text return in seconds.

**Important caveats** (inherited from TRIBE v2):

- Predictions are cortical-surface only; subcortical structures (amygdala,
  hippocampus, basal ganglia) are NOT represented.
- All cognitive/emotional labels are approximations based on cortical correlates.
- This is NOT suitable for clinical diagnosis or treatment decisions.

See Also
--------
wiki/10_api_reference.md : Full API documentation with examples.
tribev2.brain_states : Lower-level scoring, profiling, and classification.
tribev2.demo_utils.TribeModel : The model wrapper used internally.
"""

from __future__ import annotations

import logging
import tempfile
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BrainResult — structured output from analyze()
# ---------------------------------------------------------------------------


@dataclass
class BrainResult:
    """Structured result from a TRIBE v2 brain-state analysis.

    Every ``BrainAPI.analyze()`` call returns one of these.  The most useful
    fields are always present; heavier data (raw predictions, time series)
    is opt-in.

    Attributes
    ----------
    text : str
        The input text that was analyzed.
    scores : dict[str, float]
        Normalized activation score (0–1) per functional region group.
        A score of 0.5 means baseline (average) activation; >0.5 means
        the region is more active than the brain-wide average.

        Groups: ``prefrontal``, ``reward_vmPFC``, ``anterior_cingulate``,
        ``default_mode``, ``insula``, ``temporal``, ``visual``,
        ``attention_parietal``, ``motor``, ``fusiform_parahip``.
    valence : float
        Emotional valence composite.  Positive values suggest happier /
        more positive affect; negative values suggest sadder / more
        negative affect.  Computed as:
        ``reward_vmPFC - (insula + anterior_cingulate) / 2``.
    learning : float
        Learning-readiness composite.  Higher values indicate deeper
        cognitive / semantic processing.  Computed as:
        ``(prefrontal + anterior_cingulate + temporal) / 3``.
    attention : float
        Attention-engagement composite.  Higher values indicate stronger
        focused attention.  Computed as:
        ``(attention_parietal + prefrontal) / 2``.
    classification : list[tuple[str, float]] or None
        Ranked state matches with Pearson correlation coefficients,
        e.g. ``[("happy", 0.85), ("sad", -0.12)]``.  ``None`` if no
        reference profiles were loaded.
    time_series : dict[str, list[float]] or None
        Per-timestep activation score for each region group.  ``None``
        unless ``include_time_series=True`` was passed to ``analyze()``.
    raw_predictions : numpy.ndarray or None
        The full ``(n_timesteps, 20484)`` vertex-level predictions.
        ``None`` unless ``include_raw=True`` was passed to ``analyze()``.
    n_timesteps : int
        Number of 1-second time windows in the prediction.

    Examples
    --------
    >>> result = api.analyze("The sunset painted the sky in gold.")
    >>> result.valence
    0.018
    >>> result.top_regions(3)
    [('visual', 0.71), ('temporal', 0.63), ('prefrontal', 0.58)]
    >>> print(result.summary())
    Brain State Analysis
    ════════════════════
    Input: "The sunset painted the sky in gold."
    Timesteps: 8
    ...
    """

    text: str
    scores: dict[str, float]
    valence: float
    learning: float
    attention: float
    classification: list[tuple[str, float]] | None = None
    time_series: dict[str, list[float]] | None = None
    raw_predictions: np.ndarray | None = field(default=None, repr=False)
    n_timesteps: int = 0

    # ── Helper methods ────────────────────────────────────────────────

    def top_regions(self, n: int = 3) -> list[tuple[str, float]]:
        """Return the *n* most activated region groups, sorted descending.

        Parameters
        ----------
        n : int
            Number of top regions to return (default 3).

        Returns
        -------
        list[tuple[str, float]]
            ``(group_name, score)`` pairs sorted by score descending.

        Examples
        --------
        >>> result.top_regions(3)
        [('temporal', 0.63), ('prefrontal', 0.58), ('reward_vmPFC', 0.55)]
        """
        ranked = sorted(self.scores.items(), key=lambda kv: -kv[1])
        return ranked[:n]

    def to_dict(self) -> dict[str, tp.Any]:
        """Convert to a JSON-serializable dictionary.

        Raw predictions (numpy array) are excluded from the output.
        Use ``raw_predictions`` directly if you need the array.

        Returns
        -------
        dict
            All fields except ``raw_predictions``, with numpy types
            converted to native Python types.
        """
        d: dict[str, tp.Any] = {
            "text": self.text,
            "scores": dict(self.scores),
            "valence": float(self.valence),
            "learning": float(self.learning),
            "attention": float(self.attention),
            "n_timesteps": self.n_timesteps,
        }
        if self.classification is not None:
            d["classification"] = [
                {"state": s, "correlation": float(c)}
                for s, c in self.classification
            ]
        if self.time_series is not None:
            d["time_series"] = {
                k: [float(v) for v in vs]
                for k, vs in self.time_series.items()
            }
        return d

    def summary(self) -> str:
        """Return a human-readable multi-line summary.

        Returns
        -------
        str
            Formatted summary including scores, composites, and
            classification (if available).

        Examples
        --------
        >>> print(result.summary())
        Brain State Analysis
        ════════════════════
        Input: "She opened the letter and tears of joy..."
        ...
        """
        # Truncate long text for display
        display_text = self.text
        if len(display_text) > 80:
            display_text = display_text[:77] + "..."

        lines = [
            "Brain State Analysis",
            "════════════════════",
            f'Input: "{display_text}"',
            f"Timesteps: {self.n_timesteps}",
            "",
            "Region Scores (0–1, 0.5 = baseline):",
        ]

        # Display-friendly names
        display_names = {
            "prefrontal": "Prefrontal (Executive)",
            "reward_vmPFC": "Reward / vmPFC",
            "anterior_cingulate": "Anterior Cingulate",
            "default_mode": "Default Mode Network",
            "insula": "Insula (Emotion)",
            "temporal": "Temporal (Language)",
            "visual": "Visual Cortex",
            "attention_parietal": "Attention (Parietal)",
            "motor": "Motor Cortex",
            "fusiform_parahip": "Memory (Fusiform)",
        }

        for group, score in self.scores.items():
            name = display_names.get(group, group)
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"  {name:<26} {score:>6.1%}  {bar}")

        lines.extend([
            "",
            "Composite Scores:",
            f"  Emotional Valence:   {self.valence:+.3f}  (positive = happy)",
            f"  Learning Readiness:  {self.learning:.3f}   (higher = deeper processing)",
            f"  Attention Level:     {self.attention:.3f}   (higher = more focused)",
        ])

        if self.classification:
            lines.extend(["", "Classification (Pearson r):"])
            for state, corr in self.classification:
                lines.append(f"  {state:<20} r = {corr:+.3f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Clean one-line representation."""
        text_preview = self.text[:40] + "..." if len(self.text) > 40 else self.text
        cls_str = ""
        if self.classification:
            top = self.classification[0]
            cls_str = f", top_match='{top[0]}' (r={top[1]:.2f})"
        return (
            f"BrainResult("
            f'text="{text_preview}", '
            f"valence={self.valence:+.3f}, "
            f"learning={self.learning:.3f}, "
            f"attention={self.attention:.3f}"
            f"{cls_str})"
        )


# ---------------------------------------------------------------------------
# BrainAPI — the main entry point
# ---------------------------------------------------------------------------


class BrainAPI:
    """High-level API for TRIBE v2 brain-state analysis.

    Wraps the full pipeline — model loading, event construction, feature
    extraction, Transformer inference, atlas mapping, scoring, and
    classification — into a single ``analyze()`` call.

    Use the ``load()`` classmethod to create an instance::

        api = BrainAPI.load()
        result = api.analyze("Your text here")

    Parameters
    ----------
    model : TribeModel
        A loaded TRIBE v2 model instance (from ``TribeModel.from_pretrained``).
    atlas : BrainAtlas
        A Destrieux atlas instance for region mapping.
    classifier : BrainStateClassifier or None
        A classifier with pre-built reference profiles.  If ``None``,
        classification results will not be included in ``analyze()`` output.
    cache_dir : Path
        Directory used for caching temp files and features.

    See Also
    --------
    BrainAPI.load : The recommended way to create an instance.
    BrainResult : The structured result type returned by ``analyze()``.

    Notes
    -----
    **Performance** (Mac Mini M4, 16 GB):

    - First run per text: ~3–5 minutes (feature extraction)
    - Cached runs: ~5–10 seconds
    - Model loading: ~15 seconds (first run downloads ~1 GB)

    **Thread Safety**: This class is NOT thread-safe.  The underlying
    model uses GPU/MPS resources that cannot be shared across threads.
    For concurrent use, create separate ``BrainAPI`` instances.
    """

    def __init__(
        self,
        model: tp.Any,
        atlas: tp.Any,
        classifier: tp.Any | None = None,
        cache_dir: Path = Path("./cache"),
    ) -> None:
        self._model = model
        self._atlas = atlas
        self._classifier = classifier
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(
        cls,
        model_id: str = "facebook/tribev2",
        *,
        cache_dir: str | Path = "./cache",
        stimuli_dir: str | Path = "./stimuli",
        build_profiles: bool = False,
        max_profile_passages: int = 3,
        config_update: dict | None = None,
        device: str = "auto",
    ) -> "BrainAPI":
        """Load the TRIBE v2 model, atlas, and (optionally) reference profiles.

        This is the recommended way to create a ``BrainAPI`` instance.

        Parameters
        ----------
        model_id : str
            HuggingFace repo id or local path containing ``config.yaml``
            and ``best.ckpt``.  Default: ``"facebook/tribev2"``.
        cache_dir : str or Path
            Directory for caching features and temp files.  Created
            automatically if it doesn't exist.
        stimuli_dir : str or Path
            Directory containing reference stimulus files (``happy.txt``,
            ``sad.txt``, etc.) for building classification profiles.
            Default: ``"./stimuli"``.
        build_profiles : bool
            If ``True``, immediately build reference profiles from the
            stimuli directory.  If ``False`` (default), profiles are built
            lazily on the first ``analyze()`` call (only if ``stimuli_dir``
            exists and contains ``.txt`` files).
        max_profile_passages : int
            Maximum number of passages per state file to use when building
            profiles.  Default: 3.
        config_update : dict, optional
            Additional config overrides passed to
            ``TribeModel.from_pretrained()``.  Common overrides::

                {
                    "data.text_feature.model_name": "alpindale/Llama-3.2-3B",
                    "data.text_feature.device": "cpu",
                }

        device : str
            Torch device string.  ``"auto"`` (default) selects MPS on
            Apple Silicon, CUDA on NVIDIA GPUs, or CPU as fallback.

        Returns
        -------
        BrainAPI
            Ready-to-use API instance.

        Raises
        ------
        RuntimeError
            If model loading fails (e.g. network error, missing checkpoint).

        Examples
        --------
        Basic usage with defaults::

            api = BrainAPI.load()

        With custom config (e.g., alternative LLaMA source)::

            api = BrainAPI.load(
                config_update={
                    "data.text_feature.model_name": "alpindale/Llama-3.2-3B",
                    "data.text_feature.device": "cpu",
                }
            )

        Pre-build all reference profiles at load time::

            api = BrainAPI.load(build_profiles=True, max_profile_passages=5)
        """
        from tribev2.brain_states import (
            BrainAtlas,
            BrainStateClassifier,
            BrainStateProfiler,
        )
        from tribev2.demo_utils import TribeModel

        cache_dir = Path(cache_dir)
        stimuli_dir = Path(stimuli_dir)

        # 1. Load the model
        logger.info("Loading TRIBE v2 model from '%s'...", model_id)
        default_config = {
            "data.text_feature.model_name": "alpindale/Llama-3.2-3B",
            "data.text_feature.device": "cpu",
        }
        if config_update:
            default_config.update(config_update)

        model = TribeModel.from_pretrained(
            model_id,
            cache_folder=str(cache_dir),
            device=device,
            config_update=default_config,
        )

        # 2. Initialize atlas
        atlas = BrainAtlas()

        # 3. Optionally build reference profiles
        classifier = None
        if stimuli_dir.is_dir() and any(stimuli_dir.glob("*.txt")):
            if build_profiles:
                logger.info(
                    "Building reference profiles from '%s'...", stimuli_dir
                )
                profiler = BrainStateProfiler(
                    model=model,
                    stimuli_dir=str(stimuli_dir),
                    cache_dir=str(cache_dir / "brain_states"),
                )
                profiles = profiler.build_profiles(
                    max_passages=max_profile_passages
                )
                classifier = BrainStateClassifier(
                    profiles=profiles, atlas=atlas
                )
                logger.info(
                    "Loaded %d reference profiles: %s",
                    len(profiles),
                    list(profiles.keys()),
                )
            else:
                # Lazy: try to load pre-built cached profiles
                classifier = cls._try_load_cached_profiles(
                    stimuli_dir, cache_dir / "brain_states", atlas
                )
        else:
            logger.info(
                "No stimuli directory found at '%s'; classification disabled. "
                "Create .txt files there to enable emotion classification.",
                stimuli_dir,
            )

        api = cls(
            model=model,
            atlas=atlas,
            classifier=classifier,
            cache_dir=cache_dir,
        )
        logger.info("BrainAPI ready.")
        return api

    @staticmethod
    def _try_load_cached_profiles(
        stimuli_dir: Path,
        cache_dir: Path,
        atlas: tp.Any,
    ) -> tp.Any | None:
        """Attempt to load pre-built profiles from cached .npy files.

        Returns a ``BrainStateClassifier`` if cached profiles exist for
        at least one state, otherwise ``None``.
        """
        from tribev2.brain_states import BrainStateClassifier

        if not cache_dir.is_dir():
            return None

        profiles: dict[str, np.ndarray] = {}
        for txt_file in sorted(stimuli_dir.glob("*.txt")):
            state = txt_file.stem
            # Look for any cached passage files for this state
            cached_files = sorted(cache_dir.glob(f"{state}_*.npy"))
            if cached_files:
                patterns = [np.load(f) for f in cached_files]
                profiles[state] = np.stack(patterns).mean(axis=0)

        if profiles:
            logger.info(
                "Loaded %d cached profiles: %s",
                len(profiles),
                list(profiles.keys()),
            )
            return BrainStateClassifier(profiles=profiles, atlas=atlas)
        return None

    # ── Public API ────────────────────────────────────────────────────

    def analyze(
        self,
        text: str,
        *,
        include_raw: bool = False,
        include_time_series: bool = True,
    ) -> BrainResult:
        """Analyze a text string and return brain-state scores.

        This is the main method.  It handles the entire pipeline:
        text → temp file → events → feature extraction → Transformer
        inference → atlas mapping → scoring → classification.

        Parameters
        ----------
        text : str
            The text to analyze.  Can be any length, but very short
            texts (< 5 words) may produce unreliable predictions.
        include_raw : bool
            If ``True``, include the full ``(n_timesteps, 20484)``
            vertex-level predictions in the result.  Default ``False``
            to keep results lightweight.
        include_time_series : bool
            If ``True`` (default), include per-timestep activation
            scores for each region group.

        Returns
        -------
        BrainResult
            Structured result with scores, composites, and optionally
            classification and raw data.

        Raises
        ------
        ValueError
            If ``text`` is empty or only whitespace.

        Examples
        --------
        Basic analysis::

            result = api.analyze("The child laughed and ran through the park.")
            print(result.valence)   # +0.025
            print(result.scores)    # {"prefrontal": 0.58, ...}

        With raw predictions for custom analysis::

            result = api.analyze("...", include_raw=True)
            preds = result.raw_predictions  # shape: (n_timesteps, 20484)

        Access the summary::

            print(result.summary())
        """
        text = text.strip()
        if not text:
            raise ValueError("Text must not be empty.")

        from tribev2.brain_states import compute_normalized_scores

        # 1. Write text to a temp file, run the pipeline, clean up
        preds = self._run_inference(text)

        # 2. Compute normalized scores
        norm = compute_normalized_scores(preds, self._atlas)

        # 3. Classify against reference profiles (if available)
        classification = None
        if self._classifier is not None:
            mean_pattern = preds.mean(axis=0)
            classification = self._classifier.classify(mean_pattern)

        # 4. Build result
        return BrainResult(
            text=text,
            scores=norm["scores"],
            valence=norm["valence"],
            learning=norm["learning"],
            attention=norm["attention"],
            classification=classification,
            time_series=norm["time_series"] if include_time_series else None,
            raw_predictions=preds if include_raw else None,
            n_timesteps=preds.shape[0],
        )

    def analyze_file(
        self,
        path: str | Path,
        *,
        include_raw: bool = False,
        include_time_series: bool = True,
    ) -> BrainResult:
        """Analyze text from a file path.

        Convenience wrapper around ``analyze()`` that reads the file first.

        Parameters
        ----------
        path : str or Path
            Path to a ``.txt`` file containing the text to analyze.
        include_raw : bool
            If ``True``, include raw vertex predictions.
        include_time_series : bool
            If ``True`` (default), include per-timestep scores.

        Returns
        -------
        BrainResult
            Same structured result as ``analyze()``.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is empty.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        text = path.read_text(encoding="utf-8")
        return self.analyze(
            text, include_raw=include_raw, include_time_series=include_time_series
        )

    def compare(
        self,
        text_a: str,
        text_b: str,
        *,
        include_raw: bool = False,
    ) -> dict[str, tp.Any]:
        """Compare brain-state responses between two texts.

        Runs ``analyze()`` on both texts and returns a differential
        analysis showing which brain regions respond differently.

        Parameters
        ----------
        text_a : str
            First text (treated as the "positive" direction).
        text_b : str
            Second text (treated as the "negative" direction).
        include_raw : bool
            If ``True``, include raw vertex predictions for both texts
            and the differential map.

        Returns
        -------
        dict with keys:
            - ``"result_a"`` — ``BrainResult`` for text_a
            - ``"result_b"`` — ``BrainResult`` for text_b
            - ``"score_diff"`` — dict of ``group → (score_a - score_b)``
            - ``"valence_diff"`` — valence difference (a - b)
            - ``"differential_map"`` — ``(20484,)`` vertex-level diff
              (only if ``include_raw=True``)

        Examples
        --------
        >>> diff = api.compare(
        ...     "The child laughed and ran through the sunlit park.",
        ...     "The empty house echoed with silence and regret."
        ... )
        >>> diff["score_diff"]
        {"prefrontal": 0.03, "reward_vmPFC": 0.08, "insula": -0.05, ...}
        >>> diff["valence_diff"]
        0.12
        """
        result_a = self.analyze(text_a, include_raw=include_raw)
        result_b = self.analyze(text_b, include_raw=include_raw)

        score_diff = {
            group: result_a.scores[group] - result_b.scores[group]
            for group in result_a.scores
        }

        output: dict[str, tp.Any] = {
            "result_a": result_a,
            "result_b": result_b,
            "score_diff": score_diff,
            "valence_diff": result_a.valence - result_b.valence,
            "learning_diff": result_a.learning - result_b.learning,
            "attention_diff": result_a.attention - result_b.attention,
        }

        if include_raw and result_a.raw_predictions is not None and result_b.raw_predictions is not None:
            diff_a = result_a.raw_predictions.mean(axis=0)
            diff_b = result_b.raw_predictions.mean(axis=0)
            output["differential_map"] = diff_a - diff_b

        return output

    # ── Internal helpers ──────────────────────────────────────────────

    def _run_inference(self, text: str) -> np.ndarray:
        """Run the full TRIBE v2 inference pipeline on a text string.

        Writes the text to a temporary file, builds the events DataFrame,
        runs prediction, and cleans up.  Feature extraction results are
        cached by the underlying model, so repeated calls with the same
        text are fast.

        Parameters
        ----------
        text : str
            The text to process (must be non-empty, already stripped).

        Returns
        -------
        np.ndarray
            Predictions of shape ``(n_timesteps, 20484)``.
        """
        # Use a deterministic temp file path so caching works across calls
        # with the same text content.
        import hashlib

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        tmp_path = self._cache_dir / f"_api_input_{text_hash}.txt"

        try:
            tmp_path.write_text(text, encoding="utf-8")
            df = self._model.get_events_dataframe(text_path=str(tmp_path))
            preds, _ = self._model.predict(events=df, verbose=True)
            return preds
        finally:
            # Clean up the temp file (cached features persist separately)
            if tmp_path.exists():
                tmp_path.unlink()

    @property
    def has_profiles(self) -> bool:
        """Whether reference profiles are loaded for classification."""
        return self._classifier is not None and bool(self._classifier.profiles)

    @property
    def available_states(self) -> list[str]:
        """List of available reference states for classification.

        Returns an empty list if no profiles are loaded.
        """
        if self._classifier is None:
            return []
        return list(self._classifier.profiles.keys())

    @property
    def region_groups(self) -> list[str]:
        """List of the functional region group names used in scoring."""
        from tribev2.brain_states import REGION_GROUPS

        return list(REGION_GROUPS.keys())

    def unload(self) -> None:
        """Release all model resources to free memory.

        Clears the TribeModel (including its internal Fusion Transformer
        and all feature extractors), the BrainAtlas, and the
        BrainStateClassifier.  Also flushes GPU/MPS caches and triggers
        garbage collection.

        After calling ``unload()``, this ``BrainAPI`` instance is no
        longer usable.  Create a new one via ``BrainAPI.load()``.

        This method is safe to call multiple times.
        """
        import gc

        # 1. Free feature extractor models held by TribeModel/TribeExperiment
        if self._model is not None:
            # The TribeModel (a TribeExperiment subclass) has a `data`
            # attribute with extractors for each modality.
            data = getattr(self._model, "data", None)
            if data is not None:
                for modality in ("text", "audio", "video", "image"):
                    extractor = getattr(data, f"{modality}_feature", None)
                    if extractor is None:
                        continue
                    # Free the HuggingFace model + tokenizer cached on
                    # the extractor (same logic as _free_extractor_model
                    # in main.py, but more thorough).
                    targets = [extractor]
                    if hasattr(extractor, "image"):
                        targets.append(extractor.image)
                    for target in targets:
                        for attr in ("_model", "_tokenizer", "_processor"):
                            if hasattr(target, attr):
                                try:
                                    delattr(target, attr)
                                except Exception:
                                    pass
                    logger.debug("Freed extractor: %s_feature", modality)

            # 2. Free the Fusion Transformer (stored as _model on the
            #    TribeExperiment instance).
            fusion_model = getattr(self._model, "_model", None)
            if fusion_model is not None:
                try:
                    self._model._model = None
                except Exception:
                    pass
                del fusion_model

        # 3. Clear our own references
        self._model = None
        self._atlas = None
        self._classifier = None

        # 4. Force garbage collection
        gc.collect()

        # 5. Flush device caches
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("BrainAPI unloaded — all model resources freed")

    @property
    def is_loaded(self) -> bool:
        """Whether the model is still loaded and usable."""
        return self._model is not None

    def __repr__(self) -> str:
        if not self.is_loaded:
            return "BrainAPI(unloaded)"
        states = self.available_states
        state_str = f", states={states}" if states else ""
        return f"BrainAPI(model=loaded{state_str})"
