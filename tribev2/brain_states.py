# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Brain state analysis: derive emotional/cognitive states from TRIBE v2 predictions.

This module provides tools to:
1. Map TRIBE v2's ~20k vertex predictions to named brain regions (Destrieux atlas)
2. Build reference "brain signatures" for emotional/cognitive states
3. Classify novel stimuli by comparing predicted brain patterns to references
4. Compute composite scores for valence, learning readiness, and attention

Important caveats:
- TRIBE v2 predicts cortical surface activity only; subcortical structures
  (amygdala, hippocampus, basal ganglia) are NOT represented.
- All labels are approximations based on cortical correlates.
- This is NOT suitable for clinical use.
"""

import logging
import typing as tp
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Destrieux atlas region groupings for emotional/cognitive analysis
# Keys = semantic group name, Values = list of Destrieux label strings
# ---------------------------------------------------------------------------

# Prefrontal cortex regions — executive function, planning, decision-making
PREFRONTAL_REGIONS = [
    "G_front_sup",              # Superior frontal gyrus
    "G_front_middle",           # Middle frontal gyrus
    "G_front_inf-Opercular",    # Inferior frontal gyrus (opercular, Broca's)
    "G_front_inf-Triangul",     # Inferior frontal gyrus (triangular, Broca's)
    "G_front_inf-Orbital",      # Inferior frontal gyrus (orbital)
    "S_front_sup",              # Superior frontal sulcus
    "S_front_middle",           # Middle frontal sulcus
    "S_front_inf",              # Inferior frontal sulcus
]

# Orbitofrontal / ventromedial PFC — reward, positive valence
REWARD_REGIONS = [
    "G_orbital",                # Orbital gyri
    "G_rectus",                 # Gyrus rectus (medial orbital)
    "G_subcallosal",            # Subcallosal gyrus
    "S_orbital_lateral",        # Lateral orbital sulcus
    "S_orbital_med-olfact",     # Medial orbital / olfactory sulcus
    "S_orbital-H_Shaped",      # H-shaped orbital sulcus
    "S_suborbital",             # Suborbital sulcus
    "G_and_S_frontomargin",     # Frontomarginal gyrus and sulcus
]

# Anterior cingulate cortex — conflict monitoring, curiosity, uncertainty
ACC_REGIONS = [
    "G_and_S_cingul-Ant",       # Anterior cingulate
    "G_and_S_cingul-Mid-Ant",   # Mid-anterior cingulate
]

# Posterior cingulate / precuneus — default mode network, self-referential
DMN_REGIONS = [
    "G_and_S_cingul-Mid-Post",  # Mid-posterior cingulate
    "G_cingul-Post-dorsal",     # Posterior cingulate (dorsal)
    "G_cingul-Post-ventral",    # Posterior cingulate (ventral)
    "G_precuneus",              # Precuneus
]

# Insula — interoception, negative affect, emotional awareness
INSULA_REGIONS = [
    "G_insular_short",          # Short insular gyri (anterior insula)
    "G_Ins_lg_and_S_cent_ins",  # Long insular gyrus + central insular sulcus
    "S_circular_insula_ant",    # Anterior circular insular sulcus
    "S_circular_insula_inf",    # Inferior circular insular sulcus
    "S_circular_insula_sup",    # Superior circular insular sulcus
]

# Temporal lobe — language, auditory, social processing
TEMPORAL_REGIONS = [
    "G_temp_sup-Lateral",       # Superior temporal gyrus (lateral)
    "G_temp_sup-Plan_polar",    # Planum polare
    "G_temp_sup-Plan_tempo",    # Planum temporale
    "G_temp_sup-G_T_transv",    # Transverse temporal gyrus (Heschl's)
    "G_temporal_middle",        # Middle temporal gyrus
    "G_temporal_inf",           # Inferior temporal gyrus
    "S_temporal_sup",           # Superior temporal sulcus
    "S_temporal_inf",           # Inferior temporal sulcus
    "S_temporal_transverse",    # Transverse temporal sulcus
    "Pole_temporal",            # Temporal pole
]

# Visual / occipital cortex — visual processing
VISUAL_REGIONS = [
    "G_cuneus",                 # Cuneus
    "G_occipital_sup",          # Superior occipital gyrus
    "G_occipital_middle",       # Middle occipital gyrus
    "G_and_S_occipital_inf",    # Inferior occipital gyrus and sulcus
    "Pole_occipital",           # Occipital pole
    "S_calcarine",              # Calcarine sulcus (V1)
    "G_oc-temp_med-Lingual",    # Lingual gyrus
    "S_oc_middle_and_Lunatus",  # Middle occipital sulcus
    "S_oc_sup_and_transversal", # Superior occipital sulcus
]

# Parietal regions — attention, spatial processing
ATTENTION_REGIONS = [
    "G_parietal_sup",           # Superior parietal lobule
    "G_pariet_inf-Angular",     # Angular gyrus
    "G_pariet_inf-Supramar",    # Supramarginal gyrus
    "S_intrapariet_and_P_trans",# Intraparietal sulcus (DAN hub)
]

# Sensorimotor cortex
MOTOR_REGIONS = [
    "G_precentral",             # Precentral gyrus (primary motor)
    "G_postcentral",            # Postcentral gyrus (primary somatosensory)
    "G_and_S_paracentral",      # Paracentral lobule
    "G_and_S_subcentral",       # Subcentral gyrus and sulcus
]

# Fusiform / parahippocampal — face/object recognition, memory-related cortex
FUSIFORM_PARAHIP_REGIONS = [
    "G_oc-temp_lat-fusifor",    # Fusiform gyrus
    "G_oc-temp_med-Parahip",    # Parahippocampal gyrus
]

# All functional groups for convenient iteration
REGION_GROUPS: dict[str, list[str]] = {
    "prefrontal": PREFRONTAL_REGIONS,
    "reward_vmPFC": REWARD_REGIONS,
    "anterior_cingulate": ACC_REGIONS,
    "default_mode": DMN_REGIONS,
    "insula": INSULA_REGIONS,
    "temporal": TEMPORAL_REGIONS,
    "visual": VISUAL_REGIONS,
    "attention_parietal": ATTENTION_REGIONS,
    "motor": MOTOR_REGIONS,
    "fusiform_parahip": FUSIFORM_PARAHIP_REGIONS,
}


class BrainAtlas:
    """Wraps the Destrieux cortical atlas for fsaverage5, providing
    region-level activation extraction from TRIBE v2 predictions.

    Parameters
    ----------
    mesh : str
        FreeSurfer mesh name (default ``"fsaverage5"``).
    """

    def __init__(self, mesh: str = "fsaverage5"):
        from nilearn import datasets

        self.mesh = mesh
        atlas = datasets.fetch_atlas_surf_destrieux()
        self.labels: list[str] = [str(l) for l in atlas["labels"]]
        self.map_left: np.ndarray = np.asarray(atlas["map_left"])
        self.map_right: np.ndarray = np.asarray(atlas["map_right"])
        self._label_to_idx = {name: i for i, name in enumerate(self.labels)}

    def region_mask(self, region_name: str, hemisphere: str = "both") -> np.ndarray:
        """Return a boolean mask over the full vertex array for a named region.

        Parameters
        ----------
        region_name : str
            Destrieux label (e.g. ``"G_front_sup"``).
        hemisphere : str
            ``"left"``, ``"right"``, or ``"both"`` (default).

        Returns
        -------
        np.ndarray
            Boolean mask of shape ``(20484,)`` (or ``(10242,)`` for single hemi).
        """
        idx = self._label_to_idx.get(region_name)
        if idx is None:
            raise ValueError(
                f"Unknown region '{region_name}'. "
                f"Available: {[l for l in self.labels if l != 'Unknown']}"
            )

        left_mask = self.map_left == idx
        right_mask = self.map_right == idx

        if hemisphere == "left":
            return left_mask
        elif hemisphere == "right":
            return right_mask
        else:
            return np.concatenate([left_mask, right_mask])

    def group_mask(self, group_name: str) -> np.ndarray:
        """Return a combined boolean mask for a functional region group.

        Parameters
        ----------
        group_name : str
            One of the keys in ``REGION_GROUPS`` (e.g. ``"prefrontal"``).
        """
        regions = REGION_GROUPS.get(group_name)
        if regions is None:
            raise ValueError(
                f"Unknown group '{group_name}'. Available: {list(REGION_GROUPS)}"
            )
        mask = np.zeros(len(self.map_left) + len(self.map_right), dtype=bool)
        for r in regions:
            try:
                mask |= self.region_mask(r)
            except ValueError:
                logger.warning("Skipping unknown region '%s'", r)
        return mask

    def extract_region_activations(
        self, preds: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Extract mean activation per Destrieux region for each timestep.

        Parameters
        ----------
        preds : np.ndarray
            Predictions array of shape ``(n_timesteps, 20484)``.

        Returns
        -------
        dict[str, np.ndarray]
            Region name → array of shape ``(n_timesteps,)``.
        """
        results = {}
        for label in self.labels:
            if label == "Unknown" or label == "Medial_wall":
                continue
            mask = self.region_mask(label)
            if mask.sum() > 0:
                results[label] = preds[:, mask].mean(axis=1)
        return results

    def extract_group_activations(
        self, preds: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Extract mean activation per functional group for each timestep.

        Parameters
        ----------
        preds : np.ndarray
            Predictions array of shape ``(n_timesteps, 20484)``.

        Returns
        -------
        dict[str, np.ndarray]
            Group name → array of shape ``(n_timesteps,)``.
        """
        results = {}
        for group_name in REGION_GROUPS:
            mask = self.group_mask(group_name)
            if mask.sum() > 0:
                results[group_name] = preds[:, mask].mean(axis=1)
        return results


# ---------------------------------------------------------------------------
# Normalized scoring (z-score → sigmoid, adopted from the ad-project approach)
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)),
    )


def compute_normalized_scores(
    preds: np.ndarray,
    atlas: "BrainAtlas",
) -> dict[str, tp.Any]:
    """Compute z-score-normalized, sigmoid-mapped scores per region group.

    This is the main scoring pipeline, adopted from the ad-project's proven
    approach.  Each functional group's mean activation is z-scored relative
    to global brain activation, then mapped to 0–1 via sigmoid.

    - 0.5 = average activation
    - >0.5 = above-average engagement
    - <0.5 = below-average engagement

    Parameters
    ----------
    preds : np.ndarray
        Predictions of shape ``(n_timesteps, 20484)``.
    atlas : BrainAtlas
        Atlas instance for region grouping.

    Returns
    -------
    dict with keys:
        - ``"scores"`` — dict of group_name → float (time-averaged 0–1 score)
        - ``"time_series"`` — dict of group_name → list[float] (per-timestep)
        - ``"valence"`` — float, positive = happy, negative = sad
        - ``"learning"`` — float, learning readiness score
        - ``"attention"`` — float, attention engagement score
    """
    global_mean = np.mean(preds)
    global_std = np.std(preds)
    if global_std < 1e-8:
        global_std = 1.0

    scores = {}
    time_series = {}

    for group_name in REGION_GROUPS:
        mask = atlas.group_mask(group_name)
        if mask.sum() == 0:
            scores[group_name] = 0.5
            time_series[group_name] = [0.5] * preds.shape[0]
            continue

        region_ts = preds[:, mask].mean(axis=1)
        region_z = (region_ts - global_mean) / global_std
        region_norm = _sigmoid(region_z)

        time_series[group_name] = region_norm.tolist()
        scores[group_name] = float(np.mean(region_norm))

    # Composite scores
    valence = scores.get("reward_vmPFC", 0.5) - (
        scores.get("insula", 0.5) + scores.get("anterior_cingulate", 0.5)
    ) / 2

    learning = (
        scores.get("prefrontal", 0.5)
        + scores.get("anterior_cingulate", 0.5)
        + scores.get("temporal", 0.5)
    ) / 3

    attention = (
        scores.get("attention_parietal", 0.5)
        + scores.get("prefrontal", 0.5)
    ) / 2

    return {
        "scores": scores,
        "time_series": time_series,
        "valence": valence,
        "learning": learning,
        "attention": attention,
    }


# ---------------------------------------------------------------------------
# Plotly visualization (adopted from the ad-project's approach)
# ---------------------------------------------------------------------------

# Display-friendly names for region groups
_GROUP_DISPLAY_NAMES = {
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

_PLOTLY_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def create_radar_chart(
    scores: dict[str, float],
    title: str = "Brain State Profile",
) -> tp.Any:
    """Create a Plotly radar chart showing scores per region group.

    Parameters
    ----------
    scores : dict
        Group name → normalized score (0–1).
    title : str
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    categories = list(scores.keys())
    display_names = [_GROUP_DISPLAY_NAMES.get(c, c) for c in categories]
    values = [scores[c] for c in categories]

    # Close the polygon
    display_names_closed = display_names + [display_names[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=display_names_closed,
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.2)",
        line=dict(color="rgba(99, 110, 250, 0.8)", width=2),
        name="Activation",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            tickvals=[0.25, 0.5, 0.75, 1.0]),
        ),
        showlegend=False,
        title=dict(text=title, x=0.5),
        margin=dict(l=80, r=80, t=60, b=40),
        height=500,
    )
    return fig


def create_timeline_chart(
    time_series: dict[str, list[float]],
    selected_groups: list[str] | None = None,
    tr_seconds: float = 1.0,
) -> tp.Any:
    """Create a Plotly timeline chart showing activation over time.

    Parameters
    ----------
    time_series : dict
        Group name → list of per-timestep scores.
    selected_groups : list of str, optional
        Which groups to plot (default: all).
    tr_seconds : float
        Seconds per TR for x-axis labeling.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    if selected_groups is None:
        selected_groups = list(time_series.keys())

    fig = go.Figure()
    for i, group in enumerate(selected_groups):
        if group not in time_series:
            continue
        ts = time_series[group]
        x_sec = [t * tr_seconds for t in range(len(ts))]
        display_name = _GROUP_DISPLAY_NAMES.get(group, group)
        fig.add_trace(go.Scatter(
            x=x_sec, y=ts,
            mode="lines",
            name=display_name,
            line=dict(color=_PLOTLY_COLORS[i % len(_PLOTLY_COLORS)], width=2),
        ))

    fig.update_layout(
        title="Brain Activation Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Activation Score (0–1)",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        margin=dict(l=60, r=20, t=50, b=100),
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Brain State Profiler
# ---------------------------------------------------------------------------


class BrainStateProfiler:
    """Build reference brain signatures by running TRIBE v2 on curated stimuli.

    Parameters
    ----------
    model : TribeModel
        A loaded TRIBE v2 model instance.
    stimuli_dir : str or Path
        Directory containing ``<state>.txt`` files (e.g., ``happy.txt``).
    cache_dir : str or Path
        Directory to cache per-passage predictions as ``.npy`` files.
    """

    def __init__(
        self,
        model: tp.Any,
        stimuli_dir: str | Path = "./stimuli",
        cache_dir: str | Path = "./cache/brain_states",
    ):
        self.model = model
        self.stimuli_dir = Path(stimuli_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.profiles: dict[str, np.ndarray] = {}

    def _parse_passages(self, filepath: Path) -> list[str]:
        """Split a text file into individual passages (separated by blank lines)."""
        text = filepath.read_text(encoding="utf-8").strip()
        passages = [p.strip() for p in text.split("\n\n") if p.strip()]
        return passages

    def _predict_text(self, text: str, label: str, index: int) -> np.ndarray:
        """Run TRIBE v2 on a single text passage, with caching."""
        cache_file = self.cache_dir / f"{label}_{index}.npy"
        if cache_file.exists():
            logger.info("Loading cached prediction: %s", cache_file.name)
            return np.load(cache_file)

        # Write text to a temporary file for the model
        tmp_path = self.cache_dir / f"_tmp_{label}_{index}.txt"
        tmp_path.write_text(text, encoding="utf-8")

        try:
            df = self.model.get_events_dataframe(text_path=str(tmp_path))
            preds, _ = self.model.predict(events=df, verbose=False)
            # Average across time → single spatial pattern (20484,)
            pattern = preds.mean(axis=0)
            np.save(cache_file, pattern)
            logger.info("Saved prediction: %s (from %d timesteps)", cache_file.name, preds.shape[0])
            return pattern
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def build_profiles(
        self, states: list[str] | None = None, max_passages: int = 5
    ) -> dict[str, np.ndarray]:
        """Build or load reference brain profiles for each state category.

        Parameters
        ----------
        states : list of str, optional
            State names to build (must match ``<state>.txt`` filenames).
            If ``None``, auto-discovers from ``stimuli_dir``.
        max_passages : int
            Maximum number of passages to use per state.

        Returns
        -------
        dict[str, np.ndarray]
            State name → mean brain pattern of shape ``(20484,)``.
        """
        if states is None:
            states = sorted(
                p.stem for p in self.stimuli_dir.glob("*.txt")
            )

        for state in states:
            filepath = self.stimuli_dir / f"{state}.txt"
            if not filepath.exists():
                logger.warning("Stimulus file not found: %s", filepath)
                continue

            passages = self._parse_passages(filepath)[:max_passages]
            logger.info(
                "Building profile for '%s' (%d passages)...", state, len(passages)
            )

            patterns = []
            for i, passage in enumerate(passages):
                pattern = self._predict_text(passage, state, i)
                patterns.append(pattern)

            self.profiles[state] = np.stack(patterns).mean(axis=0)
            logger.info(
                "Profile '%s': mean of %d patterns, shape %s",
                state, len(patterns), self.profiles[state].shape,
            )

        return self.profiles

    def differential_map(self, state_a: str, state_b: str) -> np.ndarray:
        """Compute the vertex-wise difference between two state profiles.

        Parameters
        ----------
        state_a, state_b : str
            State names (must already be in ``self.profiles``).

        Returns
        -------
        np.ndarray
            Differential map of shape ``(20484,)``.
            Positive = state_a > state_b.
        """
        if state_a not in self.profiles or state_b not in self.profiles:
            available = list(self.profiles.keys())
            raise ValueError(
                f"Both states must be built first. Available: {available}"
            )
        return self.profiles[state_a] - self.profiles[state_b]


class BrainStateClassifier:
    """Classify novel brain predictions by comparing to reference profiles.

    Parameters
    ----------
    profiles : dict[str, np.ndarray]
        State name → reference brain pattern (from ``BrainStateProfiler``).
    atlas : BrainAtlas, optional
        If provided, also computes ROI-level interpretability scores.
    """

    def __init__(
        self,
        profiles: dict[str, np.ndarray],
        atlas: BrainAtlas | None = None,
    ):
        self.profiles = profiles
        self.atlas = atlas

    def classify(
        self,
        brain_pattern: np.ndarray,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Compare a brain pattern to all reference profiles.

        Parameters
        ----------
        brain_pattern : np.ndarray
            Single brain pattern of shape ``(20484,)`` (e.g., averaged prediction).
        top_k : int, optional
            Return only the top-k matches. If ``None``, return all.

        Returns
        -------
        list of (state_name, correlation)
            Ranked by descending Pearson correlation.
        """
        scores = []
        for name, ref in self.profiles.items():
            corr = float(np.corrcoef(brain_pattern, ref)[0, 1])
            scores.append((name, corr))
        scores.sort(key=lambda x: -x[1])
        if top_k is not None:
            scores = scores[:top_k]
        return scores

    def analyze(
        self, preds: np.ndarray
    ) -> dict[str, tp.Any]:
        """Full analysis of a prediction array.

        Parameters
        ----------
        preds : np.ndarray
            Predictions of shape ``(n_timesteps, 20484)``.

        Returns
        -------
        dict with keys:
            - ``"classification"`` — ranked state matches (if profiles loaded)
            - ``"scores"`` — per-group normalized scores (0–1)
            - ``"time_series"`` — per-group per-timestep scores
            - ``"valence"`` — emotional valence (positive = happy)
            - ``"learning"`` — learning readiness score
            - ``"attention"`` — attention engagement score
        """
        result: dict[str, tp.Any] = {}

        # Classify using time-averaged pattern (if profiles available)
        if self.profiles:
            mean_pattern = preds.mean(axis=0)
            result["classification"] = self.classify(mean_pattern)

        # Compute normalized scores (z-score + sigmoid)
        if self.atlas is not None:
            norm = compute_normalized_scores(preds, self.atlas)
            result["scores"] = norm["scores"]
            result["time_series"] = norm["time_series"]
            result["valence"] = norm["valence"]
            result["learning"] = norm["learning"]
            result["attention"] = norm["attention"]

        return result
