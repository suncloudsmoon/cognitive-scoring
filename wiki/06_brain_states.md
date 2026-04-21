# Brain State Analysis (`brain_states.py`)

## Overview

The `tribev2/brain_states.py` module is the core analytical engine built on top of TRIBE v2. It translates the raw 20,484-vertex predictions into interpretable metrics.

It provides three main analytical approaches:
1. **Normalized Scoring (Z-Score + Sigmoid)**
2. **Composite Cognitive Indices**
3. **Reference Profiling & Classification**

## 1. Normalized Scoring Pipeline

Raw prediction values mean little on their own. We adopted a robust normalization pipeline mathematically equivalent to the one used in the HuggingFace Ad Scorer project.

**Process in `compute_normalized_scores(preds, atlas)`**:
1. Compute the global `mean` and `std` of the entire brain state at that timestep.
2. For each region group (e.g., `prefrontal`), calculate its raw mean activation.
3. Compute the **Z-score**: `(group_mean - global_mean) / global_std`
4. Apply a **sigmoid function**: `1 / (1 + exp(-z))`

**Why this is good**:
- It maps unbounded raw values to a clean `[0.0, 1.0]` range.
- `0.5` represents baseline (global average) activity.
- Scores > 0.5 indicate that specific network is *more active* than the rest of the brain.
- It normalizes for overall volume (e.g., loud audio causes global spikes; this normalization isolates which specific regions spiked the most).

## 2. Composite Cognitive Indices

Using the normalized group scores, we calculate "meta-scores" that try to answer specific behavioral questions.

*Note: These are heuristic formulas based on general neuroscience principles.*

**Emotional Valence**
* `Valence = Reward_PFC - (Insula + ACC) / 2`
* Estimates positive vs. negative affect by contrasting the reward network (vmPFC) against the salience network (Insula, ACC). Positive values = happier/positive. Negative values = sad/distressed/negative.

**Learning Readiness**
* `Learning = (PFC + ACC + Temporal) / 3`
* High engagement in frontal executive areas (PFC/ACC) and semantic/language areas (Temporal). Indicates deep cognitive processing.

**Attention Engagement**
* `Attention = (Parietal + PFC) / 2`
* Averages the dorsal attention network (Parietal) and executive control (PFC).

## 3. Reference Profiling & Classification

Instead of relying solely on region heuristics, we can use an empirical, data-driven approach: feed the model a "happy" stimulus, see what the brain does, and save that pattern as a **Reference Profile**.

### `BrainStateProfiler`
This class builds the reference signatures.
1. It reads text files in `stimuli/` (e.g., `happy.txt`, `sad.txt`, `curious.txt`).
2. Each file contains multiple paragraphs exemplifying that state.
3. It runs TRIBE v2 on each paragraph.
4. It averages the resulting vertex patterns across all paragraphs for that state, yielding a single 20,484-vector representing, for example, the "canonical happy brain shape."
5. Resulting profiles are cached in `cache/brain_states/*.npy` to avoid recomputation.

### `BrainStateClassifier`
This class takes a *new* prediction (e.g., an ad you are testing) and compares it to the reference profiles.
1. It computes the **Pearson correlation coefficient (r)** between the new pattern and each reference profile.
2. It ranks the matches. E.g., `[("happy", 0.65), ("calm", 0.32), ("sad", -0.15)]`.

### Differential Mapping
The profiler can also subtract one profile from another (`profiler.differential_map("happy", "sad")`). This produces a new map showing exactly *where* the "happy" brain differs from the "sad" brain, which can be visualized using `PlotBrain`.

## 4. Visualizations

The module includes Plotly functions for generating reports:
- `create_radar_chart(scores)`: Displays the 10 region group scores on a web-like graph, providing an instant visual signature of the cognitive state.
- `create_timeline_chart(time_series)`: Shows line graphs tracking how specific networks (e.g., `visual` vs. `prefrontal`) rise and fall over the duration of a video or audio clip.

## Key Source File
- `tribev2/brain_states.py`
