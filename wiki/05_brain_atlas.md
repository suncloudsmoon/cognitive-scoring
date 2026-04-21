# The Brain Atlas (Destrieux)

## Overview

TRIBE v2 predicts activity at 20,484 vertices on the cortical surface. To make sense of this data, we need a map that tells us what each vertex represents (e.g., "this vertex is in the visual cortex," "this vertex is in the prefrontal cortex").

This mapping is called a **brain atlas**. In this project, we use the **Destrieux Atlas**, which divides the cortical surface into 74 distinct regions per hemisphere (148 total regions + "Unknown" / "Medial Wall").

## How the Atlas Works

The atlas is loaded using the `nilearn` library in `tribev2/brain_states.py` (`BrainAtlas` class).

1. `nilearn.datasets.fetch_atlas_surf_destrieux()` downloads the atlas.
2. The atlas contains two arrays (`map_left` and `map_right`), each with 10,242 integers.
3. Each integer is an index into a list of region names (labels).
4. If `map_left[500] == 12`, it means vertex 500 on the left hemisphere belongs to region index 12 (e.g., `"G_front_sup"`).

## Grouping Regions into Functional Networks

The 148 raw Destrieux regions are too granular for high-level cognitive analysis. For example, there are 8 different sub-regions in the prefrontal cortex alone.

In `tribev2/brain_states.py`, we defined `REGION_GROUPS`, which maps groups of Destrieux regions to higher-level functional/cognitive networks:

| Group Name | Destrieux Regions Included | Functional Role |
|------------|---------------------------|-----------------|
| `prefrontal` | `G_front_sup`, `G_front_middle`, `G_front_inf-*`, etc. | Executive function, planning, complex cognition |
| `reward_vmPFC` | `G_orbital`, `G_rectus`, `G_subcallosal`, etc. | Reward processing, positive valence, valuation |
| `anterior_cingulate`| `G_and_S_cingul-Ant`, `G_and_S_cingul-Mid-Ant` | Conflict monitoring, curiosity, emotion regulation |
| `default_mode` | `G_and_S_cingul-Mid-Post`, `G_precuneus`, etc. | Self-referential thought, mind-wandering |
| `insula` | `G_insular_short`, `S_circular_insula_*`, etc. | Interoception, emotional awareness, negative affect |
| `temporal` | `G_temp_sup-*`, `G_temporal_middle`, `Pole_temporal`, etc. | Language comprehension, auditory processing, social cognition |
| `visual` | `G_cuneus`, `G_occipital_*`, `S_calcarine`, etc. | Visual processing |
| `attention_parietal`| `G_parietal_sup`, `G_pariet_inf-*`, `S_intrapariet_*` | Spatial attention, dorsal attention network |
| `motor` | `G_precentral`, `G_postcentral`, etc. | Sensorimotor processing |
| `fusiform_parahip` | `G_oc-temp_lat-fusifor`, `G_oc-temp_med-Parahip` | Memory encoding, face/object recognition |

## Extracting Region Activations

The `BrainAtlas` class provides methods to go from raw predictions to region scores:

1. **`extract_region_activations(preds)`**: Averages the vertices within each of the 148 Destrieux regions.
2. **`extract_group_activations(preds)`**: Averages the vertices within each of the 10 custom groups defined above.

```python
from tribev2.brain_states import BrainAtlas

atlas = BrainAtlas()
# preds shape: (n_timesteps, 20484)
groups = atlas.extract_group_activations(preds)
# Returns dict: {"prefrontal": [0.5, 0.6, ...], "visual": [1.2, 1.1, ...]}
```

## Important Caveat: The Cortex vs. Subcortex

It is vital to remember that **fsaverage5 maps the cortical surface only**. It is like a map of the Earth's crust.

Crucial "deep brain" (subcortical) structures are **missing**:
- **Amygdala**: The core center for fear and intense emotion.
- **Hippocampus**: The core center for episodic memory formation.
- **Basal Ganglia / Striatum**: Core movement and reward pathways.

Because these regions are invisible to TRIBE v2, our analysis in `brain_states.py` relies entirely on **cortical correlates**. For example, since we can't see the amygdala, we look at the insula and anterior cingulate as proxies for emotional arousal and negative affect. Since we can't see the core hippocampus, we look at the parahippocampal gyrus and temporal pole for memory processing.
