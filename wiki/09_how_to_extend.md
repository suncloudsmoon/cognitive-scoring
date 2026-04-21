# How to Extend the System

This guide explains how you or another LLM can build upon the current TRIBE v2 framework.

## Adding a New Cognitive State (Reference Profile)

If you want the system to be able to classify a new state (e.g., "Confusion" or "Excitement"), you must build a new Reference Profile.

1. Create a new text file: `stimuli/confusion.txt`
2. Write 5 different short paragraphs that clearly evoke the feeling of confusion or cognitive dissonance. Separate each paragraph with a dual newline (`\n\n`).
3. Open or run code similar to `brain_state_demo.ipynb`.
4. Run the profiler builder:
   ```python
   profiler = BrainStateProfiler(model, "./stimuli", "./cache/brain_states")
   # Automatically discovers confusion.txt and generates confusion_0.npy ... confusion_4.npy
   profiles = profiler.build_profiles() 
   ```
5. The `BrainStateClassifier` will now automatically include "confusion" in its Pearson correlation rankings.

## Creating a New Brain Region Group

If you need to track a specific network—for example, the "Language Network"—that isn't currently tracked in the radar chart:

1. Open `tribev2/brain_states.py`
2. Look up the appropriate `Destrieux` labels (you can print `atlas.labels` to see the 148 options).
3. Create a new list of regions near the top of the file:
   ```python
   LANGUAGE_REGIONS = [
       "G_front_inf-Opercular", # Broca's
       "G_front_inf-Triangul",
       "G_temp_sup-Lateral",    # Wernicke's area
       "G_temporal_middle",
       "G_pariet_inf-Supramar",
       "G_pariet_inf-Angular"
   ]
   ```
4. Add it to `REGION_GROUPS`:
   ```python
   REGION_GROUPS = {
       ...
       "language_network": LANGUAGE_REGIONS,
   }
   ```
5. Add a friendly display name to `_GROUP_DISPLAY_NAMES` so it looks nice on the Plotly charts:
   ```python
   _GROUP_DISPLAY_NAMES = {
       ...
       "language_network": "Language Processing",
   }
   ```
The scoring algorithms `compute_normalized_scores`, `create_radar_chart`, and `create_timeline_chart` dynamically iterate over `REGION_GROUPS`, so your new network will automatically appear in all outputs and charts without any changes to the scoring logic!

## Adding a New Heuristic Composite Score

If you want to create a new meta-score (like `Valence` or `Learning Readiness`), you can add a calculation based on the Z-score/Sigmoid output.

1. Open `tribev2/brain_states.py`
2. Locate the `compute_normalized_scores` function.
3. Add your logic where `valence` and `learning` are defined. For example, a "Sensory Overload" score:
   ```python
   sensory_overload = (
       scores.get("visual", 0.5) 
       + scores.get("temporal", 0.5) # Auditory proxy
       + scores.get("insula", 0.5)   # Salience proxy
   ) / 3
   ```
4. Add it to the return dictionary.
5. Update the `analyze()` method dict documentation.

## Modifying Hardware / MPS Rules

If running on new hardware (e.g., an M4 Max with 64GB RAM, or an NVIDIA setup):

- To disable all MPS patching, simply remove `import tribev2._mps_compat` from `tribev2/__init__.py`.
- If memory is no longer constrained, you can disable `float16` by modifying `tribev2/demo_utils.py` (line 246): remove `model.half()`, and edit `_mps_compat.py` (line 150) to load LLaMA in `float32` or `bfloat16`.
- If PyTorch ever fixes the MPS kernel for Grouped Query Attention, you can remove the `HuggingFaceText` specific bypass in `_mps_compat.py` `_hf_model_post_init` (lines 117-122).
