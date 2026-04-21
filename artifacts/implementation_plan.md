# Brain State Analysis Tool for TRIBE v2

Build a practical Python module and notebook that derives emotional states (happy, sad, calm, fearful) and cognitive states (learning-ready, curious, disengaged) from TRIBE v2's predicted fMRI brain activity.

## User Review Required

> [!IMPORTANT]
> This builds on top of your existing TRIBE v2 setup. The tool creates reference "brain signatures" by running the model on curated text stimuli, then uses those signatures to classify new inputs.

> [!WARNING]
> All emotion/cognition labels are **approximations** based on cortical surface patterns. Subcortical structures (amygdala, hippocampus) are not represented in fsaverage5. This is NOT suitable for clinical use.

---

## Proposed Changes

### Brain State Analysis Module

#### [NEW] [brain_states.py](file:///Volumes/dev/ai_work/meta/tribe_v2/code/tribev2/brain_states.py)

Core analysis module with these components:

1. **`BrainAtlas` class** — Wraps nilearn's Destrieux atlas for fsaverage5, maps vertices → named brain regions, extracts region-level activation from TRIBE v2 predictions
2. **`BrainStateProfiler` class** — Runs TRIBE v2 on curated stimulus sets per state category, averages predictions to create reference "brain signatures," computes differential maps between states
3. **`BrainStateClassifier` class** — Given a novel stimulus's brain prediction, correlates it against reference profiles to rank most-likely emotional/cognitive states, also outputs ROI-level scores for interpretability
4. **Composite score functions** — `emotion_valence_score()` (positive vs negative), `learning_readiness_score()` (frontal+temporal engagement vs passive visual), `attention_score()` (dorsal attention network activation)

---

#### [NEW] [brain_state_demo.ipynb](file:///Volumes/dev/ai_work/meta/tribe_v2/code/brain_state_demo.ipynb)

Interactive notebook demonstrating:

1. Load TRIBE v2 model (reuses existing setup from `tribe_demo.ipynb`)
2. Build reference profiles for 5 states: Happy, Sad, Calm, Curious/Learning, Fearful
3. Visualize differential brain maps (e.g., Happy vs Sad on cortical surface)
4. Run a novel text stimulus and classify its brain state
5. Display ROI-level bar charts and radar chart for interpretability

---

### Curated Stimulus Library

#### [NEW] [stimuli/](file:///Volumes/dev/ai_work/meta/tribe_v2/code/stimuli/)

Text files for each emotional/cognitive category:

- `stimuli/happy.txt` — 5 joyful/uplifting passages
- `stimuli/sad.txt` — 5 melancholic/grief passages  
- `stimuli/calm.txt` — 5 peaceful/relaxing passages
- `stimuli/curious.txt` — 5 mystery/educational passages
- `stimuli/fearful.txt` — 5 suspenseful/threatening passages

Each file contains multiple short paragraphs (2-3 sentences each) separated by blank lines.

---

## Open Questions

> [!IMPORTANT]
> **Scope question**: Should I start with just the core Python module (`brain_states.py`) and a simple demo notebook? Or do you want the full visualization dashboard from the start?

> [!NOTE]
> **Performance**: Each text stimulus requires running TRIBE v2's full pipeline (TTS → WhisperX → feature extraction → Transformer). On your Mac Mini M4, each prediction takes ~2-3 minutes. Building 5 reference profiles × 5 stimuli each = ~50-75 minutes of initial computation. These results can be cached for future use.

---

## Verification Plan

### Automated Tests
- Run the module on at least 2 stimulus categories and verify differential maps are non-zero
- Verify ROI extraction produces correct number of regions (74 per hemisphere for Destrieux)
- Verify classifier ranks the correct state highest when tested on training stimuli (sanity check)

### Manual Verification
- Visual inspection of brain surface heatmaps — do Happy vs Sad differences appear in expected regions (PFC, insula, temporal pole)?
- Does the "learning readiness" score rank educational content higher than passive/monotone content?
