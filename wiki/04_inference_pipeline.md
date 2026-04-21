# Inference Pipeline

## How to Generate Brain Predictions

This page explains the exact steps to go from "I have a piece of text" to "I have brain activation scores."

## Step 1: Load the Model

```python
from tribev2 import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2")
```

### What `from_pretrained` Does Internally

1. Downloads `config.yaml` and `best.ckpt` from HuggingFace Hub (cached after first run)
2. Parses the YAML config into a `ConfDict` (a nested dictionary with dot-notation access)
3. Overrides cluster-specific settings for local use (worker count, cache paths)
4. Creates a `TribeModel` instance (subclass of `TribeExperiment`)
5. Loads the checkpoint weights into the `FmriEncoderModel`
6. Moves the model to the best available device (`mps` on Mac, `cuda` on NVIDIA)
7. Converts to float16 and sets to eval mode

**Source**: `tribev2/demo_utils.py`, `TribeModel.from_pretrained()` (line 150)

### Config Overrides

You can override config values at load time:

```python
model = TribeModel.from_pretrained(
    "facebook/tribev2",
    config_update={
        "data.text_feature.model_name": "alpindale/Llama-3.2-3B",  # Alternative Llama source
        "data.text_feature.device": "cpu",  # Force text to CPU
    },
)
```

## Step 2: Build the Events DataFrame

```python
# From text
df = model.get_events_dataframe(text_path="my_text.txt")

# From audio
df = model.get_events_dataframe(audio_path="speech.wav")

# From video
df = model.get_events_dataframe(video_path="clip.mp4")
```

### What Happens for Text Input

The text goes through this pipeline:

```
1. Read text file → "She opened the letter and tears of joy..."
2. Detect language → "en"
3. Google TTS → generates audio.mp3
4. ExtractAudioFromVideo → (no-op for audio)
5. ChunkEvents → splits long audio into 30-60 second chunks
6. ExtractWordsFromAudio (WhisperX) → word-level transcription
   Output: [("She", 0.0, 0.3), ("opened", 0.3, 0.6), ...]
7. AddText → attaches text to each word event
8. AddSentenceToWords → groups words into sentences
9. AddContextToWords → adds surrounding text context (up to 1024 chars)
10. RemoveMissing → drops events with missing data
```

The result is a DataFrame with rows like:

```
type=Word, word="opened", start=0.3, duration=0.3, context="She opened the letter..."
type=Audio, filepath="audio.mp3", start=0.0, duration=5.2
```

**Source**: `tribev2/demo_utils.py`, `get_audio_and_text_events()` (line 66) and `TextToEvents` (line 98)

### What Happens for Video Input

```
1. ExtractAudioFromVideo → extracts audio.wav from the video file
2. ChunkEvents → splits Audio and Video events into 30-60s chunks
3. ExtractWordsFromAudio (WhisperX) → transcribes the extracted audio
4. AddText, AddSentenceToWords, AddContextToWords, RemoveMissing
```

The DataFrame has Video events, Audio events, and Word events.

### What Happens for Audio-Only Input

Same as video but without the video extraction step. Audio events and Word events only.

## Step 3: Run Prediction

```python
preds, segments = model.predict(events=df)
```

### What `predict()` Does Internally

1. **Creates a DataLoader** from the events DataFrame
   - The `Data.get_loaders()` method in `tribev2/main.py` does this
   - It passes events through the feature extractors (V-JEPA, Wav2Vec, LLaMA)
   - Feature extraction results are cached to disk in `./cache/`
   - After each extractor finishes, its GPU model is freed from memory

2. **Segments the data** into chunks of `duration_trs` timesteps (default: 40)
   - Each segment is split into individual TRs (1-second windows)
   - Empty segments (no events in that time window) are removed by default

3. **Runs the Transformer** on each batch
   - The model forward pass: `model(batch)` → `(batch, n_vertices, n_timesteps)`
   - Output is rearranged to `(batch * timesteps, n_vertices)` and filtered by the `keep` mask

4. **Concatenates all predictions** into a final numpy array

**Source**: `tribev2/demo_utils.py`, `TribeModel.predict()` (line 330)

### Output Shape

```python
preds.shape  # (n_kept_segments, 20484)
# n_kept_segments = number of 1-second time windows with at least one event
# 20484 = 10242 (left hemisphere) + 10242 (right hemisphere)
```

### The Segments List

`segments` is a list of segment objects aligned with `preds`. Each segment has:
- `.start` — start time in seconds
- `.duration` — always 1.0 (one TR)
- `.ns_events` — the events that fall within this time window

## Step 4: Analyze the Predictions

### Option A: Brain Surface Visualization

```python
from tribev2.plotting import PlotBrain

plotter = PlotBrain(mesh="fsaverage5")
fig = plotter.plot_brain(preds.mean(axis=0), cmap="fire")
```

### Option B: Region-Level Analysis

```python
from tribev2.brain_states import BrainAtlas, compute_normalized_scores

atlas = BrainAtlas()
result = compute_normalized_scores(preds, atlas)
# result["scores"] → {"prefrontal": 0.62, "insula": 0.45, ...}
# result["valence"] → 0.03 (positive = happy)
# result["learning"] → 0.58 (higher = more engaged)
```

### Option C: Compare to Reference Profiles

```python
from tribev2.brain_states import BrainStateProfiler, BrainStateClassifier

profiler = BrainStateProfiler(model=model, stimuli_dir="./stimuli")
profiles = profiler.build_profiles(states=["happy", "sad"])
classifier = BrainStateClassifier(profiles=profiles, atlas=atlas)
analysis = classifier.analyze(preds)
# analysis["classification"] → [("happy", 0.85), ("sad", 0.42)]
```

## Caching

Feature extraction is the slowest part (2-3 minutes per stimulus). Results are cached to disk:

```
./cache/
├── <hash1>/          ← Cached V-JEPA features for a specific video
├── <hash2>/          ← Cached Wav2Vec features for a specific audio
├── <hash3>/          ← Cached LLaMA features for a specific text
└── brain_states/     ← Cached brain state profiles (.npy files)
    ├── happy_0.npy
    ├── happy_1.npy
    ├── sad_0.npy
    └── ...
```

On subsequent runs with the same input, feature extraction is skipped entirely.

## Performance on Mac Mini M4 (16 GB)

| Step | Time | Notes |
|------|------|-------|
| Model loading | ~15s | First run downloads ~1 GB from HuggingFace |
| Text → events | ~30-60s | gTTS + WhisperX transcription |
| Feature extraction (first run) | ~2-3 min | V-JEPA + Wav2Vec + LLaMA, sequential |
| Feature extraction (cached) | ~5s | Just loads from disk |
| Transformer inference | ~1-2s | Very fast (small model) |
| **Total (first run)** | **~3-5 min** | Per stimulus |
| **Total (cached)** | **~5-10s** | Per stimulus |
