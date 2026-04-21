# Model Architecture

## Overview

TRIBE v2 is a three-stage pipeline:

```
Input Media ──→ Feature Extractors ──→ Fusion Transformer ──→ Brain Predictions
  (video,         (V-JEPA, Wav2Vec,       (256-dim hidden,        (20,484 vertices,
   audio,          LLaMA 3.2)             attention-based)         1 per second)
   text)
```

This page explains each stage in detail, with references to the exact source files.

## Stage 1: Feature Extraction

Before the media reaches the TRIBE v2 model itself, pretrained AI models extract rich features from each modality. These are the same kinds of models used in computer vision, speech recognition, and NLP.

### Video Features — V-JEPA2 (ViT-G)

- **What it is**: A Vision Transformer (ViT) with approximately 2 billion parameters, pretrained on video using the Joint Embedding Predictive Architecture (JEPA).
- **What it does**: Takes video frames and produces a sequence of high-dimensional feature vectors that encode visual content — objects, scenes, motion, faces, colors.
- **Output shape**: A tensor of shape `(batch, n_layers, feature_dim, n_timesteps)`.
- **Configured by**: `data.video_feature` in the YAML config (a `HuggingFaceVideo` extractor from `neuralset`).

### Audio Features — Wav2Vec-BERT 2.0

- **What it is**: A speech representation model that processes raw audio waveforms.
- **What it does**: Converts audio into feature vectors that encode speech sounds, music, prosody (tone of voice), and other acoustic properties.
- **Output shape**: Same tensor structure as video.
- **Configured by**: `data.audio_feature` in the YAML config.

### Text Features — LLaMA 3.2-3B

- **What it is**: A 3-billion-parameter language model from Meta.
- **What it does**: Takes word-level text (with timing information from transcription) and produces contextual embeddings that encode semantic meaning.
- **Output shape**: Same tensor structure.
- **Configured by**: `data.text_feature` in the YAML config.
- **Special note**: On Apple Silicon, this runs on **CPU only** due to MPS incompatibility with Grouped Query Attention. See `03_apple_silicon.md` for details.

### How Text Input Works

When you feed text (not audio or video), the pipeline does extra work:

```
Raw Text ("Hello world")
    ↓
Google TTS (gTTS) → generates audio file (audio.mp3)
    ↓
WhisperX → transcribes back to word-level events with timestamps
    ↓
Events DataFrame with precise word timing
    ↓
LLaMA 3.2 processes the text semantics
Wav2Vec processes the generated audio
```

This round-trip (text → speech → transcription) might seem redundant, but it gives the model the temporal structure it needs — each word has a specific start time and duration, which matters for the brain prediction.

**Source**: `tribev2/demo_utils.py`, class `TextToEvents` (line 98) and function `get_audio_and_text_events` (line 66).

## Stage 2: The Fusion Transformer

The core TRIBE v2 model is defined in `tribev2/model.py`, class `FmriEncoderModel`.

### Input Aggregation

The three feature streams (video, audio, text) are first projected to a common dimension and then combined:

```python
# Simplified from model.py, aggregate_features()
for modality in ['video', 'audio', 'text']:
    features = batch.data[modality]                # (B, Layers, Dim, Time)
    features = rearrange(features, 'b l d t -> b (l d) t')  # Concatenate layers
    features = features.transpose(1, 2)            # (B, Time, LayerDim)
    projected = self.projectors[modality](features) # (B, Time, Hidden//3)
    tensors.append(projected)

combined = torch.cat(tensors, dim=-1)  # (B, Time, Hidden)
```

Key design choices:
- **`extractor_aggregation: "cat"`** — Features from different modalities are concatenated, not summed. Each modality gets `hidden//3` dimensions (≈85 each for hidden=256).
- **`layer_aggregation: "cat"`** — Multiple layers from each feature extractor are concatenated, giving the model access to both low-level and high-level features.
- **Modality dropout** — During training, entire modalities were randomly zeroed out (`modality_dropout`) so the model learned to work with any subset of inputs.

### Transformer Encoder

After aggregation, the combined features pass through a standard Transformer encoder:

```python
# Simplified from model.py, transformer_forward()
x = self.combiner(x)                    # MLP: input_dim → hidden (256)
x = x + self.time_pos_embed[:, :T]      # Add positional encoding
x = self.encoder(x)                      # Multi-head self-attention + FFN
```

This is where the model learns cross-modal relationships — how visual content relates to what's being said, how the audio tone relates to the text meaning, etc.

### Configuration

Key hyperparameters (from the pretrained config):
- `hidden: 256` — Transformer hidden dimension
- `max_seq_len: 1024` — Maximum sequence length
- `time_pos_embedding: true` — Learnable positional embeddings
- `subject_layers` — Per-subject output layers (uses average-subject in demo mode)

## Stage 3: Brain Prediction

The Transformer output is mapped to brain vertices:

```python
# Simplified from model.py, forward()
x = self.encoder(x)           # (B, Time, Hidden)  →  (B, Hidden, Time)
x = self.predictor(x)         # Subject-specific linear layer → (B, 20484, Time)
x = self.pooler(x)            # AdaptiveAvgPool1d → (B, 20484, n_output_timesteps)
```

- **Subject layers** (`SubjectLayers`): The final prediction layer has separate weights per subject. In average-subject mode (which we use for inference), it uses averaged weights across all training subjects.
- **Temporal smoothing**: An optional 1D Gaussian convolution smooths the temporal dimension.
- **Pooler**: An adaptive average pool aligns the variable-length Transformer output to the desired number of output timesteps.

## Data Flow Summary

```
Video file ──→ V-JEPA2    ─┐
                            │   cat    ┌─────────────┐     ┌────────────┐
Audio file ──→ Wav2Vec    ──┼────────→ │ Transformer │ ──→ │ Subject    │ ──→ (B, 20484, T)
                            │          │ Encoder     │     │ Predictor  │     Brain predictions
Text file  ──→ LLaMA 3.2 ──┘          └─────────────┘     └────────────┘
```

## Key Source Files

| File | Role |
|------|------|
| `tribev2/model.py` | `FmriEncoder` config + `FmriEncoderModel` — the Transformer model |
| `tribev2/main.py` | `Data` class (loader setup), `TribeExperiment` (training pipeline) |
| `tribev2/demo_utils.py` | `TribeModel` (inference wrapper), event DataFrame construction |
| `tribev2/utils_fmri.py` | `TribeSurfaceProjector` — projecting volumetric data to cortical surface |
| `tribev2/pl_module.py` | PyTorch Lightning module for training |

## The Events DataFrame

Before features can be extracted, the input must be converted to a **pandas DataFrame** called the "events DataFrame." This is the central data structure that drives the entire pipeline.

Each row represents one event (a word, an audio segment, a video segment):

| Column | Type | Description |
|--------|------|-------------|
| `type` | str | Event type: `"Audio"`, `"Video"`, `"Word"`, `"Sentence"`, `"Text"` |
| `start` | float | Start time in seconds |
| `duration` | float | Duration in seconds |
| `stop` | float | End time (start + duration) |
| `filepath` | str | Path to the media file |
| `timeline` | str | Timeline name (groups related events) |
| `subject` | str | Subject identifier (always `"default"` in demo mode) |
| `word` | str | The word text (for Word events) |
| `context` | str | Surrounding text context (for Word events) |

The `get_events_dataframe()` method in `TribeModel` handles all the complexity of:
1. Converting text → speech → word events (using gTTS + WhisperX)
2. Extracting audio from video files
3. Chunking long audio/video into manageable segments
4. Adding sentence and context annotations to word events
